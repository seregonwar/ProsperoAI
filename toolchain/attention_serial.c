/*
 * ProsperoAI — Phase 2 differential: serial causal attention pipeline
 * (spec §6) vs the pai_ref_attention_f32 oracle.
 *
 * Every primitive in the plan is HW-validated on 9.40: G40 gemv,
 * G42 mul/add, G72 exp-2^x (host pre-scale ×log2e), G75 v_rcp, G76
 * v_max. This harness executes the exact §6 dispatch sequence
 * host-side (through the mirror ABIs) and cross-checks the assembled
 * attention output against the reference oracle — the pre-flight for
 * Seat A's on-GPU attention kernel, exactly like ropegen_diff was the
 * pre-flight for the on-GPU RoPE tables.
 *
 * Pipeline per (row p, head hh), KV head kh = (hh*HK)/H, hd = D/H:
 *   1. scores[t] = q[p,hh] · k[t,kh]        (G40, groups = p+1, W =
 *      K-cache rows repacked)
 *   2. scores[t] *= 1/sqrt(hd)              (scale — softmax is NOT
 *      scale-invariant)
 *   3. m = max(scores)                      (serial reduce over v_max)
 *   4. e[t] = 2^((scores[t]-m)*log2e)       (G72 + host pre-scale)
 *   5. sum + rcp(sum)                       (G75)
 *   6. soft[t] = e[t] * rcp                 (mul)
 *   7. out[d] = sum_t soft[t] * v[t,kh][d]  (G40, groups = hd, W =
 *      V-cache rows transposed)
 *
 * Build: host-tests/host-reference preset; binary `attention_serial`.
 */

#include <hal/host_kernels.h>
#include <ref_ops.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AT_H     2u
#define AT_HK    1u
#define AT_HD    4u
#define AT_D     (AT_H * AT_HD) /* 8 */
#define AT_SEQ   6u
#define AT_R2    2u
#define AT_TOL   1e-4f
#define AT_LOG2E 1.4426950408889634

static int g_failures;

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

/* G40 fgemv-ABI call (decoder_test mirror_gemv_g40 pattern): header
 * [K, pad, x_lo, x_hi, W...inline at hdr+4], y at ud[4:5], one group
 * per row; hdr[4+g*K+k] = W[k*N+g] (W transposed). */
static pai_status_t
at_gemv_g40(uint32_t ud[16], uint32_t h40[], const float *x,
            const float *w, float *y, uint32_t nout, uint32_t kin) {
  uint32_t gg, kk;
  h40[0] = kin;
  h40[1] = 0;
  h40[2] = (uint32_t)(uintptr_t)x;
  h40[3] = (uint32_t)((uintptr_t)x >> 32);
  for (gg = 0; gg < nout; gg++) {
    for (kk = 0; kk < kin; kk++) {
      float wv = w[kk * nout + gg];
      memcpy(&h40[4 + gg * kin + kk], &wv, 4);
    }
  }
  ud[2] = (uint32_t)(uintptr_t)h40;
  ud[3] = (uint32_t)((uintptr_t)h40 >> 32);
  ud[4] = (uint32_t)(uintptr_t)y;
  ud[5] = (uint32_t)((uintptr_t)y >> 32);
  return pai_host_kernel_fgemv(NULL, ud, 1, nout);
}

int
main(void) {
  static float q[AT_SEQ * AT_D], k[AT_SEQ * AT_D], v[AT_SEQ * AT_D];
  static float out[AT_SEQ * AT_D], ref[AT_SEQ * AT_D];
  uint32_t ud[16] = {0};
  static uint32_t h40[4 + 2 * 64]; /* scores: p+1<=6 rows x hd=4 */
  float scores[AT_SEQ], soft[AT_SEQ];
  double max_d = 0.0;
  pai_status_t st;
  uint32_t p, hh, d, t;

  printf("== Phase 2 differential: serial causal attention (spec 6) ==\n");

  /* deterministic Q/K/V (position-major [seq][D], head blocks) */
  for (uint32_t i = 0; i < AT_SEQ * AT_D; i++) {
    q[i] = (float)(sin(0.13 * (double)i) * 0.7);
    k[i] = (float)(cos(0.09 * (double)i) * 0.7);
    v[i] = (float)(0.2 * (double)(i % 9) - 0.8);
  }

  /* oracle: full causal GQA attention */
  st = pai_ref_attention_f32(AT_H, AT_HK, AT_SEQ, AT_HD, q, k, v, ref);
  REQUIRE(st == PAI_OK);

  /* the §6 pipeline */
  memset(out, 0, sizeof(out));
  for (p = 0; p < AT_SEQ; p++) {
    for (hh = 0; hh < AT_H; hh++) {
      uint32_t kh = (hh * AT_HK) / AT_H; /* 0 for HK=1 */
      const float *qp = q + (p * AT_H + hh) * AT_HD;
      float inv = 1.0f / sqrtf((float)AT_HD);
      float m, sum = 0.0f, rc = 0.0f;

      /* 1) scores via G40: W rows = k[t, kh] for t = 0..p. The helper
       * reads w[kk*nout+gg], so the buffer must be TRANSPOSED: row g
       * (=t) of the header holds k_t, i.e. wbuf[d*(p+1)+t] = k[t][d]. */
      {
        float wbuf[AT_HD * AT_SEQ];
        for (t = 0; t <= p; t++) {
          for (d = 0; d < AT_HD; d++) {
            wbuf[d * (p + 1) + t] = k[(t * AT_HK + kh) * AT_HD + d];
          }
        }
        st = at_gemv_g40(ud, h40, qp, wbuf, scores, p + 1, AT_HD);
        REQUIRE(st == PAI_OK);
      }
      /* 2) scale */
      for (t = 0; t <= p; t++) {
        scores[t] *= inv;
      }
      /* 3) row max (serial reduce over v_max) */
      m = scores[0];
      for (t = 1; t <= p; t++) {
        if (scores[t] > m) {
          m = scores[t];
        }
      }
      /* 4) e = 2^((s-m)*log2e) (G72 + host prescale) */
      for (t = 0; t <= p; t++) {
        soft[t] = exp2f((scores[t] - m) * (float)AT_LOG2E);
        sum += soft[t];
      }
      /* 5) rcp via G75 mirror */
      {
        uint32_t h1[4] = {1, 0, 0, 0};
        memcpy(&h1[2], &sum, 4);
        ud[2] = (uint32_t)(uintptr_t)h1;
        ud[3] = (uint32_t)((uintptr_t)h1 >> 32);
        ud[4] = (uint32_t)(uintptr_t)&rc;
        ud[5] = (uint32_t)((uintptr_t)&rc >> 32);
        st = pai_host_kernel_nlexp_rcp(NULL, ud, 1, 1);
        REQUIRE(st == PAI_OK);
      }
      /* 6) soft = e * rcp */
      for (t = 0; t <= p; t++) {
        soft[t] *= rc;
      }
      /* 7) out via G40: y[d] = sum_t soft[t] * v[t,kh][d]. The helper
       * reads w[kk*nout+gg] = w[t*hd+d], so the buffer is row-major:
       * wbuf[t*HD+d] = v[t][d]. */
      {
        float wbuf[AT_SEQ * AT_HD];
        for (t = 0; t <= p; t++) {
          for (d = 0; d < AT_HD; d++) {
            wbuf[t * AT_HD + d] = v[(t * AT_HK + kh) * AT_HD + d];
          }
        }
        st = at_gemv_g40(ud, h40, soft, wbuf, out + (p * AT_H + hh) * AT_HD,
                         AT_HD, p + 1);
        REQUIRE(st == PAI_OK);
      }
    }
  }

  /* differential vs oracle */
  {
    uint64_t mism = 0;
    st = pai_ref_compare_f32(out, ref, AT_SEQ * AT_D, AT_TOL, AT_TOL, &mism);
    if (st == PAI_OK) {
      printf("  PASS serial attention pipeline == pai_ref_attention_f32 "
             "(H=%u HK=%u HD=%u seq=%u)\\n", AT_H, AT_HK, AT_HD, AT_SEQ);
    } else {
      printf("  FAIL at e=%llu (got %.7f want %.7f)\\n",
             (unsigned long long)mism, (double)out[mism],
             (double)ref[mism]);
      g_failures++;
    }
  }

  /* row-norm sanity: every output element within v-range (convex combo) */
  for (uint32_t i = 0; i < AT_SEQ * AT_D; i++) {
    double dabs = fabs((double)out[i]);
    if (dabs > max_d) {
      max_d = dabs;
    }
  }
  if (max_d <= 1.0 + 1e-3) {
    printf("  PASS outputs within v-range (max |out| = %g)\\n", max_d);
  } else {
    printf("  FAIL |out| = %g out of v-range\\n", max_d);
    g_failures++;
  }

  printf("== attention_serial %s ==\n", g_failures ? "FAILED" : "PASSED");
  return g_failures ? 1 : 0;
}
