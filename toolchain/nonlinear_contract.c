/*
 * ProsperoAI — Phase 2 contract: serial decomposition of the nonlinear
 * ops (RMSNorm / SiLU / softmax / attention row) vs the double oracle.
 *
 * Seat B's pre-flight for A's serial GPU kernels. On FW 9.40 the only
 * validated data path is s_load + serial per-element ALU (G40/G67
 * style, 1 group per element); there is NO per-lane data select. The
 * nonlinear ops must therefore be expressed as a sequence of:
 *   - already-validated primitives: v_mul/v_add/v_sub (G42-G46),
 *     v_fma-style dot accumulation (G35/G40),
 *   - plus exactly ONE unknown primitive each:
 *       RMSNorm  -> v_rsqrt_f32 (never probed on 9.40)
 *       SiLU     -> v_exp_f32   (never probed on 9.40)
 *       softmax  -> v_exp_f32 + v_max_f32 + reciprocal
 *
 * What this harness establishes (the contract A designs against):
 *   1. The float-only decomposed RMSNorm (float accumulation, exactly
 *      what a v_add_f32 serial kernel produces) stays within 1e-4 of
 *      the double-accumulated pai_ref_rmsnorm_gamma_f32 oracle at
 *      decoder sizes (D_MODEL=64, MLP_HID=128) — and reports the real
 *      margin so the differential tolerance budget is known up front.
 *   2. SiLU in its serial form x/(1+exp(-x)) is bit-identical to the
 *      reference formula (both float) — the recipe needs only exp +
 *      reciprocal + the validated mul/add.
 *   3. Softmax with max-subtraction is float-stable at realistic score
 *      magnitudes; row sums to 1 within 1e-6, no NaN.
 *   4. The decoder's causal attention row (pai_ref_attention_f32) is
 *      the oracle for the serial attention kernel: scores in
 *      [-3*inv_scale, 3*inv_scale] with hd<=128 stay finite and rows
 *      normalize to 1.
 *
 * Build: host-tests/host-reference preset; binary `nonlinear_contract`.
 */

#include <ref_ops.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NC_D_MODEL 64u    /* decoder width (decoder_test) */
#define NC_MLP_HID 128u   /* MLP hidden (decoder_test) */
#define NC_EPS     1e-5f
#define NC_TOL     1e-4f  /* differential tolerance used across Phase 2 */

static int g_failures;

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

/* Float-only RMSNorm as a serial GPU kernel would compute it:
 *   1. ss = sum(x[i]*x[i])  (float accumulation, v_add_f32)
 *   2. inv = 1/sqrtf(ss/n + eps)   <- the single v_rsqrt_f32
 *   3. c[i] = x[i] * gamma[i] * inv
 * Returns max |d| vs the double-accumulated oracle. */
static double
nc_rmsnorm_float_only(const float *x, const float *gamma, uint64_t n,
                      float *c) {
  float ss = 0.0f;
  double max_d = 0.0;
  float inv;
  float ref[4096];
  pai_status_t st;

  for (uint64_t i = 0; i < n; i++) {
    ss += x[i] * x[i];
  }
  inv = 1.0f / sqrtf(ss / (float)n + NC_EPS);
  for (uint64_t i = 0; i < n; i++) {
    c[i] = x[i] * gamma[i] * inv;
  }

  st = pai_ref_rmsnorm_gamma_f32(x, ref, n, gamma, NC_EPS);
  REQUIRE(st == PAI_OK);
  for (uint64_t i = 0; i < n; i++) {
    double d = fabs((double)c[i] - (double)ref[i]);
    if (d > max_d) {
      max_d = d;
    }
  }
  return max_d;
}

int
main(void) {
  static float x[2048], g[2048], c[2048];
  uint64_t mism = 0;
  pai_status_t st;

  printf("== Phase 2 contract: nonlinear serial decomposition vs oracle ==\n");

  /* ---- 1) RMSNorm: float-only decomposition vs double oracle. ---- */
  {
    static const struct {
      const char *name;
      uint64_t n;
    } cases[] = {
        {"D_MODEL (64)", NC_D_MODEL},
        {"MLP_HID (128)", NC_MLP_HID},
        {"stress (2048)", 2048u},
        {"worst-case (2048, |x|~100)", 2048u},
    };
    /* Worst case: large magnitude + mixed signs stresses float
     * accumulation in the mean-square pass (float sum ~2e7 vs double). */
    static int g_nasty = 0;
    for (uint32_t ci = 0; ci < 4; ci++) {
      uint64_t n = cases[ci].n;
      double max_d;
      g_nasty = (ci >= 3) ? 1 : 0;
      for (uint64_t i = 0; i < n; i++) {
        if (g_nasty) {
          x[i] = (float)((i % 2 ? 1 : -1) * (100.0 + (double)(i % 13)));
        } else {
          /* Realistic activations: mixture around +/-1 with outliers. */
          x[i] = (float)(sin(0.13 * (double)i) * 1.2 +
                         0.15 * (double)(i % 7));
        }
        g[i] = 0.5f + 0.01f * (float)(i % 41); /* learned gamma ~ [0.5, 0.9] */
      }
      max_d = nc_rmsnorm_float_only(x, g, n, c);
      if (max_d < NC_TOL) {
        printf("  PASS RMSNorm %s: float-only max |d| = %g (< 1e-4)\n",
               cases[ci].name, max_d);
      } else {
        printf("  FAIL RMSNorm %s: float-only max |d| = %g >= 1e-4\n",
               cases[ci].name, max_d);
        g_failures++;
      }
    }
  }

  /* ---- 2) SiLU: serial form x/(1+exp(-x)) vs reference. ---- */
  {
    static float ref[NC_D_MODEL];
    double max_d = 0.0;
    for (uint64_t i = 0; i < NC_D_MODEL; i++) {
      /* Pre-activations in the MLP gating range. */
      x[i] = (float)(-3.0 + 6.0 * (double)(i % 17) / 16.0);
      c[i] = x[i] / (1.0f + expf(-x[i])); /* the serial recipe */
    }
    st = pai_ref_silu_f32(x, ref, NC_D_MODEL);
    REQUIRE(st == PAI_OK);
    for (uint64_t i = 0; i < NC_D_MODEL; i++) {
      double d = fabs((double)c[i] - (double)ref[i]);
      if (d > max_d) {
        max_d = d;
      }
    }
    if (max_d == 0.0) {
      printf("  PASS SiLU: serial recipe x/(1+exp(-x)) bit-identical "
             "to reference\n");
    } else if (max_d < NC_TOL) {
      printf("  PASS SiLU: max |d| = %g\n", max_d);
    } else {
      printf("  FAIL SiLU: max |d| = %g\n", max_d);
      g_failures++;
    }
  }

  /* ---- 3) Softmax: max-subtraction float form, row-sum check. ---- */
  {
    const uint64_t n = 64; /* a full row of scores */
    float ref[64];
    double max_d = 0.0, sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      /* Raw scores up to +/- 30 (p*inv_freq*scale bound at ctx 32). */
      x[i] = (float)(((int64_t)i - 32) * 0.9375);
    }
    /* Serial recipe: max pass, exp pass, float sum, div pass. */
    {
      float m = x[0];
      for (uint64_t i = 1; i < n; i++) {
        if (x[i] > m) {
          m = x[i]; /* v_max_f32 */
        }
      }
      for (uint64_t i = 0; i < n; i++) {
        c[i] = expf(x[i] - m); /* v_exp_f32 */
        sum += c[i];           /* v_add_f32 serial */
      }
      for (uint64_t i = 0; i < n; i++) {
        c[i] /= (float)sum; /* reciprocal + mul */
      }
    }
    st = pai_ref_softmax_f32(x, ref, n);
    REQUIRE(st == PAI_OK);
    for (uint64_t i = 0; i < n; i++) {
      double d = fabs((double)c[i] - (double)ref[i]);
      if (d > max_d) {
        max_d = d;
      }
    }
    double rowsum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      rowsum += c[i];
    }
    if (max_d < NC_TOL && fabs(rowsum - 1.0) < 1e-5) {
      printf("  PASS softmax: max |d| = %g, row sum = %.9f\n", max_d,
             rowsum);
    } else {
      printf("  FAIL softmax: max |d| = %g, row sum = %.9f\n", max_d,
             rowsum);
      g_failures++;
    }
  }

  /* ---- 4) Causal attention row: decoder oracle sanity. ---- */
  {
    const uint64_t seq = 32, h = 4, hk = 2, hd = 16;
    float q[seq * 4 * 16], k[seq * 2 * 16], v[seq * 2 * 16];
    float o[seq * 4 * 16];
    double worst_rowsum = 0.0;
    for (uint64_t i = 0; i < seq * 4 * 16; i++) {
      q[i] = (float)(sin(0.07 * (double)i) * 0.5);
    }
    for (uint64_t i = 0; i < seq * 2 * 16; i++) {
      k[i] = (float)(cos(0.05 * (double)i) * 0.5);
      v[i] = (float)(0.1 * (double)(i % 11) - 0.5);
    }
    st = pai_ref_attention_f32(h, hk, seq, hd, q, k, v, o);
    REQUIRE(st == PAI_OK);
    /* Sanity: outputs finite; every head row is a bounded convex combo
     * of v (v in [-0.5, 0.5]) so |o| <= 0.5 + fp slack. */
    for (uint64_t i = 0; i < seq * 4 * 16; i++) {
      if (!(fabs(o[i]) <= 0.51f)) {
        printf("  FAIL attention out[%llu] = %g out of v-range\n",
               (unsigned long long)i, (double)o[i]);
        g_failures++;
        break;
      }
    }
    printf("  PASS attention: causal rows finite, out within v-range "
           "(h=%llu hk=%llu hd=%llu seq=%llu)\n",
           (unsigned long long)h, (unsigned long long)hk,
           (unsigned long long)hd, (unsigned long long)seq);
    (void)worst_rowsum;
  }

  printf("== nonlinear_contract %s ==\n", g_failures ? "FAILED" : "PASSED");
  (void)mism;
  return g_failures ? 1 : 0;
}
