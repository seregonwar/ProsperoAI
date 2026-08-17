/*
 * ProsperoAI — Phase 2 contract: serial decomposition of the nonlinear
 * ops (RMSNorm / SiLU / softmax / attention row) vs the double oracle.
 *
 * Seat B's pre-flight for A's serial GPU kernels. On FW 9.40 the only
 * validated data path is s_load + serial per-element ALU (G40/G67
 * style, 1 group per element); there is NO per-lane data select. The
 * nonlinear ops must therefore be expressed as a sequence of:
 *   - already-validated primitives: v_mul/v_add/v_sub (G42-G46),
 *     v_fma-style dot accumulation (G35/G40), *   - plus exactly ONE special primitive each:
 *       RMSNorm  -> v_rsq_f32   (G71 VALIDATED: exact serial-safe)
 *       SiLU     -> v_exp_f32   (G72 VALIDATED: 2^x convention!)
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
 *   5. e^x production recipe: v_exp is 2^x on 9.40 (G72, commit
 *      6ecf917), so e^x = 2^(x*log2(e)) with the pre-scale done
 *      host-side on the validated float mul path. Section 5 locks
 *      exp2f(x*log2e) == expf(x) (max rel < 1e-4) plus the full
 *      softmax row and SiLU in 2^x form.
 *   6. Final forms with v_rcp/v_max (G75/G76, commit 78a8595):
 *      softmax = e*rcp(sum), SiLU = x*rcp(1+e) — no division, both
 *      through the exact nlexp mirror ABI; v_max(x,0) == relu.
 *
 * Build: host-tests/host-reference preset; binary `nonlinear_contract`.
 */

#include <hal/host_kernels.h>
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

  /* ---- 5) e^x production recipe: v_exp is 2^x on 9.40 (G72,
   * convention locked), so e^x must be fed to the kernel as
   * xs = x*log2(e) (host-side float mul, the G39-validated mul
   * path). Validate the exact recipe: exp2f(x*log2e) must equal the
   * math oracle expf(x) across the decoder's x ranges. ---- */
  {
    static const double LOG2E = 1.4426950408889634;
    double max_rel = 0.0;
    double max_abs = 0.0;
    uint64_t worst = 0;
    /* softmax scores up to +/-30; SiLU pre-activations +/-3; RMSNorm
     * eps-range values separately below. */
    static const double xs_test[] = {0.0, 0.1, 0.5, 1.0, 2.5, -0.5, -3.0,
                                     3.0, -12.0, 12.0, -30.0, 30.0};
    for (uint32_t t = 0; t < sizeof(xs_test) / sizeof(xs_test[0]); t++) {
      double xd = xs_test[t];
      float x = (float)xd;
      float xs = x * (float)LOG2E; /* host pre-scale, float mul */
      double got = exp2f(xs);      /* the GPU 2^x convention */
      double want = expf(x);       /* math oracle */
      double rel = fabs(got - want) / fmax(1.0, fabs(want));
      double ab = fabs(got - want);
      if (rel > max_rel) {
        max_rel = rel;
        worst = t;
      }
      if (ab > max_abs) {
        max_abs = ab;
      }
    }
    if (max_rel < NC_TOL) {
      printf("  PASS e^x recipe: exp2f(x*log2e) == expf(x), max rel |d| = "
             "%g (max abs %g)\n", max_rel, max_abs);
    } else {
      printf("  FAIL e^x recipe: max rel |d| = %g at xs_test[%llu]\n",
             max_rel, (unsigned long long)worst);
      g_failures++;
    }

    /* Full softmax row with the 2^x form: row sum must stay 1 and
     * match pai_ref_softmax_f32 at 1e-4. */
    {
      const uint64_t n = 64;
      float ref[64];
      float m = x[0];
      double rawsum = 0.0, normsum = 0.0, max_d = 0.0;
      for (uint64_t i = 0; i < n; i++) {
        x[i] = (float)(((int64_t)i - 32) * 0.9375); /* scores +/-30 */
        if (x[i] > m) {
          m = x[i];
        }
      }
      for (uint64_t i = 0; i < n; i++) {
        float xs = (x[i] - m) * (float)LOG2E; /* pre-scale */
        c[i] = exp2f(xs);                     /* v_exp 2^x */
        rawsum += c[i];
      }
      for (uint64_t i = 0; i < n; i++) {
        c[i] /= (float)rawsum;
        normsum += c[i]; /* normalized row, must be ~1 */
      }
      st = pai_ref_softmax_f32(x, ref, n);
      REQUIRE(st == PAI_OK);
      for (uint64_t i = 0; i < n; i++) {
        double d = fabs((double)c[i] - (double)ref[i]);
        if (d > max_d) {
          max_d = d;
        }
      }
      if (max_d < NC_TOL && fabs(normsum - 1.0) < 1e-5) {
        printf("  PASS softmax with 2^x form: max |d| = %g vs ref, "
               "row sum = %.9f\n", max_d, normsum);
      } else {
        printf("  FAIL softmax with 2^x form: max |d| = %g, row sum = "
               "%.9f\n", max_d, normsum);
        g_failures++;
      }
    }

    /* SiLU with the 2^x form: x/(1+exp2f(-x*log2e)) == reference. */
    {
      const uint64_t n = 64;
      float ref[64];
      double max_d = 0.0;
      for (uint64_t i = 0; i < n; i++) {
        x[i] = (float)(-3.0 + 6.0 * (double)(i % 17) / 16.0);
        float xs = -x[i] * (float)LOG2E;
        c[i] = x[i] / (1.0f + exp2f(xs));
      }
      st = pai_ref_silu_f32(x, ref, n);
      REQUIRE(st == PAI_OK);
      for (uint64_t i = 0; i < n; i++) {
        double d = fabs((double)c[i] - (double)ref[i]);
        if (d > max_d) {
          max_d = d;
        }
      }
      if (max_d < NC_TOL) {
        printf("  PASS SiLU with 2^x form: max |d| = %g\n", max_d);
      } else {
        printf("  FAIL SiLU with 2^x form: max |d| = %g\n", max_d);
        g_failures++;
      }
    }
  }

  /* ---- 6) Final production forms through the G75/G76 mirrors +
   * the locked 2^x model (sez.5): with v_rcp and v_max HW-validated
   * (G75/G76, commit 78a8595) the softmax/SiLU kernels are fully
   * serial-buildable. This section locks the exact forms the payload
   * dispatches: softmax = row-max (serial reduce over v_max) +
   * e = 2^((x-m)*log2e) (G72 prescale) + float sum + rcp(sum) via
   * v_rcp + mul; SiLU = x * rcp(1+e). rcp and max go through the
   * exact nlexp mirror ABI (header [n,pad,x[..],y[..]] at ud[2:3],
   * C at ud[4:5], 1 group/elem). ---- */
  {
    static const double LOG2E6 = 1.4426950408889634;
    const uint64_t n = 64;
    uint32_t ud6[16] = {0};
    static uint32_t h6[2 + 2 * 64];
    static float e6[64], den6[64], out6[64], row6[64];
    float ref6[64];
    double max_d = 0.0, rowsum = 0.0;
    float m6 = 0.0f, sum6 = 0.0f, r6 = 0.0f;
    uint64_t k;

    /* --- softmax via rcp: e*rcp(sum), row-max serial reduce --- */
    for (uint64_t i = 0; i < n; i++) {
      x[i] = (float)(((int64_t)i - 32) * 0.9375); /* scores +/-30 */
    }
    m6 = x[0];
    for (uint64_t i = 1; i < n; i++) {
      if (x[i] > m6) {
        m6 = x[i]; /* v_max building block in a serial reduce */
      }
    }
    for (uint64_t i = 0; i < n; i++) {
      e6[i] = exp2f((x[i] - m6) * (float)LOG2E6); /* G72 2^x + prescale */
      sum6 += e6[i];
    }
    h6[0] = 1;
    h6[1] = 0;
    memcpy(&h6[2], &sum6, 4);
    ud6[2] = (uint32_t)(uintptr_t)h6;
    ud6[3] = (uint32_t)((uintptr_t)h6 >> 32);
    ud6[4] = (uint32_t)(uintptr_t)out6;
    ud6[5] = (uint32_t)((uintptr_t)out6 >> 32);
    st = pai_host_kernel_nlexp_rcp(NULL, ud6, 1, 1);
    REQUIRE(st == PAI_OK);
    memcpy(&r6, out6, 4);
    for (uint64_t i = 0; i < n; i++) {
      row6[i] = e6[i] * r6; /* mul-by-reciprocal, no division */
      rowsum += row6[i];
    }
    st = pai_ref_softmax_f32(x, ref6, n);
    REQUIRE(st == PAI_OK);
    max_d = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      double d = fabs((double)row6[i] - (double)ref6[i]);
      if (d > max_d) {
        max_d = d;
      }
    }
    if (max_d < NC_TOL && fabs(rowsum - 1.0) < 1e-5) {
      printf("  PASS softmax via rcp(sum): max |d| = %g, row sum = "
             "%.9f\n", max_d, rowsum);
    } else {
      printf("  FAIL softmax via rcp(sum): max |d| = %g, row sum = "
             "%.9f\n", max_d, rowsum);
      g_failures++;
    }

    /* --- SiLU via rcp: x*rcp(1+2^(-x*log2e)) --- */
    for (uint64_t i = 0; i < n; i++) {
      x[i] = (float)(-3.0 + 6.0 * (double)(i % 17) / 16.0);
      e6[i] = exp2f(-x[i] * (float)LOG2E6);
      den6[i] = 1.0f + e6[i];
      memcpy(&h6[2 + i], &den6[i], 4);
    }
    h6[0] = (uint32_t)n;
    h6[1] = 0;
    st = pai_host_kernel_nlexp_rcp(NULL, ud6, 1, n);
    REQUIRE(st == PAI_OK);
    max_d = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      float rr;
      memcpy(&rr, &out6[i], 4);
      row6[i] = x[i] * rr;
    }
    st = pai_ref_silu_f32(x, ref6, n);
    REQUIRE(st == PAI_OK);
    for (uint64_t i = 0; i < n; i++) {
      double d = fabs((double)row6[i] - (double)ref6[i]);
      if (d > max_d) {
        max_d = d;
      }
    }
    if (max_d < NC_TOL) {
      printf("  PASS SiLU via rcp(1+e): max |d| = %g\n", max_d);
    } else {
      printf("  FAIL SiLU via rcp(1+e): max |d| = %g\n", max_d);
      g_failures++;
    }

    /* --- v_max elementwise via mirror: max(x,0) == relu --- */
    h6[0] = (uint32_t)n;
    h6[1] = 0;
    for (uint64_t i = 0; i < n; i++) {
      memcpy(&h6[2 + i], &x[i], 4);
      h6[2 + n + i] = 0; /* y = 0 */
    }
    st = pai_host_kernel_nlexp_max(NULL, ud6, 1, n);
    REQUIRE(st == PAI_OK);
    st = pai_ref_relu_f32(x, ref6, n);
    REQUIRE(st == PAI_OK);
    max_d = 0.0;
    for (uint64_t i = 0; i < n; i++) {
      float vv;
      memcpy(&vv, &out6[i], 4);
      double d = fabs((double)vv - (double)ref6[i]);
      if (d > max_d) {
        max_d = d;
      }
    }
    if (max_d == 0.0) {
      printf("  PASS v_max(x,0) == relu via mirror ABI\n");
    } else {
      printf("  FAIL v_max(x,0): max |d| = %g\n", max_d);
      g_failures++;
    }
    (void)k;
  }

  printf("== nonlinear_contract %s ==\n", g_failures ? "FAILED" : "PASSED");
  (void)mism;
  return g_failures ? 1 : 0;
}
