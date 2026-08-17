#include "test.h"

#include <string.h>

#include <ref_ops.h>

#include <math.h>
#include <stdlib.h>

TEST_MAIN_BEGIN()

{
  /* vecadd vs manual computation */
  float a[64], b[64], c[64];
  for (int i = 0; i < 64; i++) {
    a[i] = (float)i * 0.5f;
    b[i] = (float)(i % 7) - 3.0f;
  }
  CHECK(pai_ref_vecadd_f32(a, b, c, 64) == PAI_OK);
  for (int i = 0; i < 64; i++) {
    if (fabsf(c[i] - (a[i] + b[i])) > 1e-6f) {
      CHECK(0);
      break;
    }
  }
}

{
  /* gemm 4x3x2 vs manual */
  float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float b[6] = {1, 0, 2, 0, 1, 2}; /* 2x3 row-major */
  float c[12];
  CHECK(pai_ref_gemm_f32(4, 3, 2, a, b, c) == PAI_OK);

  /* row0 = a[0..1] x b */
  CHECK(fabsf(c[0] - (1 * 1 + 2 * 0)) < 1e-6f);
  CHECK(fabsf(c[1] - (1 * 0 + 2 * 1)) < 1e-6f);
  CHECK(fabsf(c[2] - (1 * 2 + 2 * 2)) < 1e-6f);
  /* row3 = a[6..7] x b */
  CHECK(fabsf(c[9] - (7 * 1 + 8 * 0)) < 1e-6f);
  CHECK(fabsf(c[10] - (7 * 0 + 8 * 1)) < 1e-6f);
  CHECK(fabsf(c[11] - (7 * 2 + 8 * 2)) < 1e-6f);
}

{
  /* compare detects mismatches and reports the first index */
  float a[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  float b[8] = {1, 1, 1, 9, 1, 1, 1, 1};
  uint64_t first = 0;
  CHECK(pai_ref_compare_f32(a, b, 8, 1e-6f, 1e-6f, &first) ==
        PAI_ERR_MISMATCH);
  CHECK_EQ_UINT(first, 3);
  CHECK(pai_ref_compare_f32(a, a, 8, 1e-6f, 1e-6f, &first) == PAI_OK);
}

{
  /* memset16 pattern fill */
  uint32_t out[16];
  uint32_t pat[4] = {0xDEADBEEF, 0x11223344, 0x55667788, 0x99AABBCC};
  CHECK(pai_ref_memset16(out, pat, 4) == PAI_OK);
  for (int i = 0; i < 16; i++) {
    CHECK_EQ_UINT(out[i], pat[i % 4]);
  }
}

{
  /* vecmul element-wise */
  float a[4] = {1, -2, 3, 0.5f};
  float b[4] = {4, 5, -1, 2};
  float c[4];
  CHECK(pai_ref_vecmul_f32(a, b, c, 4) == PAI_OK);
  CHECK(fabsf(c[0] - 4.0f) < 1e-6f);
  CHECK(fabsf(c[1] - -10.0f) < 1e-6f);
  CHECK(fabsf(c[2] - -3.0f) < 1e-6f);
  CHECK(fabsf(c[3] - 1.0f) < 1e-6f);
  CHECK(pai_ref_vecmul_f32(NULL, b, c, 4) == PAI_ERR_INVALID_ARG);
}

{
  /* relu: negatives clamp to 0 */
  float a[6] = {-2, -0.5f, 0, 0.25f, 3, -1e-3f};
  float c[6];
  CHECK(pai_ref_relu_f32(a, c, 6) == PAI_OK);
  for (int i = 0; i < 6; i++) {
    CHECK(fabsf(c[i] - (a[i] > 0 ? a[i] : 0.0f)) < 1e-7f);
  }
}

{
  /* softmax: non-negative, sums to 1, order preserved */
  float a[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  float c[5];
  float sum = 0;
  CHECK(pai_ref_softmax_f32(a, c, 5) == PAI_OK);
  for (int i = 0; i < 5; i++) {
    CHECK(c[i] >= 0.0f && c[i] <= 1.0f);
    sum += c[i];
  }
  CHECK(fabsf(sum - 1.0f) < 1e-6f);
  CHECK(c[4] > c[3] && c[3] > c[2]);
  /* numerically stable: large magnitudes do not overflow */
  float big[3] = {1000.0f, 1000.5f, 1001.0f};
  CHECK(pai_ref_softmax_f32(big, c, 3) == PAI_OK);
  CHECK(fabsf(c[0] + c[1] + c[2] - 1.0f) < 1e-6f);
  CHECK(pai_ref_softmax_f32(a, c, 0) == PAI_ERR_INVALID_ARG);
}

{
  /* rmsnorm vs direct formula */
  float a[4] = {1, 2, 3, 4};
  float c[4];
  double acc = 0;
  CHECK(pai_ref_rmsnorm_f32(a, c, 4, 0.0f) == PAI_OK);
  for (int i = 0; i < 4; i++) {
    acc += (double)a[i] * (double)a[i];
  }
  for (int i = 0; i < 4; i++) {
    float expect = a[i] / sqrtf((float)(acc / 4.0) + 1e-5f);
    CHECK(fabsf(c[i] - expect) < 1e-6f);
  }
}

{
  /* layernorm with gamma/beta vs direct formula */
  float a[4] = {1, 2, 3, 4};
  float g[4] = {2, 2, 2, 2};
  float b[4] = {1, 1, 1, 1};
  float c[4];
  double sum = 0, sumsq = 0;
  float mean, inv_std;
  CHECK(pai_ref_layernorm_f32(a, c, 4, g, b, 0.0f) == PAI_OK);
  for (int i = 0; i < 4; i++) {
    sum += a[i];
    sumsq += (double)a[i] * a[i];
  }
  mean = (float)(sum / 4.0);
  inv_std = 1.0f / sqrtf((float)(sumsq / 4.0 - (double)mean * mean) + 1e-5f);
  for (int i = 0; i < 4; i++) {
    float expect = (a[i] - mean) * inv_std * g[i] + b[i];
    CHECK(fabsf(c[i] - expect) < 1e-5f);
  }
  /* gamma/beta optional */
  CHECK(pai_ref_layernorm_f32(a, c, 4, NULL, NULL, 0.0f) == PAI_OK);
}

{
  /* concat ordering */
  float a[2] = {1, 2};
  float b[3] = {3, 4, 5};
  float c[5];
  CHECK(pai_ref_concat_f32(a, b, c, 2, 3) == PAI_OK);
  for (int i = 0; i < 5; i++) {
    CHECK(fabsf(c[i] - (float)(i + 1)) < 1e-6f);
  }
}

{
  /* silu: x / (1 + e^-x) */
  float a[4] = {0.0f, 1.0f, -1.0f, 2.0f};
  float c[4];
  CHECK(pai_ref_silu_f32(a, c, 4) == PAI_OK);
  CHECK(fabsf(c[0] - 0.0f) < 1e-6f);
  CHECK(fabsf(c[1] - (1.0f / (1.0f + expf(-1.0f)))) < 1e-6f);
  CHECK(fabsf(c[2] - (-1.0f / (1.0f + expf(1.0f)))) < 1e-6f);
  CHECK(fabsf(c[3] - (2.0f / (1.0f + expf(-2.0f)))) < 1e-6f);
}

{
  /* rmsnorm with gamma: out[i] = a[i] * gamma[i] / sqrt(mean(a^2)+eps).
   * a = {1,1,1,1}: rms = 1, so out = gamma (eps is negligible). */
  float a[4] = {1, 1, 1, 1};
  float g[4] = {1, 2, 3, 4};
  float c[4];
  CHECK(pai_ref_rmsnorm_gamma_f32(a, c, 4, g, 1e-9f) == PAI_OK);
  for (int i = 0; i < 4; i++) {
    CHECK(fabsf(c[i] - g[i]) < 1e-5f);
  }
  /* a = {3, 4}: mean = 12.5, rms = 1/sqrt(12.5) ~ 0.282843. */
  {
    float a2[2] = {3, 4};
    float g2[2] = {1, 1};
    float c2[2];
    float rms = 1.0f / sqrtf(12.5f);
    CHECK(pai_ref_rmsnorm_gamma_f32(a2, c2, 2, g2, 1e-9f) == PAI_OK);
    CHECK(fabsf(c2[0] - 3.0f * rms) < 1e-6f);
    CHECK(fabsf(c2[1] - 4.0f * rms) < 1e-6f);
  }
}

{
  /* rope position-major: 2 heads, hd=4, seq=2, r2=2. Row layout is
   * [p*heads + h], so position comes from row/heads. Rotary pairs are
   * (i, i+r2) per llama.cpp: (0,2) and (1,3). cos/sin tables for
   * p=0: (c=1,s=0), p=1: (c=0,s=1). */
  float cos_t[4] = {1, 1, 0, 0}; /* p0: c=1; p1: c=0 */
  float sin_t[4] = {0, 0, 1, 1}; /* p0: s=0; p1: s=1 */
  float x[16] = {1, 3, 2, 4,    /* p0h0: pairs (1,2),(3,4) */
                 5, 7, 6, 8,    /* p0h1 */
                 10, 30, 20, 40, /* p1h0: pairs (10,20),(30,40) */
                 50, 70, 60, 80}; /* p1h1 */
  float c[16];
  CHECK(pai_ref_rope_f32(x, 4, 4, 2, 2, cos_t, sin_t, 2, c) == PAI_OK);
  /* p0 (c=1, s=0): identity. */
  CHECK(fabsf(c[0] - 1.0f) < 1e-6f);
  CHECK(fabsf(c[1] - 3.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 2.0f) < 1e-6f);
  CHECK(fabsf(c[3] - 4.0f) < 1e-6f);
  /* p1 (c=0, s=1): (x0*c - x1*s, x0*s + x1*c) = (-x1, x0). */
  CHECK(fabsf(c[8] - (-20.0f)) < 1e-6f);
  CHECK(fabsf(c[9] - (-40.0f)) < 1e-6f);
  CHECK(fabsf(c[10] - 10.0f) < 1e-6f);
  CHECK(fabsf(c[11] - 30.0f) < 1e-6f);
  CHECK(fabsf(c[12] - (-60.0f)) < 1e-6f);
  CHECK(fabsf(c[13] - (-80.0f)) < 1e-6f);
  CHECK(fabsf(c[14] - 50.0f) < 1e-6f);
  CHECK(fabsf(c[15] - 70.0f) < 1e-6f);
}

{
  /* rope cos/sin tables (LLaMA): dim=4, base=10000, ctx=2.
   * theta_i(p) = p * 10000^(-2i/4): i=0 -> p, i=1 -> p*0.01. */
  float cos_t[4], sin_t[4];
  float c[16];
  CHECK(pai_ref_rope_cossin_f32(2, 2, 10000.0f, cos_t, sin_t) == PAI_OK);
  /* p=0: cos=1, sin=0. */
  CHECK(fabsf(cos_t[0] - 1.0f) < 1e-6f);
  CHECK(fabsf(cos_t[1] - 1.0f) < 1e-6f);
  CHECK(fabsf(sin_t[0]) < 1e-6f);
  CHECK(fabsf(sin_t[1]) < 1e-6f);
  /* p=1: theta0=1, theta1=0.01. */
  CHECK(fabsf(cos_t[2] - cosf(1.0f)) < 1e-6f);
  CHECK(fabsf(sin_t[2] - sinf(1.0f)) < 1e-6f);
  CHECK(fabsf(cos_t[3] - cosf(0.01f)) < 1e-6f);
  CHECK(fabsf(sin_t[3] - sinf(0.01f)) < 1e-6f);
  /* base 0 -> default 10000 (same tables). */
  {
    float ct2[4], st2[4];
    CHECK(pai_ref_rope_cossin_f32(2, 2, 0.0f, ct2, st2) == PAI_OK);
    for (int i = 0; i < 4; i++) {
      CHECK(fabsf(ct2[i] - cos_t[i]) < 1e-6f);
      CHECK(fabsf(st2[i] - sin_t[i]) < 1e-6f);
    }
  }
  /* invalid args */
  CHECK(pai_ref_rope_cossin_f32(0, 2, 10000.0f, cos_t, sin_t) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_rope_cossin_f32(2, 0, 10000.0f, cos_t, sin_t) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_rope_cossin_f32(2, 2, 10000.0f, NULL, sin_t) ==
        PAI_ERR_INVALID_ARG);

  /* GPU equivalence contract (G65/G66, commit a8d9089): on 9.40
   * v_cos/v_sin take the operand in TURNS (x2pi), so the effective
   * GPU table is c[l] = cos/sin(2*pi*scale*(4l+3)) with the G15 lane
   * quirk (4l+3). With per-column scale = inv_freq/(2*pi), that is
   * exactly the oracle row p = 4l+3: cos(inv_freq*p). Verify the
   * mapping analytically against the float oracle tables. */
  {
    const uint32_t rows = 32, r2 = 4; /* ctx 32 covers p=3,7,11,... */
    const double pi = 3.14159265358979323846;
    float ct[rows * r2], st[rows * r2];
    const float base = 10000.0f;
    CHECK(pai_ref_rope_cossin_f32(rows, r2, base, ct, st) == PAI_OK);
    for (uint32_t l = 0; l < 8; l++) {
      uint32_t p = 4 * l + 3; /* G15 lane quirk */
      for (uint32_t i = 0; i < r2; i++) {
        double invf =
            pow((double)base, -2.0 * (double)i / (2.0 * (double)r2));
        double scale = invf / (2.0 * pi); /* turns per unit */
        double gpu = 2.0 * pi * scale * (double)p; /* turns x2pi */
        /* GPU turns formula == oracle row p. */
        CHECK(fabsf((float)cos(gpu) - ct[p * r2 + i]) < 1e-6f);
        CHECK(fabsf((float)sin(gpu) - st[p * r2 + i]) < 1e-6f);
        CHECK(fabsf((float)cos(gpu) - (float)cos((double)p * invf)) <
              1e-6f);
      }
    }
  }

  /* End-to-end: generator -> rope on the same 2-head/2-pos input as
   * the hand-built table test above (single head, hd=4, seq=2, r2=2).
   * p0: c=1,s=0 -> identity; p1: theta=(1, 0.01). */
  float x2[8] = {1, 3, 2, 4, 10, 30, 20, 40};
  CHECK(pai_ref_rope_f32(x2, 2, 4, 2, 1, cos_t, sin_t, 2, c) == PAI_OK);
  /* p0 identity. */
  CHECK(fabsf(c[0] - 1.0f) < 1e-6f);
  CHECK(fabsf(c[1] - 3.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 2.0f) < 1e-6f);
  CHECK(fabsf(c[3] - 4.0f) < 1e-6f);
  /* p1: c[4]/c[6] is pair (i=0): x0=x2[4]=10, x1=x2[6]=20, theta=1;
   * c[5]/c[7] is pair (i=1): x0=x2[5]=30, x1=x2[7]=40, theta=0.01.
   * Oracle uses the generated float tables (not double re-derivation). */
  CHECK(fabsf(c[4] - (10.0f * cos_t[2] - 20.0f * sin_t[2])) < 1e-6f);
  CHECK(fabsf(c[6] - (10.0f * sin_t[2] + 20.0f * cos_t[2])) < 1e-6f);
  CHECK(fabsf(c[5] - (30.0f * cos_t[3] - 40.0f * sin_t[3])) < 1e-6f);
  CHECK(fabsf(c[7] - (30.0f * sin_t[3] + 40.0f * cos_t[3])) < 1e-6f);
}

{
  /* multi-head causal attention vs an independent row-by-row oracle.
   * MHA: h=2, hk=2, seq=2, hd=4 (inv_scale = 1/sqrt(4) = 0.5).
   * Layout is position-major: buffer row = p*H + hh. */
  float q[16] = {1, 2, 0, 0,    /* p0h0 */
                 3, 0, 1, 0,    /* p0h1 */
                 2, 1, 0, 0,    /* p1h0 */
                 0, 4, 0, 1};   /* p1h1 */
  float k[16] = {1, 0, 0, 0,    /* p0h0 */
                 0, 2, 0, 0,    /* p0h1 */
                 0, 1, 0, 0,    /* p1h0 */
                 1, 0, 0, 0};   /* p1h1 */
  float v[16] = {1, 2, 3, 4,    /* p0h0 */
                 5, 6, 7, 8,    /* p0h1 */
                 9, 10, 11, 12, /* p1h0 */
                 13, 14, 15, 16}; /* p1h1 */
  float out[16];
  const float inv = 0.5f;
  int ok = 1;
  CHECK(pai_ref_attention_f32(2, 2, 2, 4, q, k, v, out) == PAI_OK);
  /* p0 attends only to itself (causal): out = v row. */
  for (int d = 0; d < 4; d++) {
    CHECK(fabsf(out[0 * 4 + d] - v[d]) < 1e-6f);
    CHECK(fabsf(out[1 * 4 + d] - v[4 + d]) < 1e-6f);
  }
  /* p1h0: scores over p0,p1 = (2, 1) * inv; softmax -> weighted sum. */
  for (int hh = 0; hh < 2 && ok; hh++) {
    int p = 1;
    float s0 = 0, s1 = 0;
    float w0, w1;
    int d;
    for (d = 0; d < 4; d++) {
      s0 += q[(p * 2 + hh) * 4 + d] * k[(0 * 2 + hh) * 4 + d];
      s1 += q[(p * 2 + hh) * 4 + d] * k[(1 * 2 + hh) * 4 + d];
    }
    w0 = expf(s0 * inv);
    w1 = expf(s1 * inv);
    for (d = 0; d < 4; d++) {
      float expect = (w0 * v[(0 * 2 + hh) * 4 + d] +
                      w1 * v[(1 * 2 + hh) * 4 + d]) /
                     (w0 + w1);
      if (fabsf(out[(p * 2 + hh) * 4 + d] - expect) > 1e-6f) {
        ok = 0;
      }
    }
  }
  CHECK(ok);
  /* GQA: h=4, hk=2, kvh = (hh * hk) / h -> hh 0,1 -> kv 0; 2,3 -> kv 1. */
  {
    float q4[32], k2[16], v2[16], out4[32];
    for (int i = 0; i < 32; i++) {
      q4[i] = (float)((i % 7) - 3);
    }
    for (int i = 0; i < 16; i++) {
      k2[i] = (float)((i % 5) - 2);
      v2[i] = (float)((i % 9) - 4);
    }
    CHECK(pai_ref_attention_f32(4, 2, 2, 4, q4, k2, v2, out4) == PAI_OK);
    /* p0 out row (hh) must equal v row (kvh) exactly (single element
     * softmax = 1). */
    for (int hh = 0; hh < 4; hh++) {
      int kvh = (hh * 2) / 4;
      for (int d = 0; d < 4; d++) {
        CHECK(fabsf(out4[hh * 4 + d] - v2[kvh * 4 + d]) < 1e-6f);
      }
    }
  }
  /* invalid args */
  CHECK(pai_ref_attention_f32(0, 2, 2, 4, q, k, v, out) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_attention_f32(2, 0, 2, 4, q, k, v, out) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_attention_f32(2, 2, 0, 4, q, k, v, out) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_attention_f32(2, 2, 2, 0, q, k, v, out) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_ref_attention_f32(3, 2, 2, 4, q, k, v, out) ==
        PAI_ERR_INVALID_ARG); /* h %% hk != 0 */
  CHECK(pai_ref_attention_f32(2, 2, 2, 4, NULL, k, v, out) ==
        PAI_ERR_INVALID_ARG);
}

{
  /* copy, including aliasing */
  float a[4] = {1, 2, 3, 4};
  float c[4];
  CHECK(pai_ref_copy_f32(a, c, 4) == PAI_OK);
  for (int i = 0; i < 4; i++) {
    CHECK(fabsf(c[i] - a[i]) < 1e-7f);
  }
  CHECK(pai_ref_copy_f32(a, a, 4) == PAI_OK);
  CHECK(fabsf(a[2] - 3.0f) < 1e-7f);
}

{
  /* vecsub */
  float a[4] = {5, 3, -2, 0.5f};
  float b[4] = {2, 7, -2, 1.5f};
  float c[4];
  CHECK(pai_ref_vecsub_f32(a, b, c, 4) == PAI_OK);
  CHECK(fabsf(c[0] - 3.0f) < 1e-6f);
  CHECK(fabsf(c[1] - -4.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 0.0f) < 1e-6f);
  CHECK(fabsf(c[3] - -1.0f) < 1e-6f);
  CHECK(pai_ref_vecsub_f32(NULL, b, c, 4) == PAI_ERR_INVALID_ARG);
}

{
  /* clip */
  float a[5] = {-3, 0, 0.5f, 7, 10};
  float c[5];
  CHECK(pai_ref_clip_f32(a, c, 5, 0.0f, 1.0f) == PAI_OK);
  CHECK(fabsf(c[0] - 0.0f) < 1e-6f);
  CHECK(fabsf(c[1] - 0.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 0.5f) < 1e-6f);
  CHECK(fabsf(c[3] - 1.0f) < 1e-6f);
  CHECK(fabsf(c[4] - 1.0f) < 1e-6f);
  CHECK(pai_ref_clip_f32(a, c, 5, 1.0f, 0.0f) == PAI_ERR_INVALID_ARG);
}

{
  /* dot: deterministic order */
  float a[5] = {1, 2, 3, 4, 5};
  float b[5] = {0.5f, -1, 2, 0.25f, -4};
  float out;
  float want = 1 * 0.5f + 2 * -1.0f + 3 * 2.0f + 4 * 0.25f + 5 * -4.0f;
  CHECK(pai_ref_dot_f32(a, b, &out, 5) == PAI_OK);
  CHECK(fabsf(out - want) < 1e-6f);
  CHECK(pai_ref_dot_f32(a, b, NULL, 5) == PAI_ERR_INVALID_ARG);
}

{
  /* l1/l2 norms */
  float a[4] = {-3, 4, 0, -2};
  float out;
  CHECK(pai_ref_l1norm_f32(a, &out, 4) == PAI_OK);
  CHECK(fabsf(out - 9.0f) < 1e-6f);
  CHECK(pai_ref_l2norm_f32(a, &out, 4) == PAI_OK);
  CHECK(fabsf(out - 5.385164807f) < 1e-6f);
}

{
  /* gemv: 3x2 matrix */
  float a[6] = {1, 2, 3, 4, 5, 6};
  float x[2] = {10, 100};
  float y[3];
  CHECK(pai_ref_gemv_f32(3, 2, a, x, y) == PAI_OK);
  CHECK(fabsf(y[0] - (10 + 200)) < 1e-6f);
  CHECK(fabsf(y[1] - (30 + 400)) < 1e-6f);
  CHECK(fabsf(y[2] - (50 + 600)) < 1e-6f);
  CHECK(pai_ref_gemv_f32(0, 2, a, x, y) == PAI_ERR_INVALID_ARG);
}

{
  /* biasadd: 2 rows x 3 cols broadcast over the last dim */
  float a[6] = {1, 2, 3, 4, 5, 6};
  float bias[3] = {10, 20, 30};
  float c[6];
  CHECK(pai_ref_biasadd_f32(a, bias, c, 2, 3) == PAI_OK);
  CHECK(fabsf(c[0] - 11.0f) < 1e-6f);
  CHECK(fabsf(c[1] - 22.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 33.0f) < 1e-6f);
  CHECK(fabsf(c[3] - 14.0f) < 1e-6f);
  CHECK(fabsf(c[4] - 25.0f) < 1e-6f);
  CHECK(fabsf(c[5] - 36.0f) < 1e-6f);
  CHECK(pai_ref_biasadd_f32(NULL, bias, c, 2, 3) == PAI_ERR_INVALID_ARG);
}

{
  /* scale in-place: a[i] *= alpha, incl. negative and zero entries */
  float a[5] = {2.0f, -3.0f, 0.5f, 0.0f, 7.0f};
  CHECK(pai_ref_scale_f32(a, 2.5f, 5) == PAI_OK);
  CHECK(fabsf(a[0] - 5.0f) < 1e-6f);
  CHECK(fabsf(a[1] - -7.5f) < 1e-6f);
  CHECK(fabsf(a[2] - 1.25f) < 1e-6f);
  CHECK(fabsf(a[3] - 0.0f) < 1e-6f);
  CHECK(fabsf(a[4] - 17.5f) < 1e-6f);
  /* alpha = 1 leaves values untouched */
  CHECK(pai_ref_scale_f32(a, 1.0f, 5) == PAI_OK);
  CHECK(fabsf(a[0] - 5.0f) < 1e-6f);
  /* n = 0 with NULL buffer is legal; non-NULL buffer required for n>0 */
  CHECK(pai_ref_scale_f32(NULL, 1.0f, 0) == PAI_OK);
  CHECK(pai_ref_scale_f32(NULL, 1.0f, 4) == PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
