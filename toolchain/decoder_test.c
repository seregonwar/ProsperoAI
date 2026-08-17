/*
 * ProsperoAI — Phase 2 reference path: small decoder transformer, one
 * forward pass producing logits and a greedy first token.
 *
 * Chained entirely through the cpu/reference oracle ops (ref_ops.h,
 * whitepaper §37): embedding lookup -> RoPE (pai_ref_rope_cossin_f32 +
 * pai_ref_rope_f32) -> causal GQA attention (pai_ref_attention_f32) ->
 * residual + RMSNorm -> SiLU MLP -> residual + RMSNorm -> output GEMM
 * -> logits -> greedy sample (pai_sampler, temperature 0).
 *
 * This is the Phase-2 CPU reference leg (Seat B): the small decoder
 * forward pass that the GPU wave-parallel kernels of Seat A (G55-G57
 * ramp/vpick family) will be differentially validated against once the
 * GEMV W-side unlock lands.
 *
 * Exit: prints per-section PASS/FAIL and returns 0 only when every
 * ref-op result matches an independent double-precision manual
 * computation and the greedy token equals the double oracle's argmax.
 *
 * Build: host-tests/host-reference preset; binary `decoder_test`.
 */

#include <pai/error.h>

#include <ref_ops.h>
#include <sampler.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tiny single-layer decoder: n_embd = H * hd = 8. */
#define D_MODEL 8
#define H_HEADS 2
#define HK_KV   1   /* GQA: 1 KV head shared by both query heads */
#define HD_DIM  4   /* D_MODEL / H_HEADS */
#define R2      2   /* rotary dim = 2 * r2 = 4 = hd */
#define SEQ     3   /* prefill length */
#define MLP_HID 16
#define VOCAB   16

static int g_failures;

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

/* Deterministic weights. All projections are [D_MODEL][D_MODEL] without
 * bias except the MLP which carries one (realistic LLaMA shape). */
static void
fill_weights(float *emb, float *wq, float *wk, float *wv, float *wo,
             float *g1, float *w1, float *b1, float *w2, float *b2,
             float *g2, float *wout) {
  int i, j;
  for (i = 0; i < VOCAB * D_MODEL; i++) {
    emb[i] = (float)((i % 11) - 5) * 0.1f;
  }
  for (i = 0; i < D_MODEL; i++) {
    for (j = 0; j < D_MODEL; j++) {
      float v = (float)(((i * 3 + j * 5) % 7) - 3) * 0.1f;
      wq[i * D_MODEL + j] = v;
      wk[i * D_MODEL + j] = v * 0.9f;
      wv[i * D_MODEL + j] = v * 1.1f;
      wo[i * D_MODEL + j] = v * 0.8f;
      wout[i * D_MODEL + j] = (float)(((i * 5 + j * 7) % 9) - 4) * 0.05f;
    }
  }
  for (i = 0; i < D_MODEL; i++) {
    g1[i] = 1.0f + 0.1f * (float)i;
    g2[i] = 1.0f - 0.05f * (float)i;
  }
  for (i = 0; i < D_MODEL; i++) {
    for (j = 0; j < MLP_HID; j++) {
      w1[i * MLP_HID + j] = (float)(((i * 3 + j * 5) % 7) - 3) * 0.1f;
    }
  }
  for (j = 0; j < MLP_HID; j++) {
    b1[j] = (float)(j % 5) * 0.05f;
  }
  for (i = 0; i < MLP_HID; i++) {
    for (j = 0; j < D_MODEL; j++) {
      w2[i * D_MODEL + j] = (float)(((i * 7 + j * 2) % 9) - 4) * 0.05f;
    }
  }
  for (j = 0; j < D_MODEL; j++) {
    b2[j] = (float)((j * 3) % 4) * 0.1f;
  }
}

/* --- independent double-precision oracle ---------------------------------
 * Same topology, computed with double arithmetic from scratch so the
 * ref-op chain is genuinely cross-checked (no shared helpers). */

static void
oracle_gemm(const double *a, const double *b, double *c, uint64_t m,
            uint64_t n, uint64_t k) {
  uint64_t i, j, t;
  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      double acc = 0.0;
      for (t = 0; t < k; t++) {
        acc += a[i * k + t] * b[t * n + j];
      }
      c[i * n + j] = acc;
    }
  }
}

static void
oracle_add_bias(const double *a, const double *bias, double *c, uint64_t rows,
                uint64_t cols) {
  uint64_t i, j;
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      c[i * cols + j] = a[i * cols + j] + bias[j];
    }
  }
}

static void
oracle_rmsnorm_gamma(const double *a, double *c, uint64_t n,
                     const double *gamma, double eps) {
  double mean = 0.0;
  uint64_t i;
  for (i = 0; i < n; i++) {
    mean += a[i] * a[i];
  }
  mean = mean / (double)n;
  {
    double inv = 1.0 / sqrt(mean + eps);
    for (i = 0; i < n; i++) {
      c[i] = a[i] * gamma[i] * inv;
    }
  }
}

static void
oracle_silu(const double *a, double *c, uint64_t n) {
  uint64_t i;
  for (i = 0; i < n; i++) {
    c[i] = a[i] / (1.0 + exp(-a[i]));
  }
}

/* Causal GQA attention in double, layout [seq, H*hd] per-head column
 * blocks, KV head hk = h * HK / H. */
static void
oracle_attention(uint64_t h, uint64_t hk, uint64_t seq, uint64_t hd,
                 const double *q, const double *k, const double *v,
                 double *out) {
  uint64_t p, hh, t;
  double inv = 1.0 / sqrt((double)hd);
  for (p = 0; p < seq; p++) {
    for (hh = 0; hh < h; hh++) {
      uint64_t kh = hh * hk / h;
      double scores[SEQ];
      double maxs = -1e30;
      double denom = 0.0;
      for (t = 0; t <= p; t++) {
        double s = 0.0;
        uint64_t d;
        for (d = 0; d < hd; d++) {
          s += q[(p * h + hh) * hd + d] * k[(t * hk + kh) * hd + d];
        }
        s *= inv;
        scores[t] = s;
        if (s > maxs) {
          maxs = s;
        }
      }
      for (t = 0; t <= p; t++) {
        scores[t] = exp(scores[t] - maxs);
        denom += scores[t];
      }
      for (t = 0; t <= p; t++) {
        uint64_t d;
        for (d = 0; d < hd; d++) {
          double w = scores[t] / denom;
          out[(p * h + hh) * hd + d] += w * v[(t * hk + kh) * hd + d];
        }
      }
    }
  }
}

/* Full forward in double: returns logits[seq*vocab] and last-token
 * argmax. `toks` holds the prefill ids. */
static void
decoder_oracle(const int *toks, const float *emb, const float *wq,
               const float *wk, const float *wv, const float *wo,
               const float *g1, const float *w1, const float *b1,
               const float *w2, const float *b2, const float *g2,
               const float *wout, double *logits, int *out_argmax) {
  double e[SEQ * D_MODEL];
  double dq[SEQ * D_MODEL], dk[SEQ * D_MODEL], dv[SEQ * D_MODEL];
  double dwq[D_MODEL * D_MODEL], dwk[D_MODEL * D_MODEL];
  double dwv[D_MODEL * D_MODEL], dwo[D_MODEL * D_MODEL];
  double dwout[D_MODEL * VOCAB];
  double dg1[D_MODEL], dg2[D_MODEL];
  double dw1[D_MODEL * MLP_HID], db1[MLP_HID];
  double dw2[MLP_HID * D_MODEL], db2[D_MODEL];
  double attn[SEQ * D_MODEL];
  double h1a[SEQ * D_MODEL], h1[SEQ * D_MODEL];
  double gmlp[SEQ * MLP_HID], ag[SEQ * MLP_HID], h2[SEQ * MLP_HID];
  double mlp[SEQ * D_MODEL], h3a[SEQ * D_MODEL], h3[SEQ * D_MODEL];
  double ct[SEQ * R2], st[SEQ * R2], theta[SEQ * R2];
  int p, i;

  for (i = 0; i < D_MODEL * D_MODEL; i++) {
    dwq[i] = (double)wq[i];
    dwk[i] = (double)wk[i];
    dwv[i] = (double)wv[i];
    dwo[i] = (double)wo[i];
  }
  for (i = 0; i < D_MODEL * VOCAB; i++) {
    dwout[i] = (double)wout[i];
  }
  for (i = 0; i < D_MODEL; i++) {
    dg1[i] = (double)g1[i];
    dg2[i] = (double)g2[i];
  }
  for (i = 0; i < D_MODEL * MLP_HID; i++) {
    dw1[i] = (double)w1[i];
  }
  for (i = 0; i < MLP_HID; i++) {
    db1[i] = (double)b1[i];
  }
  for (i = 0; i < MLP_HID * D_MODEL; i++) {
    dw2[i] = (double)w2[i];
  }
  for (i = 0; i < D_MODEL; i++) {
    db2[i] = (double)b2[i];
  }

  for (p = 0; p < SEQ; p++) {
    for (i = 0; i < D_MODEL; i++) {
      e[p * D_MODEL + i] = (double)emb[toks[p] * D_MODEL + i];
    }
  }
  /* RoPE, LLaMA style, double */
  for (i = 0; i < R2; i++) {
    double invf = pow(10000.0, -2.0 * (double)i / (double)(2 * R2));
    for (p = 0; p < SEQ; p++) {
      theta[p * R2 + i] = (double)p * invf;
      ct[p * R2 + i] = cos(theta[p * R2 + i]);
      st[p * R2 + i] = sin(theta[p * R2 + i]);
    }
  }
  /* RoPE rotates per-head column block: head hh occupies columns
   * hh*hd .. (hh+1)*hd, pairs (d, d+R2) inside each block. */
  for (p = 0; p < SEQ; p++) {
    for (int hh = 0; hh < H_HEADS; hh++) {
      double *row = e + p * D_MODEL + hh * HD_DIM;
      for (i = 0; i < R2; i++) {
        double a = row[i];
        double b = row[i + R2];
        row[i] = a * ct[p * R2 + i] - b * st[p * R2 + i];
        row[i + R2] = a * st[p * R2 + i] + b * ct[p * R2 + i];
      }
    }
  }
  oracle_gemm(e, dwq, dq, SEQ, D_MODEL, D_MODEL);
  oracle_gemm(e, dwk, dk, SEQ, D_MODEL, D_MODEL);
  oracle_gemm(e, dwv, dv, SEQ, D_MODEL, D_MODEL);
  memset(attn, 0, sizeof(attn));
  oracle_attention(H_HEADS, HK_KV, SEQ, HD_DIM, dq, dk, dv, attn);
  /* output projection + residual */
  {
    double o[SEQ * D_MODEL];
    oracle_gemm(attn, dwo, o, SEQ, D_MODEL, D_MODEL);
    for (p = 0; p < SEQ * D_MODEL; p++) {
      h1a[p] = o[p] + e[p];
    }
  }
  for (p = 0; p < SEQ; p++) {
    oracle_rmsnorm_gamma(h1a + p * D_MODEL, h1 + p * D_MODEL, D_MODEL,
                         dg1, 1e-5);
  }
  oracle_gemm(h1, dw1, gmlp, SEQ, MLP_HID, D_MODEL);
  oracle_add_bias(gmlp, db1, ag, SEQ, MLP_HID);
  oracle_silu(ag, h2, SEQ * MLP_HID);
  oracle_gemm(h2, dw2, mlp, SEQ, D_MODEL, MLP_HID);
  oracle_add_bias(mlp, db2, h3a, SEQ, D_MODEL);
  for (p = 0; p < SEQ * D_MODEL; p++) {
    h3[p] = h1[p] + h3a[p];
  }
  for (p = 0; p < SEQ; p++) {
    oracle_rmsnorm_gamma(h3 + p * D_MODEL, h1 + p * D_MODEL, D_MODEL,
                         dg2, 1e-5);
  }
  oracle_gemm(h1, dwout, logits, SEQ, VOCAB, D_MODEL);
  {
    double best = logits[(SEQ - 1) * VOCAB];
    int bi = 0;
    for (i = 1; i < VOCAB; i++) {
      if (logits[(SEQ - 1) * VOCAB + i] > best) {
        best = logits[(SEQ - 1) * VOCAB + i];
        bi = i;
      }
    }
    *out_argmax = bi;
  }
}

int
main(void) {
  static float emb[VOCAB * D_MODEL];
  static float wq[D_MODEL * D_MODEL], wk[D_MODEL * D_MODEL];
  static float wv[D_MODEL * D_MODEL], wo[D_MODEL * D_MODEL];
  static float wout[D_MODEL * VOCAB];
  static float g1[D_MODEL], g2[D_MODEL];
  static float w1[D_MODEL * MLP_HID], b1[MLP_HID];
  static float w2[MLP_HID * D_MODEL], b2[D_MODEL];
  static float cos_t[SEQ * R2], sin_t[SEQ * R2];
  static float e[SEQ * D_MODEL];
  static float q[SEQ * D_MODEL], kv[SEQ * D_MODEL], vv[SEQ * D_MODEL];
  static float o[SEQ * D_MODEL], h1a[SEQ * D_MODEL], h1[SEQ * D_MODEL];
  static float gmlp[SEQ * MLP_HID], ag[SEQ * MLP_HID], h2[SEQ * MLP_HID];
  static float mlp[SEQ * D_MODEL], h3a[SEQ * D_MODEL], h3[SEQ * D_MODEL];
  static float logits[SEQ * VOCAB];
  double logits_ref[SEQ * VOCAB];
  int argmax_oracle = -1;
  int toks[SEQ] = {2, 5, 9};
  pai_status_t st;
  int i;

  printf("== Phase 2 reference path: decoder 8->16, seq %d, H=%d HK=%d "
         "==\n", SEQ, H_HEADS, HK_KV);

  fill_weights(emb, wq, wk, wv, wo, g1, w1, b1, w2, b2, g2, wout);
  decoder_oracle(toks, emb, wq, wk, wv, wo, g1, w1, b1, w2, b2, g2, wout,
                 logits_ref, &argmax_oracle);

  /* 1) RoPE tables (LLaMA style, base 10000) */
  st = pai_ref_rope_cossin_f32(SEQ, R2, 10000.0f, cos_t, sin_t);
  REQUIRE(st == PAI_OK);
  {
    int ok = 1;
    /* p=0 row is cos=1, sin=0; the p>=1 rows are spot-checked
     * analytically inside the RoPE comparison below. */
    for (i = 0; i < R2; i++) {
      if (cos_t[i] != 1.0f || sin_t[i] != 0.0f) {
        ok = 0;
      }
    }
    REQUIRE(ok);
  }

  /* 2) embedding lookup + RoPE on the [seq, n_embd] activations */
  for (i = 0; i < SEQ * D_MODEL; i++) {
    e[i] = emb[toks[i / D_MODEL] * D_MODEL + (i % D_MODEL)];
  }
  st = pai_ref_rope_f32(e, SEQ * H_HEADS, HD_DIM, SEQ, H_HEADS, cos_t, sin_t,
                        R2, e);
  REQUIRE(st == PAI_OK);
  {
    int ok = 1;
    for (i = 0; i < SEQ * D_MODEL; i++) {
      /* the double oracle writes into logits_ref scratch? no: compare
       * against recomputed expectation: e_ref layout is [seq, n_embd]
       * = [seq, H*hd]; rope works on the [seq*H, hd] view, so pair
       * (p, h, d) at flat p*H*hd + h*hd + d = p*D_MODEL + h*hd + d. */
      int p = i / D_MODEL;
      int rem = i % D_MODEL;
      int h = rem / HD_DIM;
      int d = rem % HD_DIM;
      double want;
      double a = (double)emb[toks[p] * D_MODEL + rem];
      double b;
      double ct, stv;
      if (d < R2) {
        b = (double)emb[toks[p] * D_MODEL + h * HD_DIM + d + R2];
        ct = cos((double)p *
                 pow(10000.0, -2.0 * (double)d / (double)(2 * R2)));
        stv = sin((double)p *
                  pow(10000.0, -2.0 * (double)d / (double)(2 * R2)));
        want = a * ct - b * stv;
      } else if (d < 2 * R2) {
        /* out[d] = x[d-R2]*sin + x[d]*cos (docstring pair i=d-R2) */
        b = (double)emb[toks[p] * D_MODEL + h * HD_DIM + d - R2];
        ct = cos((double)p *
                 pow(10000.0, -2.0 * (double)(d - R2) / (double)(2 * R2)));
        stv = sin((double)p *
                  pow(10000.0, -2.0 * (double)(d - R2) / (double)(2 * R2)));
        want = b * stv + a * ct;
      } else {
        want = a;
      }
      if (fabs((double)e[i] - want) > 1e-5 * (1.0 + fabs(want))) {
        ok = 0;
      }
    }
    REQUIRE(ok);
    if (ok) {
      printf("  PASS RoPE (cossin tables + rope on [seq,n_embd])\n");
    } else {
      g_failures++;
    }
  }

  /* 3) QKV projections (no bias) */
  st = pai_ref_gemm_f32(SEQ, D_MODEL, D_MODEL, e, wq, q);
  REQUIRE(st == PAI_OK);
  st = pai_ref_gemm_f32(SEQ, D_MODEL, D_MODEL, e, wk, kv);
  REQUIRE(st == PAI_OK);
  st = pai_ref_gemm_f32(SEQ, D_MODEL, D_MODEL, e, wv, vv);
  REQUIRE(st == PAI_OK);
  /* QKV are covered end-to-end by the attention + logits differential
   * compare below (same code path as the double oracle). */

  /* 4) causal GQA attention */
  memset(o, 0, sizeof(o));
  st = pai_ref_attention_f32(H_HEADS, HK_KV, SEQ, HD_DIM, q, kv, vv, o);
  REQUIRE(st == PAI_OK);
  /* output projection */
  {
    float op[SEQ * D_MODEL];
    st = pai_ref_gemm_f32(SEQ, D_MODEL, D_MODEL, o, wo, op);
    REQUIRE(st == PAI_OK);
    memcpy(o, op, sizeof(op));
  }
  /* residual + RMSNorm (attn pre-norm) */
  for (i = 0; i < SEQ * D_MODEL; i++) {
    h1a[i] = o[i] + e[i];
  }
  for (i = 0; i < SEQ; i++) {
    st = pai_ref_rmsnorm_gamma_f32(h1a + i * D_MODEL, h1 + i * D_MODEL,
                                   D_MODEL, g1, 1e-5f);
    REQUIRE(st == PAI_OK);
  }

  /* 5) SiLU MLP + residual + RMSNorm (post-norm) */
  st = pai_ref_gemm_f32(SEQ, MLP_HID, D_MODEL, h1, w1, gmlp);
  REQUIRE(st == PAI_OK);
  st = pai_ref_biasadd_f32(gmlp, b1, ag, SEQ, MLP_HID);
  REQUIRE(st == PAI_OK);
  st = pai_ref_silu_f32(ag, h2, SEQ * MLP_HID);
  REQUIRE(st == PAI_OK);
  st = pai_ref_gemm_f32(SEQ, D_MODEL, MLP_HID, h2, w2, mlp);
  REQUIRE(st == PAI_OK);
  st = pai_ref_biasadd_f32(mlp, b2, h3a, SEQ, D_MODEL);
  REQUIRE(st == PAI_OK);
  for (i = 0; i < SEQ * D_MODEL; i++) {
    h3[i] = h1[i] + h3a[i];
  }
  for (i = 0; i < SEQ; i++) {
    st = pai_ref_rmsnorm_gamma_f32(h3 + i * D_MODEL, h1 + i * D_MODEL,
                                   D_MODEL, g2, 1e-5f);
    REQUIRE(st == PAI_OK);
  }

  /* 6) output logits */
  st = pai_ref_gemm_f32(SEQ, VOCAB, D_MODEL, h1, wout, logits);
  REQUIRE(st == PAI_OK);

  /* 7) differential: ref chain vs double oracle (snapped to float so
   * both sides are the same arithmetic type, like mlp_test). */
  {
    float logits_want[SEQ * VOCAB];
    uint64_t mism = 0;
    for (i = 0; i < SEQ * VOCAB; i++) {
      logits_want[i] = (float)logits_ref[i];
    }
    st = pai_ref_compare_f32(logits, logits_want, SEQ * VOCAB, 1e-4f,
                             1e-4f, &mism);
    if (st == PAI_OK) {
      printf("  PASS logits == double oracle (%d x %d)\n", SEQ, VOCAB);
    } else {
      printf("  FAIL logits mismatch at %llu (ref=%f oracle=%f)\n",
             (unsigned long long)mism, (double)logits[mism],
             (double)logits_want[mism]);
      g_failures++;
    }
  }

  /* 8) greedy first token (temperature 0 -> argmax) matches oracle */
  {
    pai_sampler_t sampler;
    uint32_t tok = VOCAB;
    pai_sampler_init(&sampler, 1);
    st = pai_sampler_sample_logits(&sampler, logits + (SEQ - 1) * VOCAB,
                                   VOCAB, &tok);
    REQUIRE(st == PAI_OK);
    if ((int)tok == argmax_oracle) {
      printf("  PASS greedy first token == oracle argmax (%u)\n", tok);
    } else {
      printf("  FAIL token %u != oracle argmax %d\n", tok, argmax_oracle);
      g_failures++;
    }
  }

  printf("== Phase 2 reference path %s ==\n",
         g_failures ? "FAILED" : "PASSED");
  return g_failures ? 1 : 0;
}
