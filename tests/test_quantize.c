#include "test.h"

#include <quantize.h>

#include <ref_ops.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float
rnd(void) {
  return (float)(rand() % 2000) / 1000.0f - 1.0f; /* [-1, 1) */
}

static void
fill_random(float *buf, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    buf[i] = rnd();
  }
}

TEST_MAIN_BEGIN()

{
  pai_quant_scheme_t scheme;
  float src[256];
  int8_t q[256];
  float scales[8];
  float dec[256];
  float max_abs;
  float rmse;
  uint64_t groups;

  /* Scheme validation. */
  memset(&scheme, 0, sizeof(scheme));
  CHECK_EQ_INT(pai_quant_scheme_check(NULL), PAI_ERR_INVALID_ARG);
  scheme.bit_width = 8;
  scheme.is_signed = 1;
  scheme.group_size = 0;
  CHECK_EQ_INT(pai_quant_scheme_check(&scheme), PAI_OK);
  scheme.bit_width = 16;
  CHECK_EQ_INT(pai_quant_scheme_check(&scheme), PAI_ERR_UNSUPPORTED);
  scheme.bit_width = 4;
  scheme.is_signed = 0;
  CHECK_EQ_INT(pai_quant_scheme_check(&scheme), PAI_ERR_UNSUPPORTED);
  scheme.is_signed = 1;

  /* Sizing. */
  scheme.bit_width = 8;
  CHECK_EQ_UINT(pai_quant_value_bytes(256, &scheme), 256); /* q8: 1 byte */
  scheme.bit_width = 4;
  CHECK_EQ_UINT(pai_quant_value_bytes(256, &scheme), 128); /* packed 2/byte */
  CHECK_EQ_UINT(pai_quant_value_bytes(3, &scheme), 2);     /* odd tail */
  CHECK_EQ_UINT(pai_quant_num_groups(256, &scheme), 1);    /* per-tensor */
  scheme.group_size = 64;
  CHECK_EQ_UINT(pai_quant_num_groups(256, &scheme), 4);
  CHECK_EQ_UINT(pai_quant_num_groups(257, &scheme), 5); /* partial group */
  CHECK_EQ_UINT(pai_quant_weights_bytes(256, &scheme),
                128 + 4 * 4); /* packed + scales */

  /* q8 round-trip over [-1,1): quantization error bounded by the
   * finest step (1/127 per group scale <= 1/127). */
  fill_random(src, 256);
  scheme.bit_width = 8;
  scheme.group_size = 64;
  CHECK_EQ_INT(pai_quantize_f32(src, 256, &scheme, q, scales, &groups),
               PAI_OK);
  CHECK_EQ_UINT(groups, 4);
  CHECK_EQ_INT(pai_dequantize_f32(q, scales, 256, &scheme, dec), PAI_OK);
  CHECK_EQ_INT(pai_quant_error_f32(src, dec, 256, &max_abs, &rmse), PAI_OK);
  CHECK(max_abs < 0.02f);
  CHECK(rmse < 0.01f);

  /* Exact round trip for representable values: anything with the max
   * magnitude maps onto the top quantum exactly (modulo float 1e-6).
   * (50 is deliberately absent: with scale 100/127 it sits halfway
   * between quanta and is NOT exactly representable.) */
  {
    float exact[4] = {0.0f, 100.0f, -100.0f, 100.0f};
    scheme.bit_width = 8;
    scheme.group_size = 0;
    CHECK_EQ_INT(pai_quantize_f32(exact, 4, &scheme, q, scales, &groups),
                 PAI_OK);
    CHECK_EQ_INT(pai_dequantize_f32(q, scales, 4, &scheme, dec), PAI_OK);
    for (int i = 0; i < 4; i++) {
      CHECK(fabsf(dec[i] - exact[i]) < 1e-4f);
    }
  }

  /* q4 round trip: 2.0 is exactly representable (2.0 = 7 * (2/7)). */
  {
    float exact[4] = {0.0f, 2.0f, -2.0f, 1.0f};
    scheme.bit_width = 4;
    scheme.group_size = 0;
    CHECK_EQ_INT(pai_quantize_f32(exact, 4, &scheme, q, scales, &groups),
                 PAI_OK);
    CHECK_EQ_UINT(pai_quant_value_bytes(4, &scheme), 2);
    CHECK_EQ_INT(pai_dequantize_f32(q, scales, 4, &scheme, dec), PAI_OK);
    CHECK(fabsf(dec[0] - 0.0f) < 1e-5f);
    CHECK(fabsf(dec[1] - 2.0f) < 1e-4f);
    CHECK(fabsf(dec[2] + 2.0f) < 1e-4f);
    CHECK(fabsf(dec[3] - 1.0f) < 0.2f); /* 1.0 quantizes to ~1.0 with 1/7 step */
  }

  /* Odd element count: padding nibble must not corrupt the valid ones. */
  {
    float odd[3] = {4.0f, -4.0f, 4.0f};
    scheme.bit_width = 4;
    scheme.group_size = 0;
    CHECK_EQ_INT(pai_quantize_f32(odd, 3, &scheme, q, scales, &groups),
                 PAI_OK);
    CHECK_EQ_UINT(pai_quant_value_bytes(3, &scheme), 2);
    CHECK_EQ_INT(pai_dequantize_f32(q, scales, 3, &scheme, dec), PAI_OK);
    CHECK(fabsf(dec[0] - 4.0f) < 1e-4f);
    CHECK(fabsf(dec[1] + 4.0f) < 1e-4f);
    CHECK(fabsf(dec[2] - 4.0f) < 1e-4f);
  }

  /* Invalid arguments. */
  {
    float x[4] = {1, 2, 3, 4};
    scheme.bit_width = 8;
    scheme.is_signed = 1;
    scheme.group_size = 0;
    CHECK_EQ_INT(pai_quantize_f32(NULL, 4, &scheme, q, scales, NULL),
                 PAI_ERR_INVALID_ARG);
    CHECK_EQ_INT(pai_dequantize_f32(q, NULL, 4, &scheme, dec),
                 PAI_ERR_INVALID_ARG);
    CHECK_EQ_INT(pai_quant_error_f32(x, dec, 4, NULL, &rmse),
                 PAI_ERR_INVALID_ARG);
  }
}

{
  /* Quantized GEMMs (w8 + w4) vs the f32 GEMM of dequantized weights.
   * The GEMM convention groups the k axis (reduction dim) into blocks
   * of group_size rows, with one f32 scale per (group, column), so the
   * weight matrix is quantized per column here. */
  enum { M = 3, N = 5, K = 8 };
  float a[M * K];
  float w_f32[K * N];
  int8_t w_q[K * N];
  float scales[2 * N]; /* group_size 4 -> 2 k-groups */
  float c_q8[M * N];
  float c_q4[M * N];
  float c_ref[M * N];
  float w4[K * N]; /* dequantized for the f32 comparison */
  pai_quant_scheme_t scheme;
  uint64_t first;

  fill_random(a, M * K);
  fill_random(w_f32, K * N);

  scheme.bit_width = 8;
  scheme.is_signed = 1;
  scheme.group_size = 4;
  {
    /* Quantize each column (k elements) into k-row groups and scatter
     * the scales into the (group, column) layout the GEMM expects. */
    float tmp[K];
    int8_t qc[K];
    float sc[2];
    for (uint32_t j = 0; j < N; j++) {
      for (uint32_t t = 0; t < K; t++) {
        tmp[t] = w_f32[t * N + j];
      }
      CHECK_EQ_INT(pai_quantize_f32(tmp, K, &scheme, qc, sc, NULL), PAI_OK);
      for (uint32_t t = 0; t < K; t++) {
        w_q[t * N + j] = qc[t];
      }
      for (uint32_t g = 0; g < 2; g++) {
        scales[g * N + j] = sc[g];
      }
    }
  }

  /* w8 GEMM vs the f32 GEMM of the dequantized matrix. */
  CHECK_EQ_INT(pai_ref_gemm_w8_f32(M, N, K, a, w_q, scales, 4, c_q8), PAI_OK);
  for (uint32_t t = 0; t < K; t++) {
    for (uint32_t j = 0; j < N; j++) {
      w4[t * N + j] = (float)w_q[t * N + j] * scales[(t / 4) * N + j];
    }
  }
  CHECK_EQ_INT(pai_ref_gemm_f32(M, N, K, a, w4, c_ref), PAI_OK);
  CHECK_EQ_INT(pai_ref_compare_f32(c_q8, c_ref, M * N, 1e-3f, 1e-3f, &first),
               PAI_OK);

  /* w4: same but packed nibbles over the flat layout. */
  scheme.bit_width = 4;
  {
    int8_t qs[K * N]; /* per-element quanta, pre-packing */
    float tmp[K];
    int8_t qc[K];
    float sc[2];
    memset(w_q, 0, sizeof(w_q));
    for (uint32_t j = 0; j < N; j++) {
      for (uint32_t t = 0; t < K; t++) {
        tmp[t] = w_f32[t * N + j];
      }
      CHECK_EQ_INT(pai_quantize_f32(tmp, K, &scheme, qc, sc, NULL), PAI_OK);
      for (uint32_t t = 0; t < K; t++) {
        /* Unpack the flat-packed nibbles back to per-element quanta. */
        int8_t byte = qc[t / 2];
        int8_t nib = (t & 1) == 0 ? (int8_t)(byte & 0x0F)
                                  : (int8_t)((byte >> 4) & 0x0F);
        int32_t qv = (nib & 0x08) ? (int32_t)nib - 16 : (int32_t)nib;
        qs[t * N + j] = (int8_t)qv;
      }
      for (uint32_t g = 0; g < 2; g++) {
        scales[g * N + j] = sc[g];
      }
    }
    /* Flat packing matches pai_quantize_f32 (byte [f/2], nibble by
     * flat parity). */
    for (uint32_t t = 0; t < K; t++) {
      for (uint32_t j = 0; j < N; j++) {
        uint32_t flat = t * N + j;
        int8_t nib = (int8_t)(qs[flat] & 0x0F);
        if ((flat & 1) == 0) {
          w_q[flat / 2] = (int8_t)((w_q[flat / 2] & 0xF0) | (nib & 0x0F));
        } else {
          w_q[flat / 2] = (int8_t)((w_q[flat / 2] & 0x0F) | (nib << 4));
        }
      }
    }
    for (uint32_t t = 0; t < K; t++) {
      for (uint32_t j = 0; j < N; j++) {
        w4[t * N + j] = (float)qs[t * N + j] * scales[(t / 4) * N + j];
      }
    }
  }
  CHECK_EQ_INT(pai_ref_gemm_w4_f32(M, N, K, a, w_q, scales, 4, c_q4), PAI_OK);
  CHECK_EQ_INT(pai_ref_gemm_f32(M, N, K, a, w4, c_ref), PAI_OK);
  CHECK_EQ_INT(pai_ref_compare_f32(c_q4, c_ref, M * N, 1e-2f, 1e-2f, &first),
               PAI_OK);

  /* Per-tensor block (group_size 0 = one group over k per column). */
  scheme.bit_width = 8;
  scheme.group_size = 0;
  {
    float tmp[K];
    int8_t qc[K];
    float sc[1];
    for (uint32_t j = 0; j < N; j++) {
      for (uint32_t t = 0; t < K; t++) {
        tmp[t] = w_f32[t * N + j];
      }
      CHECK_EQ_INT(pai_quantize_f32(tmp, K, &scheme, qc, sc, NULL), PAI_OK);
      for (uint32_t t = 0; t < K; t++) {
        w_q[t * N + j] = qc[t];
      }
      scales[j] = sc[0];
    }
  }
  CHECK_EQ_INT(pai_ref_gemm_w8_f32(M, N, K, a, w_q, scales, 0, c_q8), PAI_OK);
  for (uint32_t t = 0; t < K; t++) {
    for (uint32_t j = 0; j < N; j++) {
      w4[t * N + j] = (float)w_q[t * N + j] * scales[j];
    }
  }
  CHECK_EQ_INT(pai_ref_gemm_f32(M, N, K, a, w4, c_ref), PAI_OK);
  CHECK_EQ_INT(pai_ref_compare_f32(c_q8, c_ref, M * N, 1e-3f, 1e-3f, &first),
               PAI_OK);

  /* Invalid args. */
  CHECK_EQ_INT(pai_ref_gemm_w8_f32(0, N, K, a, w_q, scales, 4, c_q8),
               PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_ref_gemm_w4_f32(M, N, K, NULL, w_q, scales, 4, c_q4),
               PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
