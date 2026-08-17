#include "ref_ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

pai_status_t
pai_ref_vecadd_f32(const float *a, const float *b, float *c, uint64_t n) {
  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

pai_status_t
pai_ref_vecsub_f32(const float *a, const float *b, float *c, uint64_t n) {
  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] - b[i];
  }
  return PAI_OK;
}

pai_status_t
pai_ref_clip_f32(const float *a, float *c, uint64_t n, float lo, float hi) {
  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  if (lo > hi) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
  }
  return PAI_OK;
}

pai_status_t
pai_ref_dot_f32(const float *a, const float *b, float *out, uint64_t n) {
  float acc = 0.0f;

  if (!out || n == 0 || (!a || !b)) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    acc += a[i] * b[i];
  }
  *out = acc;
  return PAI_OK;
}

pai_status_t
pai_ref_l1norm_f32(const float *a, float *out, uint64_t n) {
  float acc = 0.0f;

  if (!out || n == 0 || !a) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    acc += fabsf(a[i]);
  }
  *out = acc;
  return PAI_OK;
}

pai_status_t
pai_ref_l2norm_f32(const float *a, float *out, uint64_t n) {
  float acc = 0.0f;

  if (!out || n == 0 || !a) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    acc += a[i] * a[i];
  }
  *out = sqrtf(acc);
  return PAI_OK;
}

pai_status_t
pai_ref_gemv_f32(uint64_t m, uint64_t k, const float *a, const float *x,
                 float *y) {
  if (m == 0 || k == 0 || !a || !x || !y) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < m; i++) {
    float acc = 0.0f;
    for (uint64_t j = 0; j < k; j++) {
      acc += a[i * k + j] * x[j];
    }
    y[i] = acc;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_biasadd_f32(const float *a, const float *bias, float *c, uint64_t rows,
                    uint64_t cols) {
  if (rows == 0 || cols == 0 || !a || !bias || !c) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < rows; i++) {
    for (uint64_t j = 0; j < cols; j++) {
      c[i * cols + j] = a[i * cols + j] + bias[j];
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ref_vecmul_f32(const float *a, const float *b, float *c, uint64_t n) {
  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] * b[i];
  }
  return PAI_OK;
}

pai_status_t
pai_ref_relu_f32(const float *a, float *c, uint64_t n) {
  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] > 0.0f ? a[i] : 0.0f;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_softmax_f32(const float *a, float *c, uint64_t n) {
  float maxv;
  float sum = 0.0f;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }

  maxv = a[0];
  for (uint64_t i = 1; i < n; i++) {
    if (a[i] > maxv) {
      maxv = a[i];
    }
  }
  for (uint64_t i = 0; i < n; i++) {
    float e = expf(a[i] - maxv);
    c[i] = e;
    sum += e;
  }
  for (uint64_t i = 0; i < n; i++) {
    c[i] /= sum;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_rmsnorm_f32(const float *a, float *c, uint64_t n, float eps) {
  double acc = 0.0;
  float rms;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (eps == 0.0f) {
    eps = 1e-5f;
  }

  for (uint64_t i = 0; i < n; i++) {
    acc += (double)a[i] * (double)a[i];
  }
  rms = 1.0f / sqrtf((float)(acc / (double)n) + eps);
  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] * rms;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_rmsnorm_gamma_f32(const float *a, float *c, uint64_t n,
                           const float *gamma, float eps) {
  double acc = 0.0;
  float rms;

  if (n == 0 || (!a || !c || !gamma)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (eps == 0.0f) {
    eps = 1e-5f;
  }

  for (uint64_t i = 0; i < n; i++) {
    acc += (double)a[i] * (double)a[i];
  }
  rms = 1.0f / sqrtf((float)(acc / (double)n) + eps);
  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] * gamma[i] * rms;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_silu_f32(const float *a, float *c, uint64_t n) {
  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] / (1.0f + expf(-a[i]));
  }
  return PAI_OK;
}

pai_status_t
pai_ref_rope_f32(const float *x, uint64_t rows, uint64_t hd, uint64_t seq,
                 uint64_t heads, const float *cos_t, const float *sin_t,
                 uint64_t r2, float *out) {
  if (rows == 0 || hd == 0 || seq == 0 || heads == 0 || r2 == 0 ||
      2 * r2 > hd || rows % heads != 0 || (!x || !out || !cos_t || !sin_t)) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t row = 0; row < rows; row++) {
    /* Position-major layout: row = p * heads + h. */
    uint64_t p = row / heads;
    const float *xr = x + row * hd;
    float *or_ = out + row * hd;
    for (uint64_t i = 0; i < r2; i++) {
      float c = cos_t[p * r2 + i];
      float s = sin_t[p * r2 + i];
      float x0 = xr[i];
      float x1 = xr[i + r2];
      or_[i] = x0 * c - x1 * s;
      or_[i + r2] = x0 * s + x1 * c;
    }
    for (uint64_t i = 2 * r2; i < hd; i++) {
      or_[i] = xr[i];
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ref_rope_cossin_f32(uint64_t ctx, uint64_t r2, float base,
                        float *cos_t, float *sin_t) {
  if (ctx == 0 || r2 == 0 || !cos_t || !sin_t) {
    return PAI_ERR_INVALID_ARG;
  }
  if (base == 0.0f) {
    base = 10000.0f;
  }

  for (uint64_t p = 0; p < ctx; p++) {
    for (uint64_t i = 0; i < r2; i++) {
      /* dim = 2*r2; inv_freq[i] = base^(-2i/dim). */
      double theta = (double)p *
                     pow((double)base, -(2.0 * (double)i) / (2.0 * (double)r2));
      cos_t[p * r2 + i] = (float)cos(theta);
      sin_t[p * r2 + i] = (float)sin(theta);
    }
  }
  return PAI_OK;
}

/*
 * Causal multi-head self-attention. All buffers position-major:
 * q: seq x H x hd; k/v: seq x HK x hd; out: seq x H x hd. Head h at
 * position p reads KV head (h * HK) / H at position p.
 */
pai_status_t
pai_ref_attention_f32(uint64_t h, uint64_t hk, uint64_t seq, uint64_t hd,
                      const float *q, const float *k, const float *v,
                      float *out) {
  float *scores;
  float inv_scale;
  pai_status_t st = PAI_OK;
  uint64_t hh;
  uint64_t p;
  uint64_t i;

  if (h == 0 || hk == 0 || seq == 0 || hd == 0 || h % hk != 0 ||
      !q || !k || !v || !out) {
    return PAI_ERR_INVALID_ARG;
  }
  scores = (float *)malloc((size_t)seq * sizeof(float));
  if (scores == NULL) {
    return PAI_ERR_NOMEM;
  }

  inv_scale = 1.0f / sqrtf((float)hd);
  for (p = 0; p < seq; p++) {
    for (hh = 0; hh < h; hh++) {
      uint64_t kvh = (hh * hk) / h;
      const float *qp = q + (p * h + hh) * hd;
      float *op = out + (p * h + hh) * hd;
      float maxv;
      float sum = 0.0f;

      /* Scores for positions 0..p (causal), scaled. */
      for (i = 0; i <= p; i++) {
        const float *kp = k + (i * hk + kvh) * hd;
        float dot = 0.0f;
        for (uint64_t d = 0; d < hd; d++) {
          dot += qp[d] * kp[d];
        }
        scores[i] = dot * inv_scale;
      }

      /* Row softmax over scores[0..p]. */
      maxv = scores[0];
      for (i = 1; i <= p; i++) {
        if (scores[i] > maxv) {
          maxv = scores[i];
        }
      }
      for (i = 0; i <= p; i++) {
        float e = expf(scores[i] - maxv);
        scores[i] = e;
        sum += e;
      }
      for (i = 0; i <= p; i++) {
        scores[i] /= sum;
      }

      /* Weighted sum of v. */
      for (uint64_t d = 0; d < hd; d++) {
        float acc = 0.0f;
        for (i = 0; i <= p; i++) {
          acc += scores[i] * v[(i * hk + kvh) * hd + d];
        }
        op[d] = acc;
      }
    }
  }

  free(scores);
  return st;
}

pai_status_t
pai_ref_layernorm_f32(const float *a, float *c, uint64_t n,
                      const float *gamma, const float *beta, float eps) {
  double sum = 0.0;
  double sumsq = 0.0;
  float mean;
  float inv_std;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (eps == 0.0f) {
    eps = 1e-5f;
  }

  for (uint64_t i = 0; i < n; i++) {
    sum += (double)a[i];
    sumsq += (double)a[i] * (double)a[i];
  }
  mean = (float)(sum / (double)n);
  inv_std = 1.0f / sqrtf((float)(sumsq / (double)n - (double)mean * (double)mean) +
                         eps);
  for (uint64_t i = 0; i < n; i++) {
    float y = (a[i] - mean) * inv_std;
    if (gamma != NULL) {
      y *= gamma[i];
    }
    if (beta != NULL) {
      y += beta[i];
    }
    c[i] = y;
  }
  return PAI_OK;
}

pai_status_t
pai_ref_concat_f32(const float *a, const float *b, float *c, uint64_t na,
                   uint64_t nb) {
  if ((!a || !b || !c) && (na + nb) != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  memcpy(c, a, (size_t)na * sizeof(float));
  memcpy(c + na, b, (size_t)nb * sizeof(float));
  return PAI_OK;
}

pai_status_t
pai_ref_copy_f32(const float *a, float *c, uint64_t n) {
  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  if (a != c && n != 0) {
    memcpy(c, a, (size_t)n * sizeof(float));
  }
  return PAI_OK;
}

pai_status_t
pai_ref_scale_f32(float *a, float alpha, uint64_t n) {
  if (!a && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    a[i] = alpha * a[i];
  }
  return PAI_OK;
}

pai_status_t
pai_ref_gemm_f32(uint64_t m, uint64_t n, uint64_t k, const float *a,
                 const float *b, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !b || !c) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t l = 0; l < k; l++) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ref_gemm_w8_f32(uint64_t m, uint64_t n, uint64_t k, const float *a,
                     const int8_t *w_q, const float *b_scales,
                     uint64_t group_size, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !w_q || !b_scales || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  if (group_size == 0 || group_size > k) {
    group_size = k;
  }

  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        float s = b_scales[(t / group_size) * n + j];
        sum += a[i * k + t] * (float)w_q[t * n + j] * s;
      }
      c[i * n + j] = sum;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ref_gemm_w4_f32(uint64_t m, uint64_t n, uint64_t k, const float *a,
                     const int8_t *w_q4, const float *b_scales,
                     uint64_t group_size, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !w_q4 || !b_scales || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  if (group_size == 0 || group_size > k) {
    group_size = k;
  }

  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        /* Flat packing: element (t, j) sits at flat index t*n+j;
         * byte = w_q4[flat/2], nibble selected by flat parity. */
        uint64_t flat = t * n + j;
        int8_t byte = w_q4[flat / 2];
        int8_t nib = (flat & 1) == 0 ? (int8_t)(byte & 0x0F)
                                     : (int8_t)((byte >> 4) & 0x0F);
        int32_t qv = (nib & 0x08) ? (int32_t)nib - 16 : (int32_t)nib;
        float s = b_scales[(t / group_size) * n + j];
        sum += a[i * k + t] * (float)qv * s;
      }
      c[i * n + j] = sum;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ref_memset16(void *dst, const uint32_t pattern[4], uint64_t blocks) {
  uint32_t *out = (uint32_t *)dst;

  if (!out || !pattern) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < blocks; i++) {
    out[i * 4 + 0] = pattern[0];
    out[i * 4 + 1] = pattern[1];
    out[i * 4 + 2] = pattern[2];
    out[i * 4 + 3] = pattern[3];
  }
  return PAI_OK;
}

pai_status_t
pai_ref_compare_f32(const float *a, const float *b, uint64_t n, float abs_tol,
                    float rel_tol, uint64_t *out_first_mismatch) {
  if (out_first_mismatch) {
    *out_first_mismatch = UINT64_MAX;
  }
  if ((!a || !b) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0; i < n; i++) {
    float diff = fabsf(a[i] - b[i]);
    float scale = fabsf(b[i]) * rel_tol;
    float tol = abs_tol > scale ? abs_tol : scale;

    if (diff > tol) {
      if (out_first_mismatch) {
        *out_first_mismatch = i;
      }
      return PAI_ERR_MISMATCH;
    }
  }
  return PAI_OK;
}
