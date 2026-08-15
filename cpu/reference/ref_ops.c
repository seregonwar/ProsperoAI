#include "ref_ops.h"

#include <math.h>
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
