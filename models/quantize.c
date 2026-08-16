#include "quantize.h"

#include <math.h>
#include <string.h>

static pai_status_t
quant_group_size(const pai_quant_scheme_t *scheme, uint64_t n,
                 uint64_t *out_gs) {
  uint64_t gs;

  if (scheme == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (scheme->bit_width != 8 && scheme->bit_width != 4) {
    return PAI_ERR_UNSUPPORTED; /* v0 schemes only (§15)                 */
  }
  if (!scheme->is_signed) {
    return PAI_ERR_UNSUPPORTED; /* v0 is signed-symmetric only          */
  }
  gs = scheme->group_size == PAI_QUANT_GROUP_AUTO ? n : scheme->group_size;
  if (gs == 0) {
    return PAI_ERR_INVALID_ARG; /* n == 0 or absurd group size          */
  }
  *out_gs = gs;
  return PAI_OK;
}

pai_status_t
pai_quant_scheme_check(const pai_quant_scheme_t *scheme) {
  uint64_t gs;
  return quant_group_size(scheme, 1, &gs);
}

uint64_t
pai_quant_num_groups(uint64_t n, const pai_quant_scheme_t *scheme) {
  uint64_t gs;
  uint64_t groups;

  if (scheme == NULL || n == 0) {
    return 0;
  }
  if (quant_group_size(scheme, n, &gs) != PAI_OK) {
    return 0;
  }
  groups = n / gs;
  if (n % gs != 0) {
    groups++;
  }
  return groups;
}

uint64_t
pai_quant_value_bytes(uint64_t n, const pai_quant_scheme_t *scheme) {
  if (scheme == NULL || n == 0) {
    return 0;
  }
  if (scheme->bit_width == 4) {
    return (n + 1) / 2; /* two values per byte                          */
  }
  return n; /* 8-bit                                                      */
}

uint64_t
pai_quant_scale_bytes(uint64_t n, const pai_quant_scheme_t *scheme) {
  return pai_quant_num_groups(n, scheme) * sizeof(float);
}

uint64_t
pai_quant_weights_bytes(uint64_t n, const pai_quant_scheme_t *scheme) {
  return pai_quant_value_bytes(n, scheme) + pai_quant_scale_bytes(n, scheme);
}

pai_status_t
pai_quantize_f32(const float *src, uint64_t n, const pai_quant_scheme_t *scheme,
                 int8_t *out_q, float *out_scales, uint64_t *out_num_groups) {
  uint64_t gs;
  pai_status_t st;
  int32_t qmax;
  uint64_t g;
  uint64_t t = 0;

  if (src == NULL || out_q == NULL || out_scales == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  st = quant_group_size(scheme, n, &gs);
  if (st != PAI_OK) {
    return st;
  }
  qmax = scheme->bit_width == 8 ? 127 : 7;

  /* Zero padding nibbles (odd trailing element) so the packed buffer
   * never contains uninitialized bytes. */
  memset(out_q, 0, (size_t)pai_quant_value_bytes(n, scheme));

  for (g = 0; t < n; g++, t += gs) {
    uint64_t t1 = t + gs < n ? t + gs : n;
    float max_abs = 0.0f;
    float scale;
    uint64_t i;

    for (i = t; i < t1; i++) {
      float av = fabsf(src[i]);
      if (av > max_abs) {
        max_abs = av;
      }
    }
    scale = max_abs > 0.0f ? max_abs / (float)qmax : 0.0f;
    out_scales[g] = scale;
    for (i = t; i < t1; i++) {
      int32_t q;
      if (scale > 0.0f) {
        q = (int32_t)lrintf(src[i] / scale);
        if (q > qmax) {
          q = qmax;
        }
        if (q < -qmax) {
          q = -qmax;
        }
      } else {
        q = 0;
      }
      if (scheme->bit_width == 4) {
        /* Two values per byte: element i -> low nibble when (i % 2 == 0),
         * high nibble otherwise. Bytes are written as signed values
         * sign-extended on read, so a negative nibble is stored as its
         * 4-bit two's-complement representation. */
        int8_t *byte = out_q + (i / 2);
        int8_t nib = (int8_t)(q & 0x0F);
        if ((i & 1) == 0) {
          *byte = (int8_t)((*byte & 0xF0) | (nib & 0x0F));
        } else {
          *byte = (int8_t)((*byte & 0x0F) | (nib << 4));
        }
      } else {
        out_q[i] = (int8_t)q;
      }
    }
  }
  if (out_num_groups != NULL) {
    *out_num_groups = g;
  }
  return PAI_OK;
}

pai_status_t
pai_dequantize_f32(const int8_t *q, const float *scales, uint64_t n,
                   const pai_quant_scheme_t *scheme, float *out) {
  uint64_t gs;
  pai_status_t st;
  uint64_t g;
  uint64_t t = 0;

  if (q == NULL || scales == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  st = quant_group_size(scheme, n, &gs);
  if (st != PAI_OK) {
    return st;
  }

  for (g = 0; t < n; g++, t += gs) {
    uint64_t t1 = t + gs < n ? t + gs : n;
    float scale = scales[g];
    uint64_t i;

    for (i = t; i < t1; i++) {
      int32_t qv;
      if (scheme->bit_width == 4) {
        int8_t byte = q[i / 2];
        int8_t nib = (i & 1) == 0 ? (int8_t)(byte & 0x0F)
                                  : (int8_t)((byte >> 4) & 0x0F);
        /* Sign-extend the 4-bit value. */
        qv = (nib & 0x08) ? (int32_t)nib - 16 : (int32_t)nib;
      } else {
        qv = q[i];
      }
      out[i] = scale * (float)qv;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_quant_error_f32(const float *a, const float *b, uint64_t n,
                    float *out_max_abs, float *out_rmse) {
  double acc = 0.0;
  float max_abs = 0.0f;

  if (a == NULL || b == NULL || out_max_abs == NULL || out_rmse == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > max_abs) {
      max_abs = d;
    }
    acc += (double)d * (double)d;
  }
  *out_max_abs = max_abs;
  *out_rmse = n > 0 ? (float)sqrt(acc / (double)n) : 0.0f;
  return PAI_OK;
}
