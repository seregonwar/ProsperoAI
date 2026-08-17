#include "host_kernels.h"

#include <math.h>
#include <string.h>

static uint64_t
pai_ud64(const uint32_t user_data[16], uint32_t lo_index) {
  return (uint64_t)user_data[lo_index] |
         ((uint64_t)user_data[lo_index + 1] << 32);
}

pai_status_t
pai_host_kernel_vecadd(void *ctx, const uint32_t user_data[16],
                       uint32_t threads_x, uint32_t group_x) {
  float *a = (float *)(uintptr_t)pai_ud64(user_data, 0);
  float *b = (float *)(uintptr_t)pai_ud64(user_data, 2);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t n = user_data[6];

  (void)ctx;
  (void)group_x; /* M0 kernel dispatches a single group */

  for (uint32_t i = 0; i < n && i < threads_x; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_memset16(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t blocks = user_data[4];
  uint32_t pattern[4];

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  pattern[0] = user_data[5];
  pattern[1] = user_data[6];
  pattern[2] = user_data[7];
  pattern[3] = user_data[8];

  for (uint32_t i = 0; i < blocks; i++) {
    dst[i * 4 + 0] = pattern[0];
    dst[i * 4 + 1] = pattern[1];
    dst[i * 4 + 2] = pattern[2];
    dst[i * 4 + 3] = pattern[3];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store_const(void *ctx, const uint32_t user_data[16],
                            uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 0);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    dst[i] = 0xABCD1234u;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store_const64(void *ctx, const uint32_t user_data[16],
                              uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    dst[i] = 0xCAFEF00Du;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store64_x2(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 2; i++) {
    dst[i] = 0xBEADF00Du;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store64_x4(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 4; i++) {
    dst[i] = 0xF00DFEEDu;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_bisect_store(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 4; i++) {
    dst[i] = 0xDEADBEEFu;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store64_smem(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 4; i++) {
    dst[i] = 0xC0FFEEEEu;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store64_gold(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 4; i++) {
    dst[i] = 0xB0DD00D1u;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_loadstore_gold(void *ctx, const uint32_t user_data[16],
                               uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = a[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store64_v0(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x * 4; i++) {
    dst[i] = 0x12345678u;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_copy_v0(void *ctx, const uint32_t user_data[16],
                        uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = a[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_store_dw(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  uint32_t *dst = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    dst[i] = 0xABCDDCBAu;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_vecadd_v0(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  float *a = (float *)(uintptr_t)pai_ud64(user_data, 2);
  float *b = (float *)(uintptr_t)pai_ud64(user_data, 4);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 6);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_mubufload(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 0);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = a[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_arith(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x) {
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 2);
  float k;
  memcpy(&k, &user_data[4], sizeof(k));

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = (float)i + k;
  }
  return PAI_OK;
}

/* F-batch ABI: k at user_data 0, C at 2-3. */
pai_status_t
pai_host_kernel_fbatch(void *ctx, const uint32_t user_data[16],
                       uint32_t threads_x, uint32_t group_x) {
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 2);
  float k;
  memcpy(&k, &user_data[0], sizeof(k));

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = (float)i + k;
  }
  return PAI_OK;
}

/* G7: x4 per-thread tid broadcast — c[4i..4i+3] = i. */
pai_status_t
pai_host_kernel_g7(void *ctx, const uint32_t user_data[16],
                   uint32_t threads_x, uint32_t group_x) {
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i * 4 + 0] = i;
    c[i * 4 + 1] = i;
    c[i * 4 + 2] = i;
    c[i * 4 + 3] = i;
  }
  return PAI_OK;
}

/* G8: per-thread int arithmetic — c[4i..4i+3] = tid*4 + k. */
pai_status_t
pai_host_kernel_g8(void *ctx, const uint32_t user_data[16],
                   uint32_t threads_x, uint32_t group_x) {
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t k = user_data[4];

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i * 4 + 0] = i * 4 + k;
    c[i * 4 + 1] = i * 4 + k;
    c[i * 4 + 2] = i * 4 + k;
    c[i * 4 + 3] = i * 4 + k;
  }
  return PAI_OK;
}

/* Integer SAXPY matching saxpy.s: C[i] = 3 * A[i] + B[i] (uint32 wrap). */
pai_status_t
pai_host_kernel_saxpy(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x) {
  uint32_t *pack = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;

  for (uint32_t i = 0; i < group_x; i++) {
    c[i] = 3u * pack[2u * i] + pack[2u * i + 1u];
  }
  return PAI_OK;
}

/* M1B serial uint32 dot matching dot_serial_u32.s (uint32 wrap). */
pai_status_t
pai_host_kernel_dot_serial_u32(void *ctx, const uint32_t user_data[16],
                               uint32_t threads_x, uint32_t group_x) {
  uint32_t *pack = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t n = pack[0];
  uint32_t sum = 0;

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < n; i++) {
    sum += pack[2u + 2u * i] * pack[3u + 2u * i];
  }
  c[0] = sum;
  return PAI_OK;
}

/* M1D serial-per-row GEMV matching gemv_serial_u32.s (uint32 wrap). */
pai_status_t
pai_host_kernel_gemv_serial_u32(void *ctx, const uint32_t user_data[16],
                                uint32_t threads_x, uint32_t group_x) {
  uint32_t *w = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *y = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t kdim = w[0];
  uint32_t *x = (uint32_t *)(uintptr_t)pai_ud64(w, 2);

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    uint32_t acc = 0;
    const uint32_t *row = w + 4u + g * kdim;
    for (uint32_t k = 0; k < kdim; k++) {
      acc += row[k] * x[k];
    }
    y[g] = acc;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_loadstore(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 0);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i] = a[i];
  }
  return PAI_OK;
}

/* h1: copy a[i] -> c[4i..4i+3] (A in T# words 0-1, C at 4-5). */
pai_status_t
pai_host_kernel_h1(void *ctx, const uint32_t user_data[16],
                   uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 0);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i * 4 + 0] = a[i];
    c[i * 4 + 1] = a[i];
    c[i * 4 + 2] = a[i];
    c[i * 4 + 3] = a[i];
  }
  return PAI_OK;
}

/* h3: c[4i..4i+3] = a[i] + b[i] (A T# 0-1, B T# 4-5, C 8-9). */
pai_status_t
pai_host_kernel_h3(void *ctx, const uint32_t user_data[16],
                   uint32_t threads_x, uint32_t group_x) {
  uint32_t *a = (uint32_t *)(uintptr_t)pai_ud64(user_data, 0);
  uint32_t *b = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 8);

  (void)ctx;
  (void)group_x;

  for (uint32_t i = 0; i < threads_x; i++) {
    c[i * 4 + 0] = a[i] + b[i];
    c[i * 4 + 1] = a[i] + b[i];
    c[i * 4 + 2] = a[i] + b[i];
    c[i * 4 + 3] = a[i] + b[i];
  }
  return PAI_OK;
}

/* T4 elementwise float ops (t4_ops.s G42-G46): packed (a,b) pairs at
 * ud 2-3, C at ud 4-5, one group per element, float semantics. */
static pai_status_t
pai_host_kernel_t4_ew(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x, uint32_t op) {
  const float *ab = (const float *)(uintptr_t)pai_ud64(user_data, 2);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    float av = ab[2u * g];
    float bv = ab[2u * g + 1u];
    switch (op) {
    case 0:
      c[g] = av + bv;
      break;
    case 1:
      c[g] = av - bv;
      break;
    case 2:
      c[g] = av * bv;
      break;
    case 3:
      c[g] = av > 0.0f ? av : 0.0f;
      break;
    default:
      c[g] = av < 0.0f ? 0.0f : (av > 1.0f ? 1.0f : av);
      break;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_t4_add1d(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_t4_ew(ctx, user_data, threads_x, group_x, 0);
}

pai_status_t
pai_host_kernel_t4_sub1d(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_t4_ew(ctx, user_data, threads_x, group_x, 1);
}

pai_status_t
pai_host_kernel_t4_mul1d(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_t4_ew(ctx, user_data, threads_x, group_x, 2);
}

pai_status_t
pai_host_kernel_t4_relu(void *ctx, const uint32_t user_data[16],
                        uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_t4_ew(ctx, user_data, threads_x, group_x, 3);
}

pai_status_t
pai_host_kernel_t4_clip(void *ctx, const uint32_t user_data[16],
                        uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_t4_ew(ctx, user_data, threads_x, group_x, 4);
}

/* T4 biasadd (G47): header at ud 2-3 [cols, pad, a_lo, a_hi,
 * bias_lo, bias_hi], C at ud 4-5; one group per cell:
 * c[g] = a[g] + bias[g % cols], groups = rows*cols. */
pai_status_t
pai_host_kernel_t4_biasadd(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  const float *a = (const float *)(uintptr_t)pai_ud64(h, 2);
  const float *bias = (const float *)(uintptr_t)pai_ud64(h, 4);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t cols = h[0];

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    c[g] = a[g] + bias[g % cols];
  }
  return PAI_OK;
}

/* T4 matmul (G48): header at ud 2-3 [K, N, a_lo, a_hi, b_lo, b_hi],
 * C at ud 4-5; one group per cell, groups = rows*N:
 * c[g] = sum_k a[i*K+k] * b[k*N+j], i = g/N, j = g%N. */
pai_status_t
pai_host_kernel_t4_matmul(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  const float *a = (const float *)(uintptr_t)pai_ud64(h, 2);
  const float *b = (const float *)(uintptr_t)pai_ud64(h, 4);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t kdim = h[0];
  uint32_t n = h[1];

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    uint32_t i = g / n;
    uint32_t j = g % n;
    const float *row = a + i * kdim;
    float acc = 0.0f;
    for (uint32_t k = 0; k < kdim; k++) {
      acc += row[k] * b[k * n + j];
    }
    c[g] = acc;
  }
  return PAI_OK;
}

/* T4 integer elementwise ops (int_ops.s G49-G53): packed (a,b) pairs at
 * ud 2-3, C at ud 4-5, one group per element, u32 wrap. */
static pai_status_t
pai_host_kernel_int_ew(void *ctx, const uint32_t user_data[16],
                       uint32_t threads_x, uint32_t group_x, uint32_t op) {
  const uint32_t *ab = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    uint32_t av = ab[2u * g];
    uint32_t bv = ab[2u * g + 1u];
    switch (op) {
    case 0:
      c[g] = av + bv;
      break;
    case 1:
      c[g] = av - bv;
      break;
    case 2:
      c[g] = av * bv;
      break;
    case 3:
      c[g] = av > 0u ? av : 0u;
      break;
    default:
      c[g] = av < 0u ? 0u : (av > 1u ? 1u : av);
      break;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_int_add2d(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_int_ew(ctx, user_data, threads_x, group_x, 0);
}

pai_status_t
pai_host_kernel_int_sub1d(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_int_ew(ctx, user_data, threads_x, group_x, 1);
}

pai_status_t
pai_host_kernel_int_mul1d(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_int_ew(ctx, user_data, threads_x, group_x, 2);
}

pai_status_t
pai_host_kernel_int_relu(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_int_ew(ctx, user_data, threads_x, group_x, 3);
}

pai_status_t
pai_host_kernel_int_clip(void *ctx, const uint32_t user_data[16],
                         uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_int_ew(ctx, user_data, threads_x, group_x, 4);
}

/* T4 integer matmul (G54): header at ud 2-3 [K, N, a_lo, a_hi,
 * b_lo, b_hi], C at ud 4-5; one group per cell, groups = rows*N:
 * c[g] = sum_k a[i*K+k] * b[k*N+j] (u32 wrap), i = g/N, j = g%N. */
pai_status_t
pai_host_kernel_ramp(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x) {
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 2);
  float k, base;
  memcpy(&k, &user_data[4], sizeof(k));
  memcpy(&base, &user_data[5], sizeof(base));

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  /* Silicon: only lanes 0-7 of the 32-thread wave store, and the
   * value path bakes in the G15 store formula (tid*4+3), so lane i
   * stores base + k*(4i+3) - same convention as G35's check. */
  for (uint32_t i = 0; i < 8; i++) {
    c[i] = base + k * (float)(4 * i + 3);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_ramp2(void *ctx, const uint32_t user_data[16],
                       uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  float *c = (float *)(uintptr_t)pai_ud64(user_data, 4);
  float k, base;
  memcpy(&k, &h[0], sizeof(k));
  memcpy(&base, &h[1], sizeof(base));

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = base + k * (float)(4 * i + 3);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_lanepick(void *ctx, const uint32_t user_data[16],
                              uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = h[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_blockdump(void *ctx, const uint32_t user_data[16],
                               uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = h[i];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_vpick(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = h[0];
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_vpick2(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = h[0];
  }
  return PAI_OK;
}

/* G64 v_movrels in-ceiling (v7..v14 + m0=7): lane i selects h[7+i],
 * so the 8 storing lanes write h[7..14]. Mirrors the real kernel. */
pai_status_t
pai_host_kernel_movrels(void *ctx, const uint32_t user_data[16],
                        uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  for (uint32_t i = 0; i < 8; i++) {
    c[i] = h[7 + i];
  }
  return PAI_OK;
}

/* G65/G66 wave-parallel cos/sin ramp: scale at ud[4] as float bits,
 * C at ud[2:3]. Value-path quirk (G35/G55): lane index reads as
 * (4i+3), so c[i] = cosf(scale*(4i+3)) / sinf(scale*(4i+3)).
 * op 0 = cos, 1 = sin. */
static pai_status_t
pai_host_kernel_cossin(void *ctx, const uint32_t user_data[16],
                       uint32_t threads_x, uint32_t group_x,
                       uint32_t op) {
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  float scale;

  (void)ctx;
  (void)threads_x;
  (void)group_x;

  /* Turns convention (HW-verified on 9.40): v_cos/v_sin read their
   * operand in full turns, i.e. the value path multiplies by 2*pi.
   * The mirror must produce cos/sin(2*pi*scale*(4i+3)). */
  memcpy(&scale, &user_data[4], 4);
  for (uint32_t i = 0; i < 8; i++) {
    float theta = 6.2831853f * scale * (float)(4u * i + 3u);
    float val = (op == 0) ? cosf(theta) : sinf(theta);
    memcpy(&c[i], &val, 4);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_cossin_cos(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_cossin(ctx, user_data, threads_x, group_x, 0);
}

pai_status_t
pai_host_kernel_cossin_sin(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_cossin(ctx, user_data, threads_x, group_x, 1);
}

/* G67/G68 on-GPU RoPE table generator mirror (ropegen.s): header
 * (r2, ctx, theta_turns[0..ctx*r2-1]) at ud[2:3], C at ud[4:5].
 * Serial-per-element: one group per element e, cos entry writes
 * c[e] = cos(2*pi*theta_turns[e]), sin entry the ctx*r2 half. */
static pai_status_t
pai_host_kernel_ropegen(void *ctx, const uint32_t user_data[16],
                        uint32_t threads_x, uint32_t group_x,
                        uint32_t op) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t ctxr = h[1];

  (void)ctx;
  (void)threads_x;

  for (uint32_t e = 0; e < group_x; e++) {
    float tt;
    memcpy(&tt, &h[2 + e], 4);
    float theta = 6.2831853f * tt;
    float val = (op == 0) ? cosf(theta) : sinf(theta);
    uint32_t idx = e + (op == 1 ? ctxr * h[0] : 0u);
    memcpy(&c[idx], &val, 4);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_ropegen_cos(void *ctx, const uint32_t user_data[16],
                            uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_ropegen(ctx, user_data, threads_x, group_x, 0);
}

pai_status_t
pai_host_kernel_ropegen_sin(void *ctx, const uint32_t user_data[16],
                            uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_ropegen(ctx, user_data, threads_x, group_x, 1);
}

/* G71/G72 serial v_rsq/v_exp mirrors (nlexp.s): header (n, pad,
 * x[]) at ud[2:3], C at ud[4:5], one group per element. The mirror
 * implements the MATH conventions (1/sqrtf, expf); the payload locks
 * the 9.40 HW convention empirically (v_exp may be 2^x). */
static pai_status_t
pai_host_kernel_nlexp(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x, uint32_t op) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;

  for (uint32_t e = 0; e < group_x; e++) {
    float x;
    float val;
    memcpy(&x, &h[2 + e], 4);
    val = (op == 0) ? (1.0f / sqrtf(x)) : expf(x);
    memcpy(&c[e], &val, 4);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_nlexp_rsq(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_nlexp(ctx, user_data, threads_x, group_x, 0);
}

pai_status_t
pai_host_kernel_nlexp_exp(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_nlexp(ctx, user_data, threads_x, group_x, 1);
}

/* G75: v_rcp_f32 serial probe (nlexp.s nlexp_rcp) - same ABI as
 * rsq/exp: c[e] = 1/x[e]. */
pai_status_t
pai_host_kernel_nlexp_rcp(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);

  (void)ctx;
  (void)threads_x;

  for (uint32_t e = 0; e < group_x; e++) {
    float x;
    float val;
    memcpy(&x, &h[2 + e], 4);
    val = 1.0f / x;
    memcpy(&c[e], &val, 4);
  }
  return PAI_OK;
}

/* G76: v_max/v_min serial probe (nlexp.s nlexp_max/min) - header
 * [n, pad, x[0..n-1], y[0..n-1]], c[e] = max(x[e], y[e]) or min. */
static pai_status_t
pai_host_kernel_nlexp_minmax(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x,
                             uint32_t op) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t n = h[0];

  (void)ctx;
  (void)threads_x;

  for (uint32_t e = 0; e < group_x; e++) {
    float x, y, val;
    memcpy(&x, &h[2 + e], 4);
    memcpy(&y, &h[2 + n + e], 4);
    val = (op == 0) ? (x > y ? x : y) : (x < y ? x : y);
    memcpy(&c[e], &val, 4);
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_nlexp_max(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_nlexp_minmax(ctx, user_data, threads_x, group_x,
                                      0);
}

pai_status_t
pai_host_kernel_nlexp_min(void *ctx, const uint32_t user_data[16],
                          uint32_t threads_x, uint32_t group_x) {
  return pai_host_kernel_nlexp_minmax(ctx, user_data, threads_x, group_x,
                                      1);
}

/* G40: VALU float GEMV serial-per-row (fgemv_serial.s). W header at
 * ud[2:3]: [K, pad, x_lo, x_hi, W[g*K+k]...]; C at ud[4:5]; TGID_X
 * = row g, one group per row: y[g] = sum_k W[g,k] * x[k]. Same float
 * accumulation order as the shader (v_add_f32 chain) - host compare
 * uses the tolerant 1e-4 compare like the payload. */
pai_status_t
pai_host_kernel_fgemv(void *ctx, const uint32_t user_data[16],
                      uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  const float *w = (const float *)(const void *)(h + 4);
  const float *x = (const float *)(uintptr_t)pai_ud64(h, 2);
  float *y = (float *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t kdim = h[0];

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    const float *row = w + g * kdim;
    float acc = 0.0f;
    for (uint32_t k = 0; k < kdim; k++) {
      acc += row[k] * x[k];
    }
    y[g] = acc;
  }
  return PAI_OK;
}

pai_status_t
pai_host_kernel_int_matmul(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x) {
  const uint32_t *h = (const uint32_t *)(uintptr_t)pai_ud64(user_data, 2);
  const uint32_t *a = (const uint32_t *)(uintptr_t)pai_ud64(h, 2);
  const uint32_t *b = (const uint32_t *)(uintptr_t)pai_ud64(h, 4);
  uint32_t *c = (uint32_t *)(uintptr_t)pai_ud64(user_data, 4);
  uint32_t kdim = h[0];
  uint32_t n = h[1];

  (void)ctx;
  (void)threads_x;

  for (uint32_t g = 0; g < group_x; g++) {
    uint32_t i = g / n;
    uint32_t j = g % n;
    const uint32_t *row = a + i * kdim;
    uint32_t acc = 0;
    for (uint32_t k = 0; k < kdim; k++) {
      acc += row[k] * b[k * n + j];
    }
    c[g] = acc;
  }
  return PAI_OK;
}
