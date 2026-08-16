#include "host_kernels.h"

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
