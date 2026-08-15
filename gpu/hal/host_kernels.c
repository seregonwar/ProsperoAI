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
