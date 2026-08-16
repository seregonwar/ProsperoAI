/*
 * ProsperoAI — host kernels
 *
 * CPU emulations of the gfx1013 shaders shipped in gpu/kernels/.
 * Semantics must mirror the GPU assembly one-to-one: same user-data
 * ABI, same thread/group decomposition.
 */

#ifndef PAI_GPU_HOST_KERNELS_H
#define PAI_GPU_HOST_KERNELS_H

#include <hal/hal.h>

/* vecadd: user_data 0-1 = A, 2-3 = B, 4-5 = C, 6 = n (f32 buffers). */
pai_status_t pai_host_kernel_vecadd(void *ctx, const uint32_t user_data[16],
                                    uint32_t threads_x, uint32_t group_x);

/* memset16: user_data 2-3 = dst, 4 = blocks, 5-8 = 16-byte pattern. */
pai_status_t pai_host_kernel_memset16(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);

/* store_const (E1): user_data 0-1 = dst; each thread writes a constant. */
pai_status_t pai_host_kernel_store_const(void *ctx, const uint32_t user_data[16],
                                         uint32_t threads_x, uint32_t group_x);

/* store_const64 (E5/E7): psbc ABI, user_data 2-3 = dst. */
pai_status_t
pai_host_kernel_store_const64(void *ctx, const uint32_t user_data[16],
                              uint32_t threads_x, uint32_t group_x);

/* store64_x2 (E12): 2 dwords per thread. */
pai_status_t pai_host_kernel_store64_x2(void *ctx,
                                        const uint32_t user_data[16],
                                        uint32_t threads_x, uint32_t group_x);

/* store64_x4 (E13): 16 bytes per thread. */
pai_status_t pai_host_kernel_store64_x4(void *ctx,
                                        const uint32_t user_data[16],
                                        uint32_t threads_x, uint32_t group_x);

/* bisect store (E19): 0xDEADBEEF, 16 bytes per thread. */
pai_status_t
pai_host_kernel_bisect_store(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x);

/* store64_smem (E22): 0xC0FFEEEE, 16 bytes per thread. */
pai_status_t
pai_host_kernel_store64_smem(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x);

/* store64_gold (E30): 0xB0DD00D1, 16 bytes per thread. */
pai_status_t
pai_host_kernel_store64_gold(void *ctx, const uint32_t user_data[16],
                             uint32_t threads_x, uint32_t group_x);

/* loadstore_gold (E31): A at 2-3, C at 4-5. */
pai_status_t
pai_host_kernel_loadstore_gold(void *ctx, const uint32_t user_data[16],
                               uint32_t threads_x, uint32_t group_x);

/* store64_v0 (E32): 0x12345678, 16 bytes per thread. */
pai_status_t
pai_host_kernel_store64_v0(void *ctx, const uint32_t user_data[16],
                           uint32_t threads_x, uint32_t group_x);

/* copy_v0 / load_v1 (E34/E35): A at 2-3, C at 4-5. */
pai_status_t pai_host_kernel_copy_v0(void *ctx, const uint32_t user_data[16],
                                     uint32_t threads_x, uint32_t group_x);

/* store_dw (E36): dst at 2-3. */
pai_status_t pai_host_kernel_store_dw(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);

/* vecadd_v0 (E37): A at 2-3, B at 4-5, C at 6-7. */
pai_status_t pai_host_kernel_vecadd_v0(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

/* mubufload (E38): A base in T# (0-3), C at 4-5. */
pai_status_t pai_host_kernel_mubufload(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

/* arith (E39): C at 2-3, float k at 4. */
pai_status_t pai_host_kernel_arith(void *ctx, const uint32_t user_data[16],
                                   uint32_t threads_x, uint32_t group_x);

/* loadstore (E2): user_data 0-1 = A, 2-3 = C; c[i] = a[i]. */
pai_status_t pai_host_kernel_loadstore(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

#endif /* PAI_GPU_HOST_KERNELS_H */
