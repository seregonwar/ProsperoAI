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

#endif /* PAI_GPU_HOST_KERNELS_H */
