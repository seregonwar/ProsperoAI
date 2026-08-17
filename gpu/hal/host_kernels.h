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

/* fbatch (F1-F4): float k at 0, C at 2-3. */
pai_status_t pai_host_kernel_fbatch(void *ctx, const uint32_t user_data[16],
                                    uint32_t threads_x, uint32_t group_x);

/* g7: c[4i..4i+3] = i. */
pai_status_t pai_host_kernel_g7(void *ctx, const uint32_t user_data[16],
                                uint32_t threads_x, uint32_t group_x);

/* g8: c[4i..4i+3] = tid*4 + k. */
pai_status_t pai_host_kernel_g8(void *ctx, const uint32_t user_data[16],
                                uint32_t threads_x, uint32_t group_x);

/* Integer SAXPY: packed AB at ud[2:3], C at ud[4:5], C[i] = 3*A[i]+B[i]. */
pai_status_t pai_host_kernel_saxpy(void *ctx, const uint32_t user_data[16],
                                   uint32_t threads_x, uint32_t group_x);

/* M1B serial uint32 dot: pack header N, then AB pairs; C[0] = sum A[i]*B[i]. */
pai_status_t pai_host_kernel_dot_serial_u32(void *ctx,
                                            const uint32_t user_data[16],
                                            uint32_t threads_x,
                                            uint32_t group_x);

/* M1D serial-per-row GEMV: W header + row-major W, x at header ptr, y[g]. */
pai_status_t pai_host_kernel_gemv_serial_u32(void *ctx,
                                             const uint32_t user_data[16],
                                             uint32_t threads_x,
                                             uint32_t group_x);

/* h1: copy a[i] -> c[4i..4i+3]. */
pai_status_t pai_host_kernel_h1(void *ctx, const uint32_t user_data[16],
                                uint32_t threads_x, uint32_t group_x);

/* h3: c[4i..4i+3] = a[i] + b[i]. */
pai_status_t pai_host_kernel_h3(void *ctx, const uint32_t user_data[16],
                                uint32_t threads_x, uint32_t group_x);

/* loadstore (E2): user_data 0-1 = A, 2-3 = C; c[i] = a[i]. */
pai_status_t pai_host_kernel_loadstore(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

/* T4 elementwise float ops: packed (a,b) at ud[2:3], C at ud[4:5],
 * one group per element (t4_ops.s G42-G46). */
pai_status_t pai_host_kernel_t4_add1d(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_t4_sub1d(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_t4_mul1d(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_t4_relu(void *ctx, const uint32_t user_data[16],
                                     uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_t4_clip(void *ctx, const uint32_t user_data[16],
                                     uint32_t threads_x, uint32_t group_x);

/* T4 biasadd (G47): header [cols, pad, a_lo, a_hi, bias_lo, bias_hi]
 * at ud[2:3], C at ud[4:5], one group per row. */
pai_status_t pai_host_kernel_t4_biasadd(void *ctx, const uint32_t user_data[16],
                                        uint32_t threads_x, uint32_t group_x);

/* T4 matmul (G48): header [K, N, a_lo, a_hi, b_lo, b_hi] at ud[2:3],
 * C at ud[4:5], one group per row. */
pai_status_t pai_host_kernel_t4_matmul(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

/* T4 integer elementwise ops (int_ops.s G49-G53): packed (a,b) at
 * ud[2:3], C at ud[4:5], one group per element, u32 wrap. */
pai_status_t pai_host_kernel_int_add2d(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_int_sub1d(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_int_mul1d(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_int_relu(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);
pai_status_t pai_host_kernel_int_clip(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);

/* T4 integer matmul (G54): header [K, N, a_lo, a_hi, b_lo, b_hi] at
 * ud[2:3], C at ud[4:5], one group per row, u32 wrap. */
pai_status_t pai_host_kernel_int_matmul(void *ctx, const uint32_t user_data[16],
                                        uint32_t threads_x, uint32_t group_x);

/* G55 wave-parallel float ramp: C at ud[2:3], k = float(ud[4]),
 * base = float(ud[5]); lane i stores base + k*(4i+3) for the 8
 * storing lanes (G15 value-path formula, like G35's check). */
pai_status_t pai_host_kernel_ramp(void *ctx, const uint32_t user_data[16],
                                  uint32_t threads_x, uint32_t group_x);

/* G56 wave-parallel float ramp, s_load-fed: header ptr at ud[2:3]
 * (hdr[0]=k, hdr[1]=base as floats), C at ud[4:5]; lane i stores
 * base + k*(4i+3) for the 8 storing lanes (G15 value-path formula). */
pai_status_t pai_host_kernel_ramp2(void *ctx, const uint32_t user_data[16],
                                   uint32_t threads_x, uint32_t group_x);

/* G57 per-lane select: header ptr at ud[2:3] (16 dwords), C at
 * ud[4:5]; lane i stores header[i] for the 8 storing lanes. */
pai_status_t pai_host_kernel_lanepick(void *ctx, const uint32_t user_data[16],
                                      uint32_t threads_x, uint32_t group_x);

/* G58 block dump: header ptr at ud[2:3] (16 dwords), C at ud[4:5];
 * c[0..7] = header[0..7] (s16..s23 copies, no movrels). */
pai_status_t pai_host_kernel_blockdump(void *ctx, const uint32_t user_data[16],
                                       uint32_t threads_x, uint32_t group_x);

/* G59 direct v16 read: header ptr at ud[2:3], C at ud[4:5]; every
 * storing lane writes header[0] (mirrors the real kernel reading v16). */
pai_status_t pai_host_kernel_vpick(void *ctx, const uint32_t user_data[16],
                                   uint32_t threads_x, uint32_t group_x);

/* G60 v8..v15 block: same as vpick (header ptr ud[2:3], C ud[4:5],
 * every storing lane writes header[0]). */
pai_status_t pai_host_kernel_vpick2(void *ctx, const uint32_t user_data[16],
                                    uint32_t threads_x, uint32_t group_x);

/* G64 v_movrels in-ceiling (v7..v14 + m0=7): lane i selects h[7+i]. */
pai_status_t pai_host_kernel_movrels(void *ctx, const uint32_t user_data[16],
                                     uint32_t threads_x, uint32_t group_x);

/* G65/G66 wave-parallel cos/sin ramp (RoPE table primitive): scale at
 * ud[4], C at ud[2:3], c[i] = cos/sin(2*pi*scale*(4i+3)) - turns
 * convention (HW-verified) + value-path (4i+3) quirk. */
pai_status_t pai_host_kernel_cossin_cos(void *ctx,
                                       const uint32_t user_data[16],
                                       uint32_t threads_x,
                                       uint32_t group_x);
pai_status_t pai_host_kernel_cossin_sin(void *ctx,
                                       const uint32_t user_data[16],
                                       uint32_t threads_x,
                                       uint32_t group_x);

/* G67/G68 on-GPU RoPE table generator mirror: header (r2, ctx,
 * scales) at ud[2:3], C at ud[4:5]; cos entry fills cos_t, sin entry
 * the ctx*r2 half. */
pai_status_t pai_host_kernel_ropegen_cos(void *ctx,
                                        const uint32_t user_data[16],
                                        uint32_t threads_x,
                                        uint32_t group_x);
pai_status_t pai_host_kernel_ropegen_sin(void *ctx,
                                        const uint32_t user_data[16],
                                        uint32_t threads_x,
                                        uint32_t group_x);

/* G71/G72 serial v_rsq/v_exp probes (nlexp.s): header (n, pad, x[])
 * at ud[2:3], C at ud[4:5]; c[e] = 1/sqrtf(x[e]) / expf(x[e]). The
 * mirror uses the MATH conventions; the payload locks the 9.40 HW
 * convention empirically (v_exp may be 2^x). */
pai_status_t pai_host_kernel_nlexp_rsq(void *ctx,
                                      const uint32_t user_data[16],
                                      uint32_t threads_x,
                                      uint32_t group_x);
pai_status_t pai_host_kernel_nlexp_exp(void *ctx,
                                      const uint32_t user_data[16],
                                      uint32_t threads_x,
                                      uint32_t group_x);

#endif /* PAI_GPU_HOST_KERNELS_H */
