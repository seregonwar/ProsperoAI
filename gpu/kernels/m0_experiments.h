/*
 * ProsperoAI — GPU kernels (M0 experiments)
 *
 * store_const (E1): 32 threads, dst[tid] = 0xABCD1234, no loads.
 * loadstore   (E2): 32 threads, c[tid] = a[tid] (flat load + store).
 *
 * Minimal hand-assembled kernels whose encodings are llvm-mc verified;
 * used to bisect the compute bring-up on 9.40.
 */

#ifndef PAI_GPU_M0_EXPERIMENT_KERNELS_H
#define PAI_GPU_M0_EXPERIMENT_KERNELS_H

#include <stdint.h>

/* Both kernels use the OpenAGC-proven RSRC1 (WGP_MODE + W32_EN). */
/* RSRC1 base for experiments. VGPRS field (bits 0-5) = 0 by default
 * (8 VGPRs, granularity 8 on gfx10); G57 needs 32 VGPRs for the
 * v[16:31] block copies AND 32 SGPRs for the s_load_dwordx16
 * s[16:31] block (SGPRS field bits 6-11), so it overrides with
 * PAI_EXP_RSRC1_BLOCK32. */
#define PAI_EXP_RSRC1  0x602C0000u
#define PAI_EXP_RSRC1_BLOCK32 0x602C0043u
#define PAI_EXP_RSRC3  0x00000000u
#define PAI_EXP_THREADS_X 32u

/* store_const: 2 user SGPRs (dst). */
#define PAI_STORE_CONST_RSRC2 0x00000004u
#define PAI_STORE_CONST_CODE_WORDS 13u
#define PAI_STORE_CONST_VALUE 0xABCD1234u

/* store_const64: psbc ABI (s0-s1 ring offsets, s2:s3 = dst),
 * 64 threads / group, idempotent constant store. */
#define PAI_STORE_CONST64_RSRC2 0x00000008u
#define PAI_STORE_CONST64_CODE_WORDS 13u
#define PAI_STORE_CONST64_THREADS_X 64u
#define PAI_STORE_CONST64_VALUE 0xCAFEF00Du

/* E12: FLAT_STORE_DWORDX2 variant (2 dwords per thread). */
#define PAI_STORE64_X2_RSRC2 0x00000008u
#define PAI_STORE64_X2_CODE_WORDS 14u
#define PAI_STORE64_X2_VALUE 0xBEADF00Du
#define PAI_STORE64_X2_FLAT_WORD 11u

/* E13: FLAT_STORE_DWORDX4 variant (16 bytes per thread, golden op). */
#define PAI_STORE64_X4_RSRC2 0x00000008u
#define PAI_STORE64_X4_CODE_WORDS 16u
#define PAI_STORE64_X4_VALUE 0xF00DFEEDu
#define PAI_STORE64_X4_FLAT_WORD 13u

/* E22: E19 + SMEM s_load prologue (golden-style first instruction). */
#define PAI_STORE64_SMEM_RSRC2 0x00000008u
#define PAI_STORE64_SMEM_CODE_WORDS 19u
#define PAI_STORE64_SMEM_VALUE 0xC0FFEEEEu
#define PAI_STORE64_SMEM_FLAT_WORD 17u

/* E30: store with the golden register layout (vaddr v[2:3], data
 * v[4:7]) — the only flat combination the 9.40 silicon accepts. */
#define PAI_STORE64_GOLD_RSRC2 0x00000008u
#define PAI_STORE64_GOLD_CODE_WORDS 16u
#define PAI_STORE64_GOLD_VALUE 0xB0DD00D1u

/* E32: v0-broadcast store hypothesis (data lives in v0). */
#define PAI_STORE64_V0_RSRC2 0x00000008u
#define PAI_STORE64_V0_CODE_WORDS 13u
#define PAI_STORE64_V0_VALUE 0x12345678u

/* E34-E37: flat v0-broadcast model kernels (gfx1013/pai_v0model.inc). */
#define PAI_COPY_V0_OFF 0u
#define PAI_COPY_V0_WORDS 20u
#define PAI_COPY_V0_RSRC2 0x0000000Cu
#define PAI_LOAD_V1_OFF 20u
#define PAI_LOAD_V1_WORDS 21u
#define PAI_LOAD_V1_RSRC2 0x0000000Cu
#define PAI_STORE_DW_OFF 41u
#define PAI_STORE_DW_WORDS 13u
#define PAI_STORE_DW_RSRC2 0x00000008u
#define PAI_STORE_DW_VALUE 0xABCDDCBAu
#define PAI_VECADD_V0_OFF 54u
#define PAI_VECADD_V0_WORDS 29u
#define PAI_VECADD_V0_RSRC2 0x00000010u

/* E38: MUBUF per-thread load (T# in s[0:3], C base s4:s5). */
#define PAI_MUBUFLOAD_RSRC2 0x0000000Cu
#define PAI_MUBUFLOAD_CODE_WORDS 14u

/* E39: arithmetic milestone kernel — c[i] = (float)i + k (s4). */
#define PAI_ARITH_RSRC2 0x0000000Cu
#define PAI_ARITH_CODE_WORDS 13u

/* E45/E46: arith under RSRC2 0x08 (k at s0, C at s2:s3). */
#define PAI_ARITH4_RSRC2 0x00000008u
#define PAI_ARITH4_OFF 0u
#define PAI_ARITH4_WORDS 13u
#define PAI_ARITH4B_OFF 13u
#define PAI_ARITH4B_WORDS 12u

/* F1-F4: dst-v0-broadcast workaround batch (gfx1013/pai_fbatch.inc). */
#define PAI_F1_OFF 0u
#define PAI_F1_WORDS 14u
#define PAI_F2_OFF 14u
#define PAI_F2_WORDS 13u
#define PAI_F3_OFF 27u
#define PAI_F3_WORDS 15u
#define PAI_F4_OFF 41u
#define PAI_F4_WORDS 16u

/* F5: shotgun — value in v0, v4 and v5 at the same time. */
#define PAI_SHOTGUN_RSRC2 0x00000008u
#define PAI_SHOTGUN_CODE_WORDS 14u
#define PAI_SHOTGUN_VALUE 0x11112222u

/* F6 (vecscalar): THE milestone kernel — c[i] = (float)i + k (s4). */
#define PAI_VECSCALAR_RSRC2 0x0000000Cu
#define PAI_VECSCALAR_CODE_WORDS 14u
#define PAI_VECSCALAR_K 1.5f

/* G7/G8: x4 broadcast + per-thread integer arithmetic (pai_gbatch.inc). */
#define PAI_G7_OFF 0u
#define PAI_G7_WORDS 11u
#define PAI_G8_OFF 11u
#define PAI_G8_WORDS 15u
#define PAI_G8_RSRC2 0x0000000Cu

/* G9-G12: store data source probes (pai_gbatch2.inc). */
#define PAI_G9_OFF 0u
#define PAI_G9_WORDS 13u
#define PAI_G10_OFF 13u
#define PAI_G10_WORDS 16u
#define PAI_G11_OFF 29u
#define PAI_G11_WORDS 16u
#define PAI_G12_OFF 45u
#define PAI_G12_WORDS 13u

/* G13/G14: literal-source probes (pai_gbatch3.inc). */
#define PAI_G13_OFF 0u
#define PAI_G13_WORDS 14u
#define PAI_G14_OFF 14u
#define PAI_G14_WORDS 13u
#define PAI_G14_VALUE 0xDEADBEEFu

/* G15: THE milestone — c[i] = i + k_int (pai_g15.inc). */
#define PAI_G15_RSRC2 0x0000000Cu
#define PAI_G15_CODE_WORDS 14u

/* G16: self-reference - store 0x12345678, load back, store the loaded
 * value at +32 words (pai_selfref.inc). */
#define PAI_G16_RSRC2 0x00000008u
#define PAI_G16_CODE_WORDS 25u

/* G17: acqrb-VA load probe (pai_acqload.inc). */
#define PAI_G17_RSRC2 0x0000000Cu
#define PAI_G17_CODE_WORDS 17u

/* G18: G16 without any s_waitcnt (pai_selfref_nowait.inc). */
#define PAI_G18_RSRC2 0x00000008u
#define PAI_G18_CODE_WORDS 23u

/* G19: SMEM scalar load probe (pai_smemload.inc). */
#define PAI_G19_RSRC2 0x00000008u
#define PAI_G19_CODE_WORDS 20u

/* G20: clean MUBUF load test (pai_mubufload_clean.inc). */
#define PAI_G20_RSRC2 0x0000000Cu
#define PAI_G20_CODE_WORDS 21u
#define PAI_G20_VALUE 0xA5A5A5A5u
#define PAI_G20_TBUF_WORD3 0x31014FACu

/* G21: MUBUF load with the G15 store formula (pai_mubufload_g15.inc). */
#define PAI_G21_RSRC2 0x0000000Cu
#define PAI_G21_CODE_WORDS 19u

/* G22: SMEM load with the G15 store formula (pai_smemload_g15.inc). */
#define PAI_G22_RSRC2 0x0000000Cu
#define PAI_G22_CODE_WORDS 16u

/* M0-G / G23: parametric 1D add C[i]=A[i]+B[i] (pai_add1d.inc).
 * RSRC2 = 6 user SGPRs + TGID_X_EN; NUM_THREAD_X = 1. */
#define PAI_ADD1D_RSRC2 0x0000008Cu
#define PAI_ADD1D_CODE_WORDS 25u
#define PAI_ADD1D_THREADS 1u
#define PAI_ADD1D_ITERS 1000u
#define PAI_G23_RSRC2 PAI_ADD1D_RSRC2
#define PAI_G23_CODE_WORDS PAI_ADD1D_CODE_WORDS

/* Integer SAXPY: C[i] = a * A[i] + B[i], a = 3 immediate (s_mulk_i32).
 * Same ABI as add1d (RSRC2 0x8C, 1 thread/group, groups_x = N). */
#define PAI_SAXPY_RSRC2 PAI_ADD1D_RSRC2
#define PAI_SAXPY_CODE_WORDS 26u
#define PAI_SAXPY_THREADS PAI_ADD1D_THREADS
#define PAI_SAXPY_ITERS PAI_ADD1D_ITERS
#define PAI_SAXPY_A 3u

/* M1B: serial uint32 dot (correctness primitive, not a fast reduction).
 * One group, NUM_THREAD_X=1, SGPR loop of s_load + mul + add. */
#define PAI_DOT_SERIAL_U32_RSRC2 PAI_ADD1D_RSRC2
#define PAI_DOT_SERIAL_U32_CODE_WORDS 33u
#define PAI_DOT_SERIAL_U32_THREADS 1u
#define PAI_DOT_SERIAL_U32_ITERS 1000u

/* M1D: serial-per-row uint32 GEMV (correctness, not a fast GEMV).
 * groups_x=M, NUM_THREAD_X=1; each group is dot_serial over one row. */
#define PAI_GEMV_SERIAL_U32_RSRC2 PAI_ADD1D_RSRC2
#define PAI_GEMV_SERIAL_U32_CODE_WORDS 50u
#define PAI_GEMV_SERIAL_U32_THREADS 1u
#define PAI_GEMV_SERIAL_U32_ITERS 1000u
#define PAI_GEMV_SERIAL_U32_HDR_DWORDS 4u

/* G24: s_load_dwordx16 alone (G23 hang bisection, pai_smemload16.inc). */
#define PAI_G24_RSRC2 0x0000000Cu
#define PAI_G24_CODE_WORDS 16u

/* G25: minimal LDS roundtrip probe (pai_dsprobe.inc).
 * Legacy 0x4C = USER_SGPR=6 | TRAP_PRESENT, LDS_SIZE=0.
 * OpenAGC gfx1013: LDS allocated in 1 KiB blocks, field still in
 * 512-byte granules → minimum legal LDS_SIZE is 2 (1 KiB). */
#define PAI_G25_RSRC2_LEGACY 0x0000004Cu
#define PAI_G25_RSRC2_512B 0x0000800Cu /* illegal odd granule on gfx1013 */
#define PAI_G25_RSRC2_1KB 0x0001000Cu  /* LDS_SIZE=2 → 1024 B */
#define PAI_G25_RSRC2_8KB 0x0008000Cu
#define PAI_G25_RSRC2 PAI_G25_RSRC2_1KB
#define PAI_G25_CODE_WORDS 21u
#define PAI_G25_THREADS 1u

/* M1C: per-lane LDS roundtrip, 8 threads, no barrier (pai_lds_lanes.inc). */
#define PAI_LDS_LANES_RSRC2 PAI_G25_RSRC2_8KB
#define PAI_LDS_LANES_CODE_WORDS 21u
#define PAI_LDS_LANES_THREADS 8u
#define PAI_LDS_LANES_BASE 0xA5000000u

/* G26: one x16 load + one ds write/read + G15 formula (pai_dsstaged.inc).
 * RSRC2 = 6 user SGPRs (s0-s5) + the G25 LDS bit. */
#define PAI_G26_RSRC2 0x0000004Cu
#define PAI_G26_CODE_WORDS 26u

/* G27: single s_load + ds write/read + G15 formula (pai_dsstaged1.inc).
 * RSRC2 = 6 user SGPRs (s0-s5) + the G25 LDS bit. */
#define PAI_G27_RSRC2 0x0000004Cu
#define PAI_G27_CODE_WORDS 26u

/* G28-G30: float ALU v2 - cvt then add (pai_fbatch2.inc). */
#define PAI_G28_RSRC2 0x0000000Cu
#define PAI_G28_OFF 0u
#define PAI_G28_WORDS 13u
#define PAI_G29_OFF 13u
#define PAI_G29_WORDS 14u
#define PAI_G30_OFF 27u
#define PAI_G30_WORDS 13u
#define PAI_G2X_K 0.5f

/* G31/G32: float add operand probes (pai_fbatch3.inc). */
#define PAI_G31_OFF 0u
#define PAI_G31_WORDS 14u
#define PAI_G32_OFF 14u

/* G33: the unlocked float form - SGPR scalar via a VGPR (pai_fbatch4.inc). */
#define PAI_G33_RSRC2 0x0000000Cu
#define PAI_G33_CODE_WORDS 16u

/* G34/G35: SGPR scalar into the float add (pai_fbatch5.inc). */
#define PAI_G34_OFF 0u
#define PAI_G34_WORDS 17u
#define PAI_G35_OFF 17u

/* G36: MUBUF with the T# in non-zeroed s[4:7] (pai_mubufload36.inc). */
#define PAI_G36_RSRC2 0x00000010u
#define PAI_G36_CODE_WORDS 19u

/* G37/G38: MUBUF format matrix (pai_mubufload37.inc). */
#define PAI_G37_OFF 0u
#define PAI_G37_WORDS 19u
#define PAI_G38_OFF 19u
#define PAI_G38_WORDS 19u
#define PAI_G37_TBUF_WORD2 0x20002000u
#define PAI_G37_TBUF_WORD3 0x31044FACu
#define PAI_G38_TBUF_WORD3 0x31040080u

/* G39: VALU float dot, serial SMEM path (pai_fdot_serial.inc). */
#define PAI_FDOT_RSRC2 0x0000008Cu
#define PAI_FDOT_THREADS 1u
#define PAI_FDOT_CODE_WORDS 33u
#define PAI_FDOT_ITERS 10u

/* G40: VALU float GEMV, serial-per-row (pai_fgemv_serial.inc). */
#define PAI_FGEMV_RSRC2 0x0000008Cu
#define PAI_FGEMV_THREADS 1u
#define PAI_FGEMV_CODE_WORDS 50u

/* G41: VALU float SAXPY, per-group (pai_fsaxpy.inc). */
#define PAI_FSAXPY_RSRC2 0x0000008Cu
#define PAI_FSAXPY_THREADS 1u
#define PAI_FSAXPY_CODE_WORDS 26u
#define PAI_FSAXPY_K 0.5f

/* G42-G48: T4 serial float op kernels (pai_t4_ops.inc).
 * Elementwise (G42-G46): packed (a,b) pairs, C[g] per group.
 * Biasadd (G47): header [cols, pad, a_lo, a_hi, bias_lo, bias_hi],
 *   C[g*cols+j] = a[g*cols+j] + bias[j], rows = groups.
 * Matmul (G48): header [K, N, a_lo, a_hi, b_lo, b_hi],
 *   C[i*N+j] = sum_k a[i*K+k] * b[k*N+j], rows = groups. */
#define PAI_T4_RSRC2 PAI_ADD1D_RSRC2
#define PAI_T4_THREADS 1u
#define PAI_T4_ADD1D_OFF 0u
#define PAI_T4_ADD1D_WORDS 24u
#define PAI_T4_SUB1D_OFF 24u
#define PAI_T4_SUB1D_WORDS 24u
#define PAI_T4_MUL1D_OFF 48u
#define PAI_T4_MUL1D_WORDS 24u
#define PAI_T4_RELU_OFF 72u
#define PAI_T4_RELU_WORDS 24u
#define PAI_T4_CLIP_OFF 96u
#define PAI_T4_CLIP_WORDS 26u
#define PAI_T4_BIASADD_OFF 122u
#define PAI_T4_BIASADD_WORDS 45u
#define PAI_T4_MATMUL_OFF 167u
#define PAI_T4_MATMUL_WORDS 63u
#define PAI_T4_CODE_WORDS 230u

/* G49-G54: T4 serial integer op kernels (int_ops.s / pai_int_ops.inc).
 * Same ABI as the float T4 ops: packed (a,b) pairs, C[g] per group.
 * Elementwise (G49-G53): add2d/sub1d/mul1d/relu/clip, u32 wrap.
 * Matmul (G54): header [K, N, a_lo, a_hi, b_lo, b_hi],
 *   C[i*N+j] = sum_k a[i*K+k] * b[k*N+j] (u32 wrap), rows = groups.
 * NOTE: b k-stride is N*4 bytes (row-major [k][N]); G48 float matmul
 * advanced b by N bytes - flagged for T4C. */
#define PAI_INT_RSRC2 PAI_ADD1D_RSRC2
#define PAI_INT_THREADS 1u
#define PAI_INT_ADD2D_OFF 0u
#define PAI_INT_ADD2D_WORDS 23u
#define PAI_INT_SUB1D_OFF 23u
#define PAI_INT_SUB1D_WORDS 23u
#define PAI_INT_MUL1D_OFF 46u
#define PAI_INT_MUL1D_WORDS 23u
#define PAI_INT_RELU_OFF 69u
#define PAI_INT_RELU_WORDS 23u
#define PAI_INT_CLIP_OFF 92u
#define PAI_INT_CLIP_WORDS 24u
#define PAI_INT_MATMUL_OFF 116u
#define PAI_INT_MATMUL_WORDS 62u
#define PAI_INT_CODE_WORDS 178u

/* G55: wave-parallel float linear ramp (ramp.s) - c[i] = base + k*i,
 * NUM_THREAD_X=32, 1 group, lanes 0-7 store. RSRC2 0x0C (G33/G35
 * ABI): s2:s3 = C, s4 = k, s5 = base. First kernel past the serial
 * NUM_THREAD_X=1 model; base for RoPE position tables (Phase 2). */
#define PAI_RAMP_RSRC2 0x0000000Cu
#define PAI_RAMP_CODE_WORDS 17u

/* G56: wave-parallel float ramp, s_load-fed (ramp2.s) - k/base read
 * from a GPU-mem header via s_load_dword inside the 32-thread wave;
 * proves the scalar-read path works wave-parallel (x-side of GEMV). */
#define PAI_RAMP2_RSRC2 PAI_RAMP_RSRC2
#define PAI_RAMP2_CODE_WORDS 22u

/* G57: per-lane select from an s_load_dwordx16 block (lanepick.s) via
 * v_movrels_b32 (m0 base + v0 index). Unlock probe for wave-parallel
 * GEMV: gives each lane its own element without vector loads. */
#define PAI_LANEPICK_RSRC2 PAI_RAMP_RSRC2
#define PAI_LANEPICK_CODE_WORDS 33u

/* G58: s_load_dwordx16 block dump (blockdump.s) - bisection probe for
 * G57: does the 16-dword block land in s[16:31] when 32 SGPRs are
 * allocated? c[0..7] = s16..s23 via direct v_mov copies, no movrels. */
#define PAI_BLOCKDUMP_RSRC2 PAI_RAMP_RSRC2
#define PAI_BLOCKDUMP_CODE_WORDS 59u

/* G59: direct v16 read (vpick.s) - decisive bisection for G57: if
 * c[0..7] == header[0], v16+ copies land and v_movrels is broken; if
 * garbage, the v16+ copies are dropped (RSRC1 VGPRS wrong). */
#define PAI_VPICK_RSRC2 PAI_RAMP_RSRC2
#define PAI_VPICK_CODE_WORDS 24u

/* G60: v8..v15 block copies under STANDARD RSRC1 (vpick2.s) - golden
 * E22 proves v6-v9 work with 0x602C0000; does v15 also? Separates a
 * real VGPR ceiling from the BLOCK32 RSRC1 change. */
#define PAI_VPICK2_RSRC2 PAI_RAMP_RSRC2
#define PAI_VPICK2_CODE_WORDS 24u

/* G64: v_movrels_b32 with in-ceiling block v7..v14 + m0=7 (movrels.s)
 * - the one untested piece: G57's v16+ copies were dropped by the
 * 16-VGPR hardware ceiling (G59/G63), so movrels read garbage there.
 * c[i] = h[7+i] = 0x10000007+i for lanes 0..7. */
#define PAI_MOVRELS_RSRC2 PAI_RAMP_RSRC2
#define PAI_MOVRELS_CODE_WORDS 25u

/* G65/G66: wave-parallel cos/sin ramp (cossin.s) - RoPE position-table
 * primitives. theta = scale*i read through the value-path quirk as
 * (4i+3): c[i] = cosf(scale*(4i+3)) / sinf(scale*(4i+3)). Same ABI as
 * G55 (s2:s3 = C, s4 = scale float); cos entry @0, sin entry @16. */
#define PAI_COSSIN_RSRC2 PAI_RAMP_RSRC2
#define PAI_COSSIN_COS_OFF 0u
#define PAI_COSSIN_COS_WORDS 16u
#define PAI_COSSIN_SIN_OFF 16u
#define PAI_COSSIN_SIN_WORDS 16u
#define PAI_COSSIN_CODE_WORDS 32u

/* G67/G68: on-GPU RoPE cos/sin table generator (ropegen.s) - serial
 * per element (T4/G40 model): groups_x = ctx*r2, TGID_X = e = p*r2+i,
 * all addressing in SGPRs, value = two-SGPR v_mul + v_cos/v_sin
 * (turns convention). Host precomputes theta_turns[e] = theta/(2pi)
 * so x2pi cancels: c[e] = cos/sin(theta) exactly, all rows 0..ctx-1.
 * Header h[0]=r2, h[1]=ctx, h[2+e]=theta_turns; C = cos then sin. */
#define PAI_ROPEGEN_RSRC2 0x0000008Cu
#define PAI_ROPEGEN_THREADS 1u
#define PAI_ROPEGEN_COS_OFF 0u
#define PAI_ROPEGEN_COS_WORDS 26u
#define PAI_ROPEGEN_SIN_OFF 26u
#define PAI_ROPEGEN_SIN_WORDS 34u
#define PAI_ROPEGEN_CODE_WORDS 60u
#define PAI_ROPEGEN_SIN_VSIN_WORD 17u /* v_sin_f32 in sin entry
                                         (abs word 43, sin off 26;
                                         7E026B01; v_cos=7E026D01) */
#define PAI_G35_WORDS 15u
#define PAI_G32_WORDS 16u
#define PAI_G25_VALUE 0xDEAD0001u
#define PAI_G19_VALUE 0xA5A5A5A5u
#define PAI_G16_VALUE 0x12345678u

/* H1/H3: MUBUF loads with the OpenAGC raw T# (pai_hbatch.inc). */
#define PAI_H1_OFF 0u
#define PAI_H1_WORDS 15u
#define PAI_H1_RSRC2 0x0000000Cu
#define PAI_H3_OFF 15u
#define PAI_H3_WORDS 19u
#define PAI_H3_RSRC2 0x00000014u
#define PAI_TBUF_WORD3_RAW 0x31014FACu

/* H7-H9: load result register probes (pai_hbatch2.inc). */
#define PAI_H7_OFF 0u
#define PAI_H7_WORDS 16u
#define PAI_H8_OFF 16u
#define PAI_H8_WORDS 16u
#define PAI_H9_OFF 32u
#define PAI_H9_WORDS 16u
#define PAI_TBUF_WORD3_EXEC 0x00080688u

/* H10: SMEM scalar load feeding the store (pai_hbatch3.inc). */
#define PAI_H10_RSRC2 0x0000000Cu
#define PAI_H10_CODE_WORDS 16u

/* H11: SMEM load with the proven sbase pair s[2:3], in-place on A. */
#define PAI_H11_RSRC2 0x00000008u
#define PAI_H11_CODE_WORDS 16u

/* H12/H13: zeroed-s0-s1 workaround (pai_hbatch5.inc). */
#define PAI_H12_OFF 0u
#define PAI_H12_WORDS 16u
#define PAI_H12_RSRC2 0x0000000Cu
#define PAI_H13_OFF 16u
#define PAI_H13_WORDS 15u
#define PAI_H13_RSRC2 0x00000010u

/* E31: loadstore with vaddr pairs 1 (v[2:3]) only. */
#define PAI_LOADSTORE_GOLD_RSRC2 0x0000000Cu
#define PAI_LOADSTORE_GOLD_CODE_WORDS 20u

/* Instruction bisection (E15-E19): word ranges in gfx1013/pai_bisect.inc. */
#define PAI_BISECT_BARE_OFF 0u
#define PAI_BISECT_BARE_WORDS 2u
#define PAI_BISECT_LSHL_OFF 2u
#define PAI_BISECT_LSHL_WORDS 3u
#define PAI_BISECT_ADDCO_OFF 5u
#define PAI_BISECT_ADDCO_WORDS 5u
#define PAI_BISECT_ADDCI_OFF 10u
#define PAI_BISECT_ADDCI_WORDS 7u
#define PAI_BISECT_STORE_OFF 17u
#define PAI_BISECT_STORE_WORDS 14u
#define PAI_BISECT_STORE_VALUE 0xDEADBEEFu
#define PAI_BISECT_STORE_FLAT_WORD 28u /* absolute index into pai_bisect_code */

/* loadstore: 4 user SGPRs (A, C). */
#define PAI_LOADSTORE_RSRC2 0x00000008u
#define PAI_LOADSTORE_CODE_WORDS 20u

/* FLAT word0 indices for the ACO bit-15 patch (payload E1b/E2b). Each
 * FLAT instruction is two words: word0 has the DC prefix (bit 15 clear
 * in the llvm-mc encoding), word1 holds ADDR/DATA — never patch word1. */
#define PAI_STORE_CONST_FLAT_WORD 10u
#define PAI_LOADSTORE_FLAT_LOAD_WORD 10u
#define PAI_LOADSTORE_FLAT_STORE_WORD 17u

extern const uint32_t pai_store_const_code[PAI_STORE_CONST_CODE_WORDS];
extern const uint32_t pai_store_const64_code[PAI_STORE_CONST64_CODE_WORDS];
extern const uint32_t pai_store64_x2_code[PAI_STORE64_X2_CODE_WORDS];
extern const uint32_t pai_store64_x4_code[PAI_STORE64_X4_CODE_WORDS];
extern const uint32_t pai_store64_smem_code[PAI_STORE64_SMEM_CODE_WORDS];
extern const uint32_t pai_store64_gold_code[PAI_STORE64_GOLD_CODE_WORDS];
extern const uint32_t pai_store64_v0_code[PAI_STORE64_V0_CODE_WORDS];
extern const uint32_t pai_mubufload_code[PAI_MUBUFLOAD_CODE_WORDS];
extern const uint32_t pai_arith_code[PAI_ARITH_CODE_WORDS];
extern const uint32_t pai_arith4_code[];
extern const uint32_t pai_fbatch_code[];
extern const uint32_t pai_shotgun_code[PAI_SHOTGUN_CODE_WORDS];
extern const uint32_t pai_vecscalar_code[PAI_VECSCALAR_CODE_WORDS];
extern const uint32_t pai_gbatch_code[];
extern const uint32_t pai_gbatch2_code[];
extern const uint32_t pai_gbatch3_code[];
extern const uint32_t pai_g15_code[PAI_G15_CODE_WORDS];
extern const uint32_t pai_selfref_code[PAI_G16_CODE_WORDS];
extern const uint32_t pai_acqload_code[PAI_G17_CODE_WORDS];
extern const uint32_t pai_selfref_nowait_code[PAI_G18_CODE_WORDS];
extern const uint32_t pai_smemload_code[PAI_G19_CODE_WORDS];
extern const uint32_t pai_mubufload_clean_code[PAI_G20_CODE_WORDS];
extern const uint32_t pai_mubufload_g15_code[PAI_G21_CODE_WORDS];
extern const uint32_t pai_smemload_g15_code[PAI_G22_CODE_WORDS];
extern const uint32_t pai_smemvecadd_code[PAI_G23_CODE_WORDS];
extern const uint32_t pai_saxpy_code[PAI_SAXPY_CODE_WORDS];
extern const uint32_t pai_dot_serial_u32_code[PAI_DOT_SERIAL_U32_CODE_WORDS];
extern const uint32_t pai_gemv_serial_u32_code[PAI_GEMV_SERIAL_U32_CODE_WORDS];
extern const uint32_t pai_smemload16_code[PAI_G24_CODE_WORDS];
extern const uint32_t pai_dsprobe_code[PAI_G25_CODE_WORDS];
extern const uint32_t pai_lds_lanes_code[PAI_LDS_LANES_CODE_WORDS];
extern const uint32_t pai_dsstaged_code[PAI_G26_CODE_WORDS];
extern const uint32_t pai_dsstaged1_code[PAI_G27_CODE_WORDS];
extern const uint32_t pai_fbatch2_code[];
extern const uint32_t pai_fbatch3_code[];
extern const uint32_t pai_fbatch4_code[PAI_G33_CODE_WORDS];
extern const uint32_t pai_fbatch5_code[];
extern const uint32_t pai_mubufload36_code[PAI_G36_CODE_WORDS];
extern const uint32_t pai_mubufload37_code[];
extern const uint32_t pai_fdot_serial_code[PAI_FDOT_CODE_WORDS];
extern const uint32_t pai_fgemv_serial_code[PAI_FGEMV_CODE_WORDS];
extern const uint32_t pai_fsaxpy_code[PAI_FSAXPY_CODE_WORDS];
extern const uint32_t pai_t4_ops_code[PAI_T4_CODE_WORDS];
extern const uint32_t pai_int_ops_code[PAI_INT_CODE_WORDS];
extern const uint32_t pai_ramp_code[PAI_RAMP_CODE_WORDS];
extern const uint32_t pai_ramp2_code[PAI_RAMP2_CODE_WORDS];
extern const uint32_t pai_lanepick_code[PAI_LANEPICK_CODE_WORDS];
extern const uint32_t pai_blockdump_code[PAI_BLOCKDUMP_CODE_WORDS];
extern const uint32_t pai_vpick_code[PAI_VPICK_CODE_WORDS];
extern const uint32_t pai_vpick2_code[PAI_VPICK2_CODE_WORDS];
extern const uint32_t pai_movrels_code[PAI_MOVRELS_CODE_WORDS];
extern const uint32_t pai_cossin_code[PAI_COSSIN_CODE_WORDS];
extern const uint32_t pai_ropegen_code[PAI_ROPEGEN_CODE_WORDS];
extern const uint32_t pai_hbatch_code[];
extern const uint32_t pai_hbatch2_code[];
extern const uint32_t pai_hbatch3_code[PAI_H10_CODE_WORDS];
extern const uint32_t pai_hbatch4_code[PAI_H11_CODE_WORDS];
extern const uint32_t pai_hbatch5_code[];
extern const uint32_t pai_v0model_code[];
extern const uint32_t pai_loadstore_gold_code[PAI_LOADSTORE_GOLD_CODE_WORDS];
extern const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS];
extern const uint32_t pai_bisect_code[];

#endif /* PAI_GPU_M0_EXPERIMENT_KERNELS_H */
