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
#define PAI_EXP_RSRC1  0x602C0000u
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
