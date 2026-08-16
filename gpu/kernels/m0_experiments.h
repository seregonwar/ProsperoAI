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

/*
 * FLAT word0 indices used by the ACO bit-15 experiment patch (payload
 * stage E1b/E2b). Each FLAT instruction is two words: word0 carries the
 * 0xDC prefix in the top byte (bit 15 clear in the llvm-mc encoding),
 * word1 carries ADDR/DATA. Patching word1 would corrupt a register
 * field or an immediate — keep these pointed at word0.
 */
#define PAI_STORE_CONST_FLAT_WORD 10u
#define PAI_LOADSTORE_FLAT_LOAD_WORD 10u
#define PAI_LOADSTORE_FLAT_STORE_WORD 17u

extern const uint32_t pai_store_const_code[PAI_STORE_CONST_CODE_WORDS];
extern const uint32_t pai_store_const64_code[PAI_STORE_CONST64_CODE_WORDS];
extern const uint32_t pai_store64_x2_code[PAI_STORE64_X2_CODE_WORDS];
extern const uint32_t pai_store64_x4_code[PAI_STORE64_X4_CODE_WORDS];
extern const uint32_t pai_store64_smem_code[PAI_STORE64_SMEM_CODE_WORDS];
extern const uint32_t pai_store64_gold_code[PAI_STORE64_GOLD_CODE_WORDS];
extern const uint32_t pai_loadstore_gold_code[PAI_LOADSTORE_GOLD_CODE_WORDS];
extern const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS];
extern const uint32_t pai_bisect_code[];

#endif /* PAI_GPU_M0_EXPERIMENT_KERNELS_H */
