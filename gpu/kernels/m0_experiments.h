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
extern const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS];

#endif /* PAI_GPU_M0_EXPERIMENT_KERNELS_H */
