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

/* loadstore: 4 user SGPRs (A, C). */
#define PAI_LOADSTORE_RSRC2 0x00000008u
#define PAI_LOADSTORE_CODE_WORDS 20u

extern const uint32_t pai_store_const_code[PAI_STORE_CONST_CODE_WORDS];
extern const uint32_t pai_loadstore_code[PAI_LOADSTORE_CODE_WORDS];

#endif /* PAI_GPU_M0_EXPERIMENT_KERNELS_H */
