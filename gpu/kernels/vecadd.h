/*
 * ProsperoAI — GPU kernels
 *
 * vecadd: c[i] = a[i] + b[i] (f32), single group, 32 threads, W32 mode.
 *
 * Kernel configuration constants and the assembled code blob.
 * The code array is generated from gfx1013/vecadd.s by
 * toolchain/assemble_shader.py (or the CMake assembly rule) and lives in
 * gfx1013/pai_vecadd_code.inc.
 */

#ifndef PAI_GPU_VECADD_KERNEL_H
#define PAI_GPU_VECADD_KERNEL_H

#include <stdint.h>

/* COMPUTE_PGM_RSRC1: WGP_MODE + W32_EN + float-mode defaults, identical
 * to the hardware-qualified OpenAGC memset kernel. */
#define PAI_VECADD_RSRC1 0x602C0000u

/* COMPUTE_PGM_RSRC2: 7 user SGPRs (A, B, C pointers + n). */
#define PAI_VECADD_RSRC2 0x0000000Eu

#define PAI_VECADD_RSRC3 0x00000000u

#define PAI_VECADD_THREADS_X 32u
#define PAI_VECADD_MAX_ELEMS 32u

/* USER_DATA register indices. */
#define PAI_VECADD_UD_A_LO 0u
#define PAI_VECADD_UD_A_HI 1u
#define PAI_VECADD_UD_B_LO 2u
#define PAI_VECADD_UD_B_HI 3u
#define PAI_VECADD_UD_C_LO 4u
#define PAI_VECADD_UD_C_HI 5u
#define PAI_VECADD_UD_N 6u

#define PAI_VECADD_CODE_WORDS 29u

extern const uint32_t pai_vecadd_code[PAI_VECADD_CODE_WORDS];

#endif /* PAI_GPU_VECADD_KERNEL_H */
