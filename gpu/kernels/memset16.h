/*
 * ProsperoAI — GPU kernels
 *
 * memset16: fills 16-byte blocks with a repeated pattern.
 *
 * Golden bring-up reference: byte-for-byte the hardware-qualified
 * OpenAGC memset-exclusive gfx1013 kernel (Apache-2.0, proven through
 * the AGC driver on FW 5.50). Used to validate the compute dispatch
 * path independently from our own kernels.
 *
 * User data ABI (COMPUTE_USER_DATA_0..8):
 *   0-1   = unused ring offsets (0)
 *   2-3   = destination GPU VA
 *   4     = number of 16-byte blocks
 *   5-8   = 16-byte pattern
 * Launch: NUM_THREAD_X = 64, group_count = ceil(blocks / 64).
 */

#ifndef PAI_GPU_MEMSET16_KERNEL_H
#define PAI_GPU_MEMSET16_KERNEL_H

#include <stdint.h>

#define PAI_MEMSET16_RSRC1      0x602C0000u
#define PAI_MEMSET16_RSRC2      0x00000092u /* 9 user SGPRs */
#define PAI_MEMSET16_RSRC3      0x00000000u
#define PAI_MEMSET16_THREADS_X  64u
#define PAI_MEMSET16_BLOCK_BYTES 16u

/* USER_DATA register indices. */
#define PAI_MEMSET16_UD_DST_LO   2u
#define PAI_MEMSET16_UD_DST_HI   3u
#define PAI_MEMSET16_UD_BLOCKS   4u
#define PAI_MEMSET16_UD_PATTERN0 5u

#define PAI_MEMSET16_CODE_WORDS 22u

extern const uint32_t pai_memset16_code[PAI_MEMSET16_CODE_WORDS];

#endif /* PAI_GPU_MEMSET16_KERNEL_H */
