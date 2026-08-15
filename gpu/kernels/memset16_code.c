/*
 * OpenAGC memset-exclusive gfx1013 kernel code (Apache-2.0).
 * Source: OpenAGC src/memset_exclusive_shader.h (sha 87f111e6...),
 * assembled by openagc-psbc from shaders/memset_exclusive.comp,
 * hardware-qualified through the AGC driver on FW 5.50.
 * Keep in sync with gpu/kernels/memset16.h.
 */

#include "memset16.h"

const uint32_t pai_memset16_code[PAI_MEMSET16_CODE_WORDS] = {
    0xD7460000u, 0x04010C09u, 0x7DA80004u, 0xBF88000Cu,
    0x3002009Fu, 0x7E080205u, 0xD7000005u, 0x02000C80u,
    0x7E0E0208u, 0xD6FF0000u, 0x02020084u, 0xD70F6A02u,
    0x02020002u, 0x50060203u, 0xDC788000u, 0x007D0402u,
    0xBF810000u, 0xBF9F0000u, 0xBF9F0000u, 0xBF9F0000u,
    0xBF9F0000u, 0xBF9F0000u,
};
