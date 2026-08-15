#include "test.h"

#include <ref_ops.h>

#include <math.h>
#include <stdlib.h>

TEST_MAIN_BEGIN()

{
  /* vecadd vs manual computation */
  float a[64], b[64], c[64];
  for (int i = 0; i < 64; i++) {
    a[i] = (float)i * 0.5f;
    b[i] = (float)(i % 7) - 3.0f;
  }
  CHECK(pai_ref_vecadd_f32(a, b, c, 64) == PAI_OK);
  for (int i = 0; i < 64; i++) {
    if (fabsf(c[i] - (a[i] + b[i])) > 1e-6f) {
      CHECK(0);
      break;
    }
  }
}

{
  /* gemm 4x3x2 vs manual */
  float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float b[6] = {1, 0, 2, 0, 1, 2}; /* 2x3 row-major */
  float c[12];
  CHECK(pai_ref_gemm_f32(4, 3, 2, a, b, c) == PAI_OK);

  /* row0 = a[0..1] x b */
  CHECK(fabsf(c[0] - (1 * 1 + 2 * 0)) < 1e-6f);
  CHECK(fabsf(c[1] - (1 * 0 + 2 * 1)) < 1e-6f);
  CHECK(fabsf(c[2] - (1 * 2 + 2 * 2)) < 1e-6f);
  /* row3 = a[6..7] x b */
  CHECK(fabsf(c[9] - (7 * 1 + 8 * 0)) < 1e-6f);
  CHECK(fabsf(c[10] - (7 * 0 + 8 * 1)) < 1e-6f);
  CHECK(fabsf(c[11] - (7 * 2 + 8 * 2)) < 1e-6f);
}

{
  /* compare detects mismatches and reports the first index */
  float a[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  float b[8] = {1, 1, 1, 9, 1, 1, 1, 1};
  uint64_t first = 0;
  CHECK(pai_ref_compare_f32(a, b, 8, 1e-6f, 1e-6f, &first) ==
        PAI_ERR_MISMATCH);
  CHECK_EQ_UINT(first, 3);
  CHECK(pai_ref_compare_f32(a, a, 8, 1e-6f, 1e-6f, &first) == PAI_OK);
}

{
  /* memset16 pattern fill */
  uint32_t out[16];
  uint32_t pat[4] = {0xDEADBEEF, 0x11223344, 0x55667788, 0x99AABBCC};
  CHECK(pai_ref_memset16(out, pat, 4) == PAI_OK);
  for (int i = 0; i < 16; i++) {
    CHECK_EQ_UINT(out[i], pat[i % 4]);
  }
}

TEST_MAIN_END()
