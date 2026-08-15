#include "test.h"

#include <pai/tensor.h>

#include <stdlib.h>

TEST_MAIN_BEGIN()

{
  pai_tensor_t t;
  uint64_t shape3[3] = {2, 3, 4};
  uint64_t bytes = 0;
  float buf[24];

  CHECK(pai_tensor_init(&t, PAI_DTYPE_F32, 3, shape3, buf) == PAI_OK);
  CHECK_EQ_UINT(t.rank, 3);
  CHECK_EQ_UINT(t.shape[0], 2);
  CHECK_EQ_UINT(t.shape[1], 3);
  CHECK_EQ_UINT(t.shape[2], 4);
  CHECK_EQ_UINT(pai_tensor_nelem(&t), 24);
  CHECK_EQ_UINT(t.size_bytes, 24 * 4);

  {
    uint64_t idx[3] = {1, 2, 3};
    CHECK_EQ_UINT(pai_tensor_offset(&t, idx), 1 * 12 + 2 * 4 + 3);
  }

  CHECK(pai_tensor_contiguous_size(PAI_DTYPE_F32, 3, shape3, &bytes) ==
        PAI_OK);
  CHECK_EQ_UINT(bytes, 96);

  {
    uint64_t big[1] = {UINT64_MAX / 2};
    CHECK(pai_tensor_contiguous_size(PAI_DTYPE_F32, 1, big, &bytes) ==
          PAI_ERR_INVALID_ARG);
    CHECK_EQ_UINT(bytes, 0);
  }
}

{
  /* degenerate shapes must be rejected */
  pai_tensor_t t;
  uint64_t shape[2] = {4, 0};
  CHECK(pai_tensor_init(&t, PAI_DTYPE_F32, 2, shape, NULL) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_tensor_init(&t, PAI_DTYPE_F32, 0, NULL, NULL) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_tensor_init(&t, (pai_dtype_t)99, 1, shape, NULL) ==
        PAI_ERR_UNSUPPORTED);
}

TEST_MAIN_END()
