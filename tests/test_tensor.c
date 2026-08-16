#include "test.h"

#include <pai/quant.h>
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

{
  /* accumulator dtype rules */
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_U8), PAI_DTYPE_I32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_I8), PAI_DTYPE_I32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_U16), PAI_DTYPE_I32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_I16), PAI_DTYPE_I32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_F16), PAI_DTYPE_F32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_BF16), PAI_DTYPE_F32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_F32), PAI_DTYPE_F32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_I32), PAI_DTYPE_I32);
  CHECK_EQ_INT(pai_dtype_accumulator(PAI_DTYPE_U32), PAI_DTYPE_U32);
  CHECK_EQ_INT(pai_dtype_accumulator((pai_dtype_t)99), PAI_DTYPE_COUNT);
}

{
  /* quant NONE: identity dequant, f32 defaults */
  pai_quant_t q;
  pai_quant_init(&q);
  CHECK_EQ_INT(q.scheme, PAI_QUANT_NONE);
  CHECK_EQ_INT(q.storage, PAI_DTYPE_F32);
  CHECK_EQ_INT(pai_quant_accumulator(&q), PAI_DTYPE_F32);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 0, 42), 42.0f);
}

{
  /* quant PER_TENSOR: u8, scale 0.5, zero point 128 */
  pai_quant_t q;
  CHECK(pai_quant_init_per_tensor(&q, PAI_DTYPE_U8, 0.5f, 128.0f) ==
        PAI_OK);
  CHECK_EQ_INT(q.scheme, PAI_QUANT_PER_TENSOR);
  CHECK_EQ_INT(pai_quant_accumulator(&q), PAI_DTYPE_I32);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 0, 100), -14.0f);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 0, 128), 0.0f);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 0, 130), 1.0f);

  /* invalid: float storage, zero scale, null quant */
  CHECK(pai_quant_init_per_tensor(&q, PAI_DTYPE_F32, 0.5f, 0.0f) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_quant_init_per_tensor(&q, PAI_DTYPE_U8, 0.0f, 0.0f) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_quant_init_per_tensor(NULL, PAI_DTYPE_U8, 0.5f, 0.0f) ==
        PAI_ERR_INVALID_ARG);
}

{
  /* quant PER_CHANNEL: 2 channels */
  pai_quant_t q;
  float scales[2] = {0.5f, 2.0f};
  float zps[2] = {0.0f, 1.0f};
  CHECK(pai_quant_init_per_channel(&q, PAI_DTYPE_I8, 2, scales, zps) ==
        PAI_OK);
  CHECK_EQ_INT(q.scheme, PAI_QUANT_PER_CHANNEL);
  CHECK_EQ_INT(pai_quant_accumulator(&q), PAI_DTYPE_I32);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 0, 10), 5.0f);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 1, 5), 8.0f);
  CHECK_EQ_FLOAT(pai_quant_dequant(&q, 2, 10), 0.0f);

  /* invalid: zero channels, null arrays */
  CHECK(pai_quant_init_per_channel(&q, PAI_DTYPE_I8, 0, scales, zps) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_quant_init_per_channel(&q, PAI_DTYPE_I8, 2, NULL, zps) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_quant_init_per_channel(&q, PAI_DTYPE_F32, 2, scales, zps) ==
        PAI_ERR_INVALID_ARG);

  /* accumulator override */
  CHECK(pai_quant_set_accumulator(&q, PAI_DTYPE_F32) == PAI_OK);
  CHECK_EQ_INT(pai_quant_accumulator(&q), PAI_DTYPE_F32);
  CHECK(pai_quant_set_accumulator(&q, (pai_dtype_t)99) ==
        PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
