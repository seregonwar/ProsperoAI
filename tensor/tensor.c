/*
 * ProsperoAI — tensor core
 *
 * Dense row-major tensor descriptors. These are views over storage;
 * ownership of the underlying buffer stays with the caller.
 */

#include <pai/tensor.h>

#include <string.h>

pai_status_t
pai_tensor_init(pai_tensor_t *tensor, pai_dtype_t dtype, uint32_t rank,
                const uint64_t *shape, void *data) {
  uint64_t stride = 1;
  uint64_t bytes = 0;
  uint32_t esize;

  if (!tensor || !shape || rank == 0 || rank > PAI_TENSOR_MAX_RANK) {
    return PAI_ERR_INVALID_ARG;
  }

  esize = pai_dtype_size(dtype);
  if (esize == 0) {
    return PAI_ERR_UNSUPPORTED;
  }

  memset(tensor, 0, sizeof(*tensor));
  tensor->dtype = dtype;
  tensor->rank = rank;
  tensor->data = data;

  for (uint32_t i = 0; i < rank; i++) {
    if (shape[i] == 0) {
      return PAI_ERR_INVALID_ARG;
    }
    tensor->shape[i] = shape[i];
  }

  /* Row-major strides: last dim strides by 1. */
  for (uint32_t i = rank; i > 0; i--) {
    tensor->strides[i - 1] = stride;
    stride *= shape[i - 1];
  }

  bytes = stride * esize;
  tensor->size_bytes = bytes;
  return PAI_OK;
}

uint64_t
pai_tensor_nelem(const pai_tensor_t *tensor) {
  uint64_t n = 1;

  if (!tensor) {
    return 0;
  }

  for (uint32_t i = 0; i < tensor->rank; i++) {
    n *= tensor->shape[i];
  }
  return n;
}

pai_status_t
pai_tensor_contiguous_size(pai_dtype_t dtype, uint32_t rank,
                           const uint64_t *shape, uint64_t *out_bytes) {
  uint64_t n = 1;
  uint32_t esize;

  if (!shape || !out_bytes || rank == 0 || rank > PAI_TENSOR_MAX_RANK) {
    return PAI_ERR_INVALID_ARG;
  }

  esize = pai_dtype_size(dtype);
  if (esize == 0) {
    return PAI_ERR_UNSUPPORTED;
  }

  for (uint32_t i = 0; i < rank; i++) {
    if (shape[i] == 0 || n > UINT64_MAX / shape[i]) {
      *out_bytes = 0;
      return PAI_ERR_INVALID_ARG;
    }
    n *= shape[i];
  }

  if (n > UINT64_MAX / esize) {
    *out_bytes = 0;
    return PAI_ERR_INVALID_ARG;
  }

  *out_bytes = n * esize;
  return PAI_OK;
}

uint64_t
pai_tensor_offset(const pai_tensor_t *tensor, const uint64_t *index) {
  uint64_t off = 0;

  for (uint32_t i = 0; i < tensor->rank; i++) {
    off += index[i] * tensor->strides[i];
  }
  return off;
}
