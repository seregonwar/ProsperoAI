/*
 * ProsperoAI — public SDK: tensor descriptors. A tensor is a view over
 * storage (shape, row-major strides, dtype, host data pointer);
 * device-resident tensors go through the GPU HAL and keep data == NULL.
 */

#ifndef PAI_TENSOR_H
#define PAI_TENSOR_H

#include <pai/dtype.h>
#include <pai/error.h>

#include <stdint.h>

#define PAI_TENSOR_MAX_RANK 6

typedef struct pai_tensor {
  pai_dtype_t dtype;
  uint32_t    rank;
  uint64_t    shape[PAI_TENSOR_MAX_RANK];   /* element counts per dim     */
  uint64_t    strides[PAI_TENSOR_MAX_RANK]; /* element stride per dim     */
  void       *data;                          /* host pointer, or NULL     */
  uint64_t    size_bytes;                    /* logical payload size      */
} pai_tensor_t;

/*
 * Initialize a dense row-major tensor view over an existing buffer.
 * Computes strides from the shape; does not copy or own `data`.
 */
pai_status_t pai_tensor_init(pai_tensor_t *tensor, pai_dtype_t dtype,
                             uint32_t rank, const uint64_t *shape, void *data);

/* Total number of logical elements. */
uint64_t pai_tensor_nelem(const pai_tensor_t *tensor);

/* Byte size of a dense row-major buffer of the given shape/dtype. */
pai_status_t pai_tensor_contiguous_size(pai_dtype_t dtype, uint32_t rank,
                                        const uint64_t *shape,
                                        uint64_t *out_bytes);

/* Linear element index for a coordinate; bounds are not checked. */
uint64_t pai_tensor_offset(const pai_tensor_t *tensor, const uint64_t *index);

#endif /* PAI_TENSOR_H */
