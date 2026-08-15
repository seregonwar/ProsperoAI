/*
 * ProsperoAI — public SDK
 *
 * Tensor datatypes. ProsperoAI is quantization-agnostic (whitepaper §15):
 * this enum carries only the storage representation; quantization
 * semantics are described by metadata, never by the dtype alone.
 */

#ifndef PAI_DTYPE_H
#define PAI_DTYPE_H

#include <stdint.h>

typedef enum pai_dtype {
  PAI_DTYPE_F32 = 0,
  PAI_DTYPE_F16,
  PAI_DTYPE_BF16,
  PAI_DTYPE_I32,
  PAI_DTYPE_U32,
  PAI_DTYPE_I16,
  PAI_DTYPE_U16,
  PAI_DTYPE_I8,
  PAI_DTYPE_U8,
  PAI_DTYPE_COUNT
} pai_dtype_t;

/* Storage size of one element in bytes. */
uint32_t pai_dtype_size(pai_dtype_t dtype);

/* Storage size of one element in bits. */
uint32_t pai_dtype_bits(pai_dtype_t dtype);

/* Stable name for diagnostics (e.g. "f32"). */
const char *pai_dtype_name(pai_dtype_t dtype);

#endif /* PAI_DTYPE_H */
