/*
 * ProsperoAI — tensor core
 *
 * dtype tables shared by the tensor descriptors and the CPU reference
 * backend. Quantization metadata (whitepaper §15) will layer on top of
 * these storage primitives without changing them.
 */

#include <pai/dtype.h>

#include <stddef.h>

typedef struct pai_dtype_info {
  uint32_t    bits;
  uint32_t    size; /* bytes; 0 for sub-byte storage */
  const char *name;
} pai_dtype_info_t;

static const pai_dtype_info_t k_pai_dtype_info[PAI_DTYPE_COUNT] = {
    [PAI_DTYPE_F32]  = {32, 4, "f32"},
    [PAI_DTYPE_F16]  = {16, 2, "f16"},
    [PAI_DTYPE_BF16] = {16, 2, "bf16"},
    [PAI_DTYPE_I32]  = {32, 4, "i32"},
    [PAI_DTYPE_U32]  = {32, 4, "u32"},
    [PAI_DTYPE_I16]  = {16, 2, "i16"},
    [PAI_DTYPE_U16]  = {16, 2, "u16"},
    [PAI_DTYPE_I8]   = {8, 1, "i8"},
    [PAI_DTYPE_U8]   = {8, 1, "u8"},
};

uint32_t
pai_dtype_size(pai_dtype_t dtype) {
  if ((uint32_t)dtype >= PAI_DTYPE_COUNT) {
    return 0;
  }
  return k_pai_dtype_info[dtype].size;
}

uint32_t
pai_dtype_bits(pai_dtype_t dtype) {
  if ((uint32_t)dtype >= PAI_DTYPE_COUNT) {
    return 0;
  }
  return k_pai_dtype_info[dtype].bits;
}

const char *
pai_dtype_name(pai_dtype_t dtype) {
  if ((uint32_t)dtype >= PAI_DTYPE_COUNT) {
    return "?";
  }
  return k_pai_dtype_info[dtype].name;
}
