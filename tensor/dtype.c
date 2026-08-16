/* ProsperoAI - dtype tables shared by the tensor descriptors and the
 * CPU reference backend. Quantization metadata (whitepaper 15) layers
 * on top of these storage primitives without changing them. */

#include <pai/dtype.h>
#include <pai/quant.h>

#include <math.h>
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

pai_dtype_t
pai_dtype_accumulator(pai_dtype_t dtype) {
  switch (dtype) {
    case PAI_DTYPE_F32:
    case PAI_DTYPE_I32:
    case PAI_DTYPE_U32:
      return dtype;
    case PAI_DTYPE_F16:
    case PAI_DTYPE_BF16:
      return PAI_DTYPE_F32;
    case PAI_DTYPE_I16:
    case PAI_DTYPE_U16:
    case PAI_DTYPE_I8:
    case PAI_DTYPE_U8:
      return PAI_DTYPE_I32;
    default:
      return PAI_DTYPE_COUNT;
  }
}

void
pai_quant_init(pai_quant_t *q) {
  if (!q) {
    return;
  }
  q->scheme = PAI_QUANT_NONE;
  q->storage = PAI_DTYPE_F32;
  q->accumulator = PAI_DTYPE_F32;
  q->scale = 1.0f;
  q->zero_point = 0.0f;
  q->channels = 0;
  q->scale_ch = NULL;
  q->zero_point_ch = NULL;
}

static int
pai_quant_storage_ok(pai_dtype_t storage) {
  switch (storage) {
    case PAI_DTYPE_I16:
    case PAI_DTYPE_U16:
    case PAI_DTYPE_I8:
    case PAI_DTYPE_U8:
      return 1;
    default:
      return 0;
  }
}

pai_status_t
pai_quant_init_per_tensor(pai_quant_t *q, pai_dtype_t storage, float scale,
                          float zero_point) {
  if (!q) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!pai_quant_storage_ok(storage)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!isfinite(scale) || scale == 0.0f) {
    return PAI_ERR_INVALID_ARG;
  }
  q->scheme = PAI_QUANT_PER_TENSOR;
  q->storage = storage;
  q->accumulator = pai_dtype_accumulator(storage);
  q->scale = scale;
  q->zero_point = zero_point;
  q->channels = 0;
  q->scale_ch = NULL;
  q->zero_point_ch = NULL;
  return PAI_OK;
}

pai_status_t
pai_quant_init_per_channel(pai_quant_t *q, pai_dtype_t storage,
                           uint32_t channels, const float *scale_ch,
                           const float *zero_point_ch) {
  if (!q) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!pai_quant_storage_ok(storage)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (channels == 0 || !scale_ch || !zero_point_ch) {
    return PAI_ERR_INVALID_ARG;
  }
  q->scheme = PAI_QUANT_PER_CHANNEL;
  q->storage = storage;
  q->accumulator = pai_dtype_accumulator(storage);
  q->scale = 1.0f;
  q->zero_point = 0.0f;
  q->channels = channels;
  q->scale_ch = scale_ch;
  q->zero_point_ch = zero_point_ch;
  return PAI_OK;
}

pai_status_t
pai_quant_set_accumulator(pai_quant_t *q, pai_dtype_t dtype) {
  if (!q || (uint32_t)dtype >= PAI_DTYPE_COUNT) {
    return PAI_ERR_INVALID_ARG;
  }
  q->accumulator = dtype;
  return PAI_OK;
}

float
pai_quant_dequant(const pai_quant_t *q, uint32_t channel, int32_t stored) {
  if (!q) {
    return (float)stored;
  }
  switch (q->scheme) {
    case PAI_QUANT_PER_TENSOR:
      return ((float)stored - q->zero_point) * q->scale;
    case PAI_QUANT_PER_CHANNEL:
      if (channel >= q->channels || !q->scale_ch || !q->zero_point_ch) {
        return 0.0f;
      }
      return ((float)stored - q->zero_point_ch[channel]) *
             q->scale_ch[channel];
    case PAI_QUANT_NONE:
    default:
      return (float)stored;
  }
}

pai_dtype_t
pai_quant_accumulator(const pai_quant_t *q) {
  if (!q) {
    return PAI_DTYPE_COUNT;
  }
  return q->accumulator;
}