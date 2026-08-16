/*
 * ProsperoAI - public SDK: quantization metadata hooks (whitepaper §15).
 * Phase 1 carries the metadata only: no quantized kernels yet. The
 * quant descriptor describes how stored integer elements map to real
 * values; it never overrides the storage dtype in pai_tensor_t.
 */

#ifndef PAI_QUANT_H
#define PAI_QUANT_H

#include <pai/dtype.h>
#include <pai/error.h>

#include <stdint.h>

typedef enum pai_quant_scheme {
  PAI_QUANT_NONE = 0,        /* stored values are the real values      */
  PAI_QUANT_PER_TENSOR,      /* one (scale, zero_point) for the tensor */
  PAI_QUANT_PER_CHANNEL,     /* per-output-channel scale/zero_point    */
  PAI_QUANT_COUNT
} pai_quant_scheme_t;

typedef struct pai_quant {
  pai_quant_scheme_t scheme;
  pai_dtype_t        storage;      /* on-disk dtype (e.g. u8)           */
  pai_dtype_t        accumulator;  /* recommended accumulate dtype      */
  float              scale;        /* PER_TENSOR scale                  */
  float              zero_point;   /* PER_TENSOR zero point             */
  uint32_t           channels;     /* PER_CHANNEL: number of channels   */
  const float       *scale_ch;     /* PER_CHANNEL: scales (owned by the caller) */
  const float       *zero_point_ch; /* PER_CHANNEL: zero points         */
} pai_quant_t;

/* NONE scheme, storage/accumulator = F32, scale 1, zero point 0. */
void pai_quant_init(pai_quant_t *q);

/* PER_TENSOR over an integer storage dtype; sets the accumulator to
 * the dtype rule (pai_dtype_accumulator). Rejects float storage and
 * non-finite or zero scale. */
pai_status_t pai_quant_init_per_tensor(pai_quant_t *q, pai_dtype_t storage,
                                       float scale, float zero_point);

/* PER_CHANNEL over an integer storage dtype. The arrays are
 * referenced, not copied. Rejects float storage, zero channels and
 * NULL arrays. */
pai_status_t pai_quant_init_per_channel(pai_quant_t *q, pai_dtype_t storage,
                                        uint32_t channels,
                                        const float *scale_ch,
                                        const float *zero_point_ch);

/* Override the recommended accumulator dtype. */
pai_status_t pai_quant_set_accumulator(pai_quant_t *q, pai_dtype_t dtype);

/* NONE: stored as-is. PER_TENSOR: (stored - zp) * scale.
 * PER_CHANNEL: (stored - zp_ch[channel]) * scale_ch[channel];
 * out-of-range channels dequantize to 0.0f. */
float pai_quant_dequant(const pai_quant_t *q, uint32_t channel,
                        int32_t stored);

/* The accumulator dtype: set by the inits (the dtype rule on the
 * storage dtype) and overridable with pai_quant_set_accumulator. */
pai_dtype_t pai_quant_accumulator(const pai_quant_t *q);

#endif /* PAI_QUANT_H */