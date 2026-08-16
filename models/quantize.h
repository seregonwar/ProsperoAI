/*
 * ProsperoAI — quantization (§15, §10.1 metadata).
 * The runtime is quantization-agnostic; this module implements the v0
 * reference schemes (symmetric, per-group scale) using the metadata the
 * Prospero IR carries (pai_ir_quant_t). The model loader dequantizes
 * canonical weights back to f32 on the reference path.
 *
 * v0 schemes:
 *
 *   q8  bit_width 8,  signed, values in [-127, 127]
 *   q4  bit_width 4,  signed, values in [-7, 7], packed two per byte
 *       (element 2i -> low nibble, 2i+1 -> high nibble, sign-extended)
 *
 * Both are symmetric per-group: one f32 scale per group of group_size
 * consecutive elements (group_size 0 = per-tensor block, §15). The
 * packed weight blob layout for n elements is:
 *
 *   [ packed quantized values ] [ f32 scales x num_groups ]
 *
 * which is what the manifest records as the tensor's size_bytes.
 */

#ifndef PAI_MODELS_QUANTIZE_H
#define PAI_MODELS_QUANTIZE_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_QUANT_GROUP_AUTO 0u /* per-tensor block (one group)          */

typedef struct pai_quant_scheme {
  uint8_t  bit_width;  /* storage bits per element (4 or 8)              */
  uint8_t  is_signed;  /* v0: signed symmetric only                      */
  uint16_t group_size; /* elements per group; PAI_QUANT_GROUP_AUTO = all */
} pai_quant_scheme_t;

/* Validate a scheme; returns PAI_ERR_UNSUPPORTED for schemes the v0
 * runtime does not execute. */
pai_status_t pai_quant_scheme_check(const pai_quant_scheme_t *scheme);

/* Number of scale groups for n elements under the scheme. */
uint64_t pai_quant_num_groups(uint64_t n, const pai_quant_scheme_t *scheme);

/* Packed value bytes for n elements (no scales). */
uint64_t pai_quant_value_bytes(uint64_t n, const pai_quant_scheme_t *scheme);

/* Full on-disk weight bytes: packed values + f32 scales. This is what
 * a manifest tensor entry must record as size_bytes. */
uint64_t pai_quant_weights_bytes(uint64_t n, const pai_quant_scheme_t *scheme);

/* Bytes of the scale array for n elements. */
uint64_t pai_quant_scale_bytes(uint64_t n, const pai_quant_scheme_t *scheme);

/*
 * Quantize n f32 elements into out_q (values) and out_scales (one f32
 * per group). out_q and out_scales must not overlap src.
 */
pai_status_t pai_quantize_f32(const float *src, uint64_t n,
                              const pai_quant_scheme_t *scheme, int8_t *out_q,
                              float *out_scales, uint64_t *out_num_groups);

/* Dequantize a packed blob (values + scales, as produced by
 * pai_quantize_f32) back to f32. out must not overlap q/scales. */
pai_status_t pai_dequantize_f32(const int8_t *q, const float *scales,
                                uint64_t n, const pai_quant_scheme_t *scheme,
                                float *out);

/* Error metrics between two f32 buffers: max absolute error and RMSE. */
pai_status_t pai_quant_error_f32(const float *a, const float *b, uint64_t n,
                                 float *out_max_abs, float *out_rmse);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_QUANTIZE_H */
