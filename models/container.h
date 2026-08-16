/*
 * ProsperoAI — shared `.pai` container builder (whitepaper §8/§20).
 * Graph description + raw f32 weights + tokenizer + metadata -> `.pai`
 * blob: graph reconstruction -> Prospero IR -> optional quantization
 * (§15) -> PAI packaging (§20). Shared by the importer and the GGUF
 * adapter.
 */

#ifndef PAI_MODELS_CONTAINER_H
#define PAI_MODELS_CONTAINER_H

#include <pai/error.h>
#include <pai/pai.h>
#include <pai/tensor.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Quantization options shared by all import paths (§15). */
typedef struct pai_container_quant {
  const char *quant;    /* NULL | "none" | "q8" | "q4"                   */
  uint16_t quant_group; /* elements per scale group; 0 = per-tensor      */
} pai_container_quant_t;

/*
 * Value/tensor descriptor. `kind`: 0 input, 1 param, 2 activation,
 * 3 output. Weight-carrying params (kind 1) must provide `name` (the
 * manifest tensor name), `data` and `n`.
 */
typedef struct pai_container_tensor {
  uint32_t id;        /* value id, 1..N ascending declaration order      */
  uint32_t dtype;     /* pai_dtype_t (f32 in v0)                         */
  uint32_t rank;
  uint64_t shape[PAI_TENSOR_MAX_RANK];
  uint32_t kind;      /* 0 input, 1 param, 2 activation, 3 output        */
  const char *name;   /* manifest name for weight-carrying params        */
  const float *data;  /* raw f32 weights, NULL when no payload           */
  uint64_t n;         /* element count of `data`                         */
  uint8_t quantize;   /* 1 = eligible for --quant packing; 0 = keep f32  */
} pai_container_tensor_t;

/* Op descriptor; `id` is 1..M ascending, inputs/outputs are value ids. */
typedef struct pai_container_op {
  uint32_t id;        /* op id, 1..M ascending                           */
  uint32_t kind;      /* pai_graph_op_kind_t                             */
  uint32_t num_in;
  uint32_t num_out;
  const uint32_t *in;
  const uint32_t *out;
} pai_container_op_t;

/* Tokenizer entry: token id + text (1..PAI_TOK_MAX_TOKEN_LEN chars). */
typedef struct pai_container_token {
  uint32_t id;
  const char *text;
} pai_container_token_t;

/* Optional BPE merge rule: (left, right) -> result (token ids). */
typedef struct pai_container_merge {
  uint32_t left;
  uint32_t right;
  uint32_t result;
} pai_container_merge_t;

/*
 * Build a `.pai` container blob. All input arrays are read-only; the
 * returned blob is malloc'd (*out_blob, caller frees). `quant` may be
 * NULL (no quantization). `merges` may be NULL when num_merges == 0.
 */
pai_status_t pai_container_build(
    const pai_pai_meta_t *meta,
    const pai_container_tensor_t *tensors, uint32_t num_tensors,
    const pai_container_op_t *ops, uint32_t num_ops,
    const pai_container_token_t *tokens, uint32_t num_tokens,
    const pai_container_merge_t *merges, uint32_t num_merges,
    const pai_container_quant_t *quant,
    uint8_t **out_blob, uint32_t *out_nbytes);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_CONTAINER_H */
