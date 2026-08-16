/*
 * ProsperoAI — model importer (whitepaper §8)
 *
 * The host-side model import pipeline, first rung of the Desktop
 * toolchain (§7): a small text model description plus raw f32 weight
 * files become a `.pai` container (graph reconstruction -> Prospero IR
 * -> optional quantization -> PAI packaging).
 *
 * Description format (one directive per line, '#' comments):
 *
 *   name tiny-lm                 model name (meta)
 *   family llm                   llm | generic
 *   context 16                   context length (KV sizing)
 *   layers 1                     transformer layers
 *   kv_bytes 8                   KV bytes per layer per token
 *   vocab 4                      vocabulary size
 *
 *   token 0 a                    tokenizer: id + text
 *   token 1 b
 *
 *   value 1 f32 [4] input                    dtypes: f32 (v0)
 *   value 2 f32 [8,4] param weights=tok_embed.bin
 *   value 3 f32 [8] activation
 *   value 4 f32 [4,8] param weights=w_out.bin
 *   value 5 f32 [4] output
 *
 *   op 1 gemv 2,1 -> 3           op <id> <kind> <in>,... -> <out>,...
 *   op 2 gemv 4,3 -> 5
 *   op 3 softmax 5 -> 6
 *
 * Value kinds: input, param, activation, output. Param values carry a
 * weight file (resolved relative to the description file). Op kinds:
 * add, mul, gemm, gemv, matmul, relu, softmax, layernorm, rmsnorm,
 * rope, attention, reshape, concat, convert, copy.
 *
 * Value ids must be 1..N in ascending declaration order (they become
 * the graph/IR value ids). Op ids must be 1..M in declaration order.
 *
 * With quantization enabled (--quant q8|q4), param weights are packed
 * with per-group scales (§15) and the IR carries the quant metadata;
 * the resulting container still runs through the reference executor
 * (the model loader dequantizes on session init).
 */

#ifndef PAI_MODELS_IMPORTER_H
#define PAI_MODELS_IMPORTER_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pai_import_options {
  const char *quant;    /* NULL | "none" | "q8" | "q4"                   */
  uint16_t quant_group; /* elements per scale group; 0 = per-tensor      */
} pai_import_options_t;

/*
 * Import a model description into a `.pai` container blob. The blob is
 * malloc'd and returned through *out_blob (caller frees). Weight files
 * are resolved relative to the description file's directory.
 */
pai_status_t pai_import_model(const char *desc_path,
                              const pai_import_options_t *opts,
                              uint8_t **out_blob, uint32_t *out_nbytes);

/* Import and write the container to `out_path`. */
pai_status_t pai_import_model_to_file(const char *desc_path,
                                      const char *out_path,
                                      const pai_import_options_t *opts);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_IMPORTER_H */
