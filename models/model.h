/*
 * ProsperoAI — model manager (whitepaper §20/§22, §27 SDK).
 * Loads `.pai` containers into an executable model and runs sessions:
 * open -> parse sections + rebuild graph, session -> plan memory, load
 * weights, init KV cache (§19)/sampler, generate -> tokenize, execute
 * the plan on the reference backend, sample, decode — streaming tokens.
 */

#ifndef PAI_MODELS_MODEL_H
#define PAI_MODELS_MODEL_H

#include <pai/api.h> /* pai_model_t / pai_session_t opaque handles */
#include <pai/error.h>

#include <cache/kv_cache.h>
#include <graph/graph.h>
#include <ir/ir.h>
#include <pai/pai.h>
#include <scheduler/scheduler.h>

#include <sampler.h>
#include <tokenizer.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Model handle (opaque externally). */
struct pai_model {
  char     name[PAI_PAI_NAME_MAX + 1];
  uint32_t family;
  uint32_t context_len;
  uint32_t num_layers;
  uint32_t kv_bytes_per_token;
  uint32_t vocab_size;

  pai_ir_program_t ir;      /* decoded program                          */
  pai_graph_t graph;        /* rebuilt compute graph                    */
  pai_graph_mem_plan_t mem_plan; /* static memory plan (§16)            */

  pai_pai_tensor_t manifest[PAI_PAI_MAX_TENSORS];
  uint32_t num_tensors;

  uint8_t *weights;         /* canonical weights (owned)                */
  uint64_t weights_bytes;

  pai_tok_t tokenizer;
};

/* Session handle (opaque externally). */
struct pai_session {
  pai_model_t *model;
  pai_sched_plan_t plan;    /* execution plan (§11/§18)                 */
  uint8_t *region;          /* planned storage region (owned)           */
  pai_kv_cache_t kv;        /* per-session KV cache (§19)               */
  pai_sampler_t sampler;    /* sampling config                          */
  uint32_t max_tokens;      /* generation cap                           */
  uint32_t eos_token;       /* stop token (0 = none)                    */
  uint32_t seed;
  int cancelled;            /* early-stop request (see pai_session_cancel) */
  uint64_t generated_tokens;
};

/* Open a `.pai` container from a file. */
pai_status_t pai_model_open_path(const char *path, pai_model_t **out_model);

/* Open from an in-memory container blob (not owned by the model). */
pai_status_t pai_model_open_blob(const uint8_t *blob, uint32_t nbytes,
                                 pai_model_t **out_model);

void pai_model_close(pai_model_t *model);

const char *pai_model_name(const pai_model_t *model);

/* Sessions */

/*
 * Create a session: builds the execution plan (default INTERACTIVE
 * policy), allocates the planned region, copies the canonical weights
 * in, and inits the KV cache + sampler. The session owns `region`
 * until pai_session_destroy.
 */
pai_status_t pai_session_init(pai_model_t *model, pai_session_t **out_session);

void pai_session_destroy(pai_session_t *session);

/* Sampler configuration (call between session init and generate). */
pai_status_t pai_session_set_sampler(pai_session_t *session,
                                     const pai_sampler_t *sampler);

/* Generation parameters (max_tokens, eos_token). */
pai_status_t pai_session_set_generation(pai_session_t *session,
                                        uint32_t max_tokens,
                                        uint32_t eos_token);

/*
 * Generate tokens from `prompt`, calling on_token(text, user) per
 * emitted token (the text is a stack buffer, valid during the call).
 * Stops at eos_token or max_tokens.
 */
pai_status_t pai_session_generate(pai_session_t *session, const char *prompt,
                                  void (*on_token)(const char *token,
                                                   void *user),
                                  void *user);

/*
 * Request early termination of the in-flight generate (§26 stop
 * sequences): safe to call from the on_token callback (the check runs
 * after each emitted token). The token that triggered the cancel has
 * already been emitted; generated_tokens counts it. Cleared at the
 * start of the next pai_session_generate.
 */
void pai_session_cancel(pai_session_t *session);

/* Embeddings (§26 /v1/embeddings) */

/* Embedding dimension of the token-embedding table; PAI_ERR_UNSUPPORTED
 * when the model has none. */
pai_status_t pai_model_embed_dim(const pai_model_t *model, uint32_t *out_dim);

/*
 * Embed `text` as the mean-pooled bag of its token embeddings
 * (dequantized when the container stores them quantized, §15).
 * `max_elements` must be >= dim; byte-fallback ids are skipped;
 * PAI_ERR_MISMATCH when no token contributes.
 */
pai_status_t pai_model_embed(pai_model_t *model, const char *text,
                             uint32_t max_elements, float *out,
                             uint32_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_MODEL_H */
