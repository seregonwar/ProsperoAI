/*
 * ProsperoAI — OpenAI-compatible gateway (whitepaper §26)
 *
 * The network layer that makes a PS5 (or host) running ProsperoAI
 * usable as a local AI endpoint by existing applications: it exposes
 * the OpenAI REST surface and translates requests into runtime
 * operations.
 *
 * v0 endpoints (the §26 minimum):
 *
 *   GET  /healthz                     liveness
 *   GET  /v1/models                   model listing
 *   GET  /v1/models/{id}              single model
 *   POST /v1/completions              text completions (streamable)
 *   POST /v1/chat/completions         chat completions (streamable)
 *   POST /v1/embeddings               embeddings (mean-pooled token
 *                                     embeddings, Phase 9 seed)
 *
 * Streaming uses chunked transfer encoding with OpenAI-style SSE
 * frames (`data: {...}\n\n`, terminal `data: [DONE]`).
 *
 * The gateway owns a small model registry (§22 repository seed):
 * entries are added by .pai path (or by scanning a directory); models
 * and their sessions are opened lazily on first use and guarded by a
 * per-model mutex so concurrent requests serialize only per model.
 *
 * The core handler pai_gw_handle_request is transport-agnostic: the
 * HTTP server (http.h) feeds it parsed requests and it writes through
 * the pai_http_resp_t sink — which also makes the whole OpenAI layer
 * unit-testable without sockets.
 */

#ifndef PAI_GATEWAY_GATEWAY_H
#define PAI_GATEWAY_GATEWAY_H

#include "http.h"
#include "json.h"

#include <models/model.h>

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_GW_MAX_MODELS 16u

typedef struct pai_gw_model_entry {
  char name[256];        /* registry id (file basename minus .pai)     */
  char path[1024];       /* .pai container path                        */
  pai_model_t *model;    /* lazy; NULL until first use                 */
  pai_session_t *session;
  pai_gw_mutex_t *lock;  /* serializes generation on this model        */
  uint32_t broken;       /* set when open/session init failed          */
} pai_gw_model_entry_t;

typedef struct pai_gw {
  pai_gw_model_entry_t models[PAI_GW_MAX_MODELS];
  uint32_t num_models;
  uint32_t seq;             /* response id counter                        */
  pai_gw_mutex_t *seq_lock; /* serializes seq allocation across entries    */
} pai_gw_t;

/* Registry management. */
pai_status_t pai_gw_init(pai_gw_t *gw);
pai_status_t pai_gw_add_file(pai_gw_t *gw, const char *path);
/* Scan `dir` for *.pai files and add each (best effort). */
pai_status_t pai_gw_add_dir(pai_gw_t *gw, const char *dir);
void pai_gw_destroy(pai_gw_t *gw);

/* Find a registry entry by model name; NULL when absent. */
const pai_gw_model_entry_t *pai_gw_find(const pai_gw_t *gw,
                                        const char *name);

/*
 * Process one OpenAI request. `target` is the raw request target
 * (path + optional query); `body`/`body_len` the request body (may be
 * NULL/0). The response is written through `resp` (begin/write/end).
 * Returns PAI_OK when the request was answered (including 4xx/5xx
 * responses); PAI_ERR_INVALID_ARG on bad arguments.
 */
pai_status_t pai_gw_handle_request(pai_gw_t *gw, const char *method,
                                   const char *target, const uint8_t *body,
                                   uint32_t body_len,
                                   pai_http_resp_t *resp);

/* Derive a registry name from a .pai path (basename minus extension). */
const char *pai_gw_name_from_path(const char *path, char *out,
                                  uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* PAI_GATEWAY_GATEWAY_H */
