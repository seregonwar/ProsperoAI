/*
 * ProsperoAI — OpenAI-compatible gateway (whitepaper §26)
 *
 * OpenAI REST surface over a local registry of .pai models (lazy open,
 * per-model mutex). v0 endpoints: GET /healthz, /v1/models,
 * /v1/models/{id}, POST /v1/completions, /v1/chat/completions,
 * /v1/embeddings. Streaming uses chunked SSE frames
 * (`data: {...}\n\n`, terminal `data: [DONE]`).
 *
 * pai_gw_handle_request is transport-agnostic: the HTTP server feeds
 * it parsed requests and it writes through pai_http_resp_t, so the
 * OpenAI layer is unit-testable without sockets.
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

struct pai_remote_pool; /* opaque persistent payload connection (remote.h) */

typedef struct pai_gw_model_entry {
  char name[256];        /* registry id (basename minus .pai)          */
  char path[1024];       /* local .pai container path                  */
  pai_model_t *model;    /* lazy; NULL until first use (NULL: remote)  */
  pai_session_t *session;
  pai_gw_mutex_t *lock;  /* serializes generation on this model        */
  uint32_t broken;       /* set when open/session init failed          */
  /* Remote (Prospero Protocol) entries: model stays NULL and each
   * generation is bridged to the payload endpoint over TCP (§24/§25).
   * The persistent connection pool is created lazily on first use
   * and reused across requests (reconnecting when the payload closes
   * it); the entry lock serializes all exchanges through it. */
  int remote;
  char remote_host[256];
  uint16_t remote_port;
  struct pai_remote_pool *remote_pool;
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
/*
 * Register a remote payload model (Prospero Protocol endpoint,
 * §24/§25). Generation is bridged per request over TCP; v0 transmits
 * only the prompt (payload defaults for sampling) and does not bridge
 * embeddings. PAI_ERR_MISMATCH on a duplicate name.
 */
pai_status_t pai_gw_add_remote(pai_gw_t *gw, const char *name,
                               const char *host, uint16_t port);
void pai_gw_destroy(pai_gw_t *gw);

/* Find a registry entry by model name; NULL when absent. */
const pai_gw_model_entry_t *pai_gw_find(const pai_gw_t *gw,
                                        const char *name);

/*
 * Process one OpenAI request. `target` is the raw request target
 * (path + optional query); `body`/`body_len` may be NULL/0. The
 * response is written through `resp` (begin/write/end). PAI_OK even
 * when the answer is a 4xx/5xx; PAI_ERR_INVALID_ARG on bad arguments.
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
