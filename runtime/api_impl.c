/*
 * ProsperoAI — high-level API implementation (whitepaper §27)
 *
 * The declarations in sdk/include/pai/api.h exist from day one so the
 * ABI stabilizes early. Since the model manager + reference executor
 * landed (models/), the model/session/generate entry points are real:
 * they load `.pai` containers and run the Phase 2 vertical slice.
 *
 * pai_model_close / pai_model_name / pai_session_destroy share their
 * signatures with the model manager and are implemented there; this
 * file only bridges the SDK-only names.
 *
 * pai_embed (Phase 9 seed) maps to the model manager's mean-pooled
 * token-embedding path, exposed by the gateway §26 /v1/embeddings.
 */

#include <pai/api.h>

#include <models/model.h>

pai_status_t
pai_model_open(pai_runtime_t *runtime, const char *path,
               pai_model_t **out_model) {
  (void)runtime;
  return pai_model_open_path(path, out_model);
}

pai_status_t
pai_session_create(pai_model_t *model, pai_session_t **out_session) {
  return pai_session_init(model, out_session);
}

pai_status_t
pai_generate(pai_session_t *session, const char *prompt,
             void (*on_token)(const char *token, void *user), void *user) {
  return pai_session_generate(session, prompt, on_token, user);
}

pai_status_t
pai_embed(pai_model_t *model, const char *text, float *out_embeddings,
          uint32_t max_elements) {
  uint32_t n = 0;
  pai_status_t st =
      pai_model_embed(model, text, max_elements, out_embeddings, &n);
  return st;
}
