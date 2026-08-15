/*
 * ProsperoAI — high-level API stubs (whitepaper §27)
 *
 * The declarations in sdk/include/pai/api.h exist from day one so the
 * ABI stabilizes early. Functionality arrives with Phases 2+; until
 * then these return PAI_ERR_UNSUPPORTED.
 */

#include <pai/api.h>

pai_status_t
pai_model_open(pai_runtime_t *runtime, const char *path,
               pai_model_t **out_model) {
  (void)runtime;
  (void)path;
  (void)out_model;
  return PAI_ERR_UNSUPPORTED;
}

void
pai_model_close(pai_model_t *model) {
  (void)model;
}

const char *
pai_model_name(const pai_model_t *model) {
  (void)model;
  return NULL;
}

pai_status_t
pai_session_create(pai_model_t *model, pai_session_t **out_session) {
  (void)model;
  (void)out_session;
  return PAI_ERR_UNSUPPORTED;
}

void
pai_session_destroy(pai_session_t *session) {
  (void)session;
}

pai_status_t
pai_generate(pai_session_t *session, const char *prompt,
             void (*on_token)(const char *token, void *user), void *user) {
  (void)session;
  (void)prompt;
  (void)on_token;
  (void)user;
  return PAI_ERR_UNSUPPORTED;
}

pai_status_t
pai_embed(pai_model_t *model, const char *text, float *out_embeddings,
          uint32_t max_elements) {
  (void)model;
  (void)text;
  (void)out_embeddings;
  (void)max_elements;
  return PAI_ERR_UNSUPPORTED;
}
