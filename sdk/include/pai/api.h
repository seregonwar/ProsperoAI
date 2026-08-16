/*
 * ProsperoAI — public SDK, high-level API surface (whitepaper §27).
 * Stable developer contract for homebrew apps; the surface is
 * intentionally few functions, the rest comes via the Expert API.
 * Entry points exist during Phase 0/1 so the ABI stabilizes early;
 * unimplemented ones return PAI_ERR_UNSUPPORTED.
 */

#ifndef PAI_API_H
#define PAI_API_H

#include <pai/error.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. */
typedef struct pai_runtime    pai_runtime_t;
typedef struct pai_model      pai_model_t;
typedef struct pai_session    pai_session_t;
typedef struct pai_gpu_device pai_gpu_device_t;

/* Runtime lifecycle. */
pai_status_t pai_runtime_init(pai_runtime_t **out_runtime);
void         pai_runtime_shutdown(pai_runtime_t *runtime);
const char  *pai_runtime_version(void);

/*
 * Model management. `.pai` containers and (via adapters) GGUF become
 * the supported inputs; these become functional in Phase 3+.
 */
pai_status_t pai_model_open(pai_runtime_t *runtime, const char *path,
                            pai_model_t **out_model);
void         pai_model_close(pai_model_t *model);
const char  *pai_model_name(const pai_model_t *model);

/* Sessions (Phase 2+). */
pai_status_t pai_session_create(pai_model_t *model, pai_session_t **out_session);
void         pai_session_destroy(pai_session_t *session);

/* Generation (Phase 2+). */
pai_status_t pai_generate(pai_session_t *session, const char *prompt,
                          void (*on_token)(const char *token, void *user),
                          void *user);

/* Embeddings (Phase 9+). */
pai_status_t pai_embed(pai_model_t *model, const char *text,
                       float *out_embeddings, uint32_t max_elements);

#ifdef __cplusplus
}
#endif

#endif /* PAI_API_H */
