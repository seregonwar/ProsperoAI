#include "runtime.h"

#include <pai/log.h>
#include <pai/version.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

pai_status_t
pai_runtime_init(pai_runtime_t **out_runtime) {
  pai_runtime_t *rt;
  pai_gpu_backend_t backend;
  pai_status_t st;

  if (!out_runtime) {
    return PAI_ERR_INVALID_ARG;
  }

  rt = (pai_runtime_t *)calloc(1, sizeof(*rt));
  if (!rt) {
    return PAI_ERR_NOMEM;
  }

  pai_arena_init(&rt->arena, rt->arena_mem, sizeof(rt->arena_mem));

  PAI_LOG_INFO_(PAI_SUB_CORE, "ProsperoAI %s (%s) starting\n",
                pai_version_string(), PAI_MILESTONE);

  st = pai_platform_detect(&rt->platform);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_CORE, "platform detection failed\n");
    free(rt);
    return st;
  }
  pai_platform_log(&rt->platform);

  backend = pai_platform_default_gpu_backend(&rt->platform);
  if (backend == PAI_GPU_BACKEND_NONE) {
    PAI_LOG_WARN_(PAI_SUB_CORE,
                  "no GPU backend available on this platform; "
                  "GPU stages will be skipped\n");
  } else {
    st = pai_gpu_device_create(backend, &rt->gpu);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_CORE, "GPU device creation failed\n");
      free(rt);
      return st;
    }
  }

  rt->initialized = 1;
  *out_runtime = rt;
  return PAI_OK;
}

void
pai_runtime_shutdown(pai_runtime_t *runtime) {
  if (!runtime) {
    return;
  }

  if (runtime->gpu) {
    pai_gpu_device_destroy(runtime->gpu);
    runtime->gpu = NULL;
  }
  runtime->initialized = 0;
  free(runtime);
}

const char *
pai_runtime_version(void) {
  return pai_version_string();
}

const pai_platform_info_t *
pai_runtime_platform(const pai_runtime_t *runtime) {
  return &runtime->platform;
}

pai_gpu_device_t *
pai_runtime_gpu(const pai_runtime_t *runtime) {
  return runtime->gpu;
}

pai_arena_t *
pai_runtime_arena(pai_runtime_t *runtime) {
  return &runtime->arena;
}
