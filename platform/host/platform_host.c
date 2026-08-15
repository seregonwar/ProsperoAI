/*
 * ProsperoAI — host platform detection
 *
 * Reference environment: everything is software-emulated. Used for
 * development, testing and the host-reference execution mode.
 */

#include "platform.h"

#include <pai/log.h>

#include <stdio.h>
#include <string.h>

#ifdef PAI_HOST

pai_status_t
pai_platform_detect(pai_platform_info_t *info) {
  memset(info, 0, sizeof(*info));

  info->fw_version = 0;
  snprintf(info->model, sizeof(info->model), "host-reference");

  info->caps |= PAI_CAP_KERNEL_RW; /* not meaningful on host */
  info->caps |= PAI_CAP_DMEM;      /* malloc-backed */
  info->caps |= PAI_CAP_DEV_GC;    /* PM4 interpreter */
  info->caps |= PAI_CAP_GPU_COMPUTE;

#if defined(__AVX2__)
  info->caps |= PAI_CAP_CPU_AVX2;
#endif

  snprintf(info->fingerprint, sizeof(info->fingerprint), "host-%s",
           info->model);

  return PAI_OK;
}

pai_gpu_backend_t
pai_platform_default_gpu_backend(const pai_platform_info_t *info) {
  (void)info;
  return PAI_GPU_BACKEND_HOST_REF;
}

#endif /* PAI_HOST */
