/*
 * ProsperoAI — platform layer
 *
 * Platform/capability detection producing the capability fingerprint
 * (whitepaper §28/§29). Every optional feature is probed at runtime;
 * hard-coded firmware offsets must never be the primary architecture.
 */

#ifndef PAI_PLATFORM_H
#define PAI_PLATFORM_H

#include <pai/error.h>
#include <hal/hal.h>

#include <stdint.h>

/* Capability flags. */
#define PAI_CAP_KERNEL_RW   (1u << 0)  /* kernel read/write primitives      */
#define PAI_CAP_DMEM        (1u << 1)  /* direct-memory (GPU-visible) alloc */
#define PAI_CAP_DEV_GC      (1u << 2)  /* /dev/gc GPU command submission    */
#define PAI_CAP_GPU_COMPUTE (1u << 3)  /* compute dispatch validated        */
#define PAI_CAP_CPU_AVX2    (1u << 4)  /* host: AVX2 available              */
#define PAI_CAP_SAFE_MODE   (1u << 5)  /* safe mode build                   */

typedef struct pai_platform_info {
  uint32_t fw_version;     /* 0 on host builds */
  char     model[32];
  uint32_t caps;
  char     fingerprint[65]; /* M0: fw/model hash; grows per §29 */
} pai_platform_info_t;

/* Probe the environment. Safe to call once at runtime init. */
pai_status_t pai_platform_detect(pai_platform_info_t *info);

/* Default GPU backend for this platform. */
pai_gpu_backend_t
pai_platform_default_gpu_backend(const pai_platform_info_t *info);

/* Log a human-readable capability summary. */
void pai_platform_log(const pai_platform_info_t *info);

#endif /* PAI_PLATFORM_H */
