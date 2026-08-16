/*
 * ProsperoAI — platform/capability detection producing the capability
 * fingerprint (whitepaper §28/§29). Every optional feature is probed
 * at runtime; hard-coded firmware offsets must never be primary.
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

/*
 * Escalate this payload to system credentials and escape the process
 * jail (PS5 only; host builds are no-ops). Required before the runtime
 * can write logs/models under /data. Mirrors MemDBG's jailbreak_self.
 */
pai_status_t pai_platform_escalate(void);

/* Default GPU backend for this platform. */
pai_gpu_backend_t
pai_platform_default_gpu_backend(const pai_platform_info_t *info);

/* Log a human-readable capability summary. */
void pai_platform_log(const pai_platform_info_t *info);

/* Deployment lifecycle (whitepaper §5, §35) */

/* Port the instance lifecycle listener binds: a fresh payload asks any
 * previous instance on this port to terminate before starting, so
 * consecutive deploys can be automated (same pattern as MemDBG). */
#define PAI_LIFECYCLE_PORT 9025u

/*
 * Ask a previous instance (if any listens on `port`) to terminate,
 * then wait for the port to free. PAI_OK when none exists or it
 * terminated in time.
 */
pai_status_t pai_lifecycle_stop_previous(uint16_t port);

/*
 * Bind the lifecycle port and start the listener that terminates this
 * instance when a newer payload asks. PAI_ERR_IO when the port cannot
 * be bound (e.g. another instance won).
 */
pai_status_t pai_lifecycle_start(uint16_t port);

/* On-console notification (PS5) or log line (host). */
void pai_notify(const char *message);

#endif /* PAI_PLATFORM_H */
