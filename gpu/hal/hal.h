/*
 * ProsperoAI — GPU Hardware Abstraction Layer
 *
 * Device/buffer/submission interface implemented by:
 *   - backend_ps5_gc.c:  raw PM4 submission through /dev/gc (PS5)
 *   - backend_host_ref.c: reference interpreter for host builds
 *
 * The HAL is deliberately small: Phase 0 only needs GPU-visible
 * buffers and a fire-and-wait submission primitive. Queues, async
 * fences and compute rings arrive with the scheduler (Phase 1+).
 */

#ifndef PAI_GPU_HAL_H
#define PAI_GPU_HAL_H

#include <pai/error.h>

#include <stdint.h>

typedef enum pai_gpu_backend {
  PAI_GPU_BACKEND_NONE = 0,
  PAI_GPU_BACKEND_PS5_GC,
  PAI_GPU_BACKEND_HOST_REF
} pai_gpu_backend_t;

typedef struct pai_gpu_device pai_gpu_device_t;

/* Buffer flags. */
#define PAI_GPU_BUF_CPU_VISIBLE   (1u << 0)
#define PAI_GPU_BUF_CPU_COHERENT  (1u << 1)

typedef struct pai_gpu_buffer {
  uint64_t size;       /* logical size requested        */
  uint64_t gpu_addr;   /* GPU virtual address           */
  void    *cpu_addr;   /* CPU mapping, or NULL          */
  uint32_t flags;
  void    *backend;    /* opaque backend state          */
} pai_gpu_buffer_t;

/* Buffer allocation granularity (PS5 direct-memory constraints). */
#define PAI_GPU_ALLOC_ALIGN (2u * 1024u * 1024u)

/* Default /dev/gc queue type used by pai_gpu_submit(). */
#define PAI_GPU_QUEUE_DEFAULT 3u

/* Backend operations — implemented by each backend, registered in hal.c. */
typedef struct pai_gpu_backend_ops {
  const char *name;
  pai_status_t (*init)(pai_gpu_device_t *device);
  pai_status_t (*shutdown)(pai_gpu_device_t *device);
  pai_status_t (*buffer_alloc)(pai_gpu_device_t *device,
                               pai_gpu_buffer_t *buffer, uint64_t size,
                               uint32_t flags);
  void (*buffer_free)(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer);
  pai_status_t (*submit)(pai_gpu_device_t *device, const uint32_t *pm4,
                         uint32_t dwords, uint32_t queue_type);
  pai_status_t (*wait_label)(pai_gpu_device_t *device, uint64_t label_addr,
                             uint32_t label_value, uint64_t timeout_ns);
  pai_status_t (*reset)(pai_gpu_device_t *device);
} pai_gpu_backend_ops_t;

struct pai_gpu_device {
  pai_gpu_backend_t backend;
  const pai_gpu_backend_ops_t *ops;
  void *state;
};

/*
 * Create the primary compute device. `backend` selects the
 * implementation; NONE lets the runtime choose the platform default.
 */
pai_status_t pai_gpu_device_create(pai_gpu_backend_t backend,
                                   pai_gpu_device_t **out_device);
void pai_gpu_device_destroy(pai_gpu_device_t *device);
const char *pai_gpu_device_name(const pai_gpu_device_t *device);
pai_gpu_backend_t pai_gpu_device_backend(const pai_gpu_device_t *device);

/* Allocate GPU-visible memory. On PS5 this is direct memory (unified). */
pai_status_t pai_gpu_buffer_alloc(pai_gpu_device_t *device,
                                  pai_gpu_buffer_t *buffer, uint64_t size,
                                  uint32_t flags);
void pai_gpu_buffer_free(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer);

/*
 * Submit a PM4 command stream. Completion is signaled by the stream
 * itself (the caller appends an EOP/RELEASE_MEM packet writing
 * `label_value` to `label_addr`), then pai_gpu_wait_label() polls it.
 */
pai_status_t pai_gpu_submit(pai_gpu_device_t *device, const uint32_t *pm4,
                            uint32_t dwords);

/* Submit on an explicit queue type (PS5 /dev/gc semantics). */
pai_status_t pai_gpu_submit_q(pai_gpu_device_t *device, const uint32_t *pm4,
                              uint32_t dwords, uint32_t queue_type);

/* Poll a GPU-written label; PAI_ERR_TIMEOUT if not observed in time. */
pai_status_t pai_gpu_wait_label(pai_gpu_device_t *device, uint64_t label_addr,
                                uint32_t label_value, uint64_t timeout_ns);

/*
 * Bring-up recovery: tear down and re-create the underlying device
 * connection (/dev/gc + register-space mmap on PS5) so a wedged ring
 * does not poison subsequent experiments.
 */
pai_status_t pai_gpu_reset(pai_gpu_device_t *device);

/* Convenience: submit + wait_label. */
pai_status_t pai_gpu_submit_wait(pai_gpu_device_t *device,
                                 const uint32_t *pm4, uint32_t dwords,
                                 uint64_t label_addr, uint32_t label_value,
                                 uint64_t timeout_ns);

/*
 * Host reference backend only: register a host function that emulates
 * the GPU shader whose code bytes live at `code_addr` (the address the
 * shader buffer was uploaded to). Called on IT_DISPATCH_DIRECT with the
 * current COMPUTE_USER_DATA_0..15 values, thread count and group count.
 */
typedef pai_status_t (*pai_host_kernel_fn)(void *ctx,
                                           const uint32_t user_data[16],
                                           uint32_t threads_x,
                                           uint32_t group_x);

pai_status_t pai_gpu_host_register_shader(pai_gpu_device_t *device,
                                          uint64_t code_addr,
                                          pai_host_kernel_fn fn, void *ctx);

#endif /* PAI_GPU_HAL_H */
