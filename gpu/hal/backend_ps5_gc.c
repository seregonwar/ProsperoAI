/*
 * ProsperoAI — PS5 GPU backend (raw /dev/gc submission)
 *
 * Phase 0 bring-up path. No GNM/AGC driver involvement:
 *   - GPU-visible memory via the direct-memory syscalls
 *     (sceKernelAllocateMainDirectMemory / sceKernelMapNamedDirectMemory)
 *   - command submission via the kernel /dev/gc ioctl nr=0x02
 *     (AGC_GC_IOCTL_SUBMIT_16, 0xC0108102), identical across standard
 *     firmwares 3.20-12.70 per OpenAGC's RE
 *   - completion via an appended IT_RELEASE_MEM EOP fence writing a
 *     label that the CPU polls (OpenAGC sceAgcDcbSetEopFlip layout)
 *
 * ABI references (see notes/re/940-gc-ioctl.md):
 *   AgcGcSubmitArgs    { u32 queue_type=3; u32 num_cbs; u64 cb_array }
 *   AgcGcCommandBuffer { u64 header([63:32]=ib_lo,[31:0]=0xC0023F00);
 *                        u64 ib_base([63:32]=ib_size,[31:0]=ib_hi)  }
 *
 * In payload contexts the graphics ring can defer the final descriptor;
 * every submit therefore carries a trailing 16-dword NOP IB
 * (FW 5.50-hardware-proven payload completion sequence, OpenAGC).
 */

#include <hal/hal.h>
#include <pm4/pm4.h>

#include <pai/log.h>

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#ifdef PAI_PS5

#define GC_DEVICE_PATH "/dev/gc"

#define AGC_GC_IOCTL_SUBMIT_16  0xC0108102u /* nr=0x02, RW, 16 bytes */
#define AGC_GC_IOCTL_CONTEXT_QUERY 0xC004812Eu /* nr=0x2e, R, 4 bytes */

/* GPU register space (OpenAGC agc_ioctl.h, SPRX-confirmed): mapped on
 * the gc fd when the context query reports an uninitialized context
 * (capability lower 16 bits == 0). */
#define AGC_GC_MMIO_BASE 0xFE0200000ULL
#define AGC_GC_MMIO_SIZE 0x4000u
#define AGC_GC_MMIO_PROT 0x22u /* WRITE | GPU_WRITE */

#define PROT_GPU_READ  0x10
#define PROT_GPU_WRITE 0x20
#define MAP_NO_COALESCE 0x400000

#define PAI_GC_CB_BUF_SIZE (2u * 1024u * 1024u)

extern int sceKernelAllocateMainDirectMemory(size_t len, size_t alignment,
                                             int type, off_t *out);
extern int sceKernelMapNamedDirectMemory(void **addr, size_t len, int prot,
                                         int flags, off_t phys, size_t align,
                                         const char *name);
extern int sceKernelSetVirtualRangeName(const void *addr, size_t len,
                                        const char *name);

typedef struct {
  uint32_t queue_type;
  uint32_t num_cbs;
  uint64_t cb_array;
} pai_gc_submit_args_t;

typedef struct {
  uint64_t header;  /* [63:32]=ib_base_lo, [31:0]=PM4 IB header */
  uint64_t ib_base; /* [63:32]=ib_size, [31:0]=ib_base_hi */
} pai_gc_cb_descriptor_t;

typedef struct pai_ps5_gc_state {
  int fd;
  uint32_t ctx_caps;
  uint32_t submissions;
  void *mmio;
  pai_gpu_buffer_t *cb_buf;
  pai_gpu_buffer_t *trailer_buf;
  uint32_t cb_words;
} pai_ps5_gc_state_t;

static void
pai_gc_sleep_us(uint64_t us) {
  struct timespec ts = {(time_t)(us / 1000000u), (long)(us % 1000000u) * 1000L};
  nanosleep(&ts, NULL);
}

static uint64_t
pai_gc_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

static pai_status_t
pai_gc_alloc_dmem(pai_gpu_buffer_t *buffer, uint64_t size, const char *name) {
  off_t phys = 0;
  void *va = NULL;
  int r;

  if (size < PAI_GPU_ALLOC_ALIGN) {
    size = PAI_GPU_ALLOC_ALIGN;
  }
  size = (size + PAI_GPU_ALLOC_ALIGN - 1) & ~(uint64_t)(PAI_GPU_ALLOC_ALIGN - 1);

  r = sceKernelAllocateMainDirectMemory((size_t)size, (size_t)size, 1, &phys);
  if (r != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "sceKernelAllocateMainDirectMemory(%s) failed: 0x%08x\n",
                   name, r);
    return PAI_ERR_NOMEM;
  }

  r = sceKernelMapNamedDirectMemory(&va, (size_t)size,
                                    PROT_READ | PROT_WRITE | PROT_GPU_READ |
                                        PROT_GPU_WRITE,
                                    MAP_NO_COALESCE, phys, (size_t)size, name);
  if (r != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "sceKernelMapNamedDirectMemory(%s) failed: 0x%08x\n", name,
                   r);
    return PAI_ERR_NOMEM;
  }

  buffer->size = size;
  buffer->gpu_addr = (uint64_t)(uintptr_t)va;
  buffer->cpu_addr = va;
  buffer->flags = PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_CPU_COHERENT;
  return PAI_OK;
}

static pai_status_t
pai_gc_buffer_alloc(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer,
                    uint64_t size, uint32_t flags) {
  (void)device;
  (void)flags;
  return pai_gc_alloc_dmem(buffer, size, "pai-gpu");
}

static void
pai_gc_buffer_free(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer) {
  (void)device;
  if (buffer->cpu_addr) {
    munmap(buffer->cpu_addr, (size_t)buffer->size);
    buffer->cpu_addr = NULL;
  }
}

/*
 * Build the two command-buffer descriptors and submit.
 * cb_descs must hold 2 descriptors; the caller's PM4 stream (including
 * any completion packets) is already in the cb buffer.
 */
static pai_status_t
pai_gc_submit(pai_ps5_gc_state_t *st, uint32_t cb_words, uint32_t queue_type) {
  pai_gc_cb_descriptor_t descs[2];
  pai_gc_submit_args_t args;
  uint64_t cb_addr = st->cb_buf->gpu_addr;
  uint64_t trailer_addr = st->trailer_buf->gpu_addr;
  int r;

  /* Descriptor: IT_INDIRECT_BUFFER header, base, size. The kernel
   * inserts the VMID into ib_base[63:52] after copyin. */
  descs[0].header = ((cb_addr & 0xFFFFFFFFu) << 32) | 0xC0023F00u;
  descs[0].ib_base = ((uint64_t)cb_words << 32) | ((cb_addr >> 32) & 0xFFFFu);
  descs[1].header = ((trailer_addr & 0xFFFFFFFFu) << 32) | 0xC0023F00u;
  descs[1].ib_base = ((uint64_t)16u << 32) | ((trailer_addr >> 32) & 0xFFFFu);

  args.queue_type = queue_type;
  args.num_cbs = 2;
  args.cb_array = (uint64_t)(uintptr_t)&descs[0];

  r = ioctl(st->fd, AGC_GC_IOCTL_SUBMIT_16, &args);
  if (r != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "gc submit ioctl failed (q=%u): %s (errno %d)\n",
                   queue_type, strerror(errno), errno);
    return PAI_ERR_IO;
  }

  st->submissions++;
  PAI_LOG_DEBUG_(PAI_SUB_GPU,
                 "gc submit #%u: q=%u cbs=2 words=%u\n", st->submissions,
                 queue_type, cb_words);
  return PAI_OK;
}

static pai_status_t
pai_gc_submit_stream(pai_gpu_device_t *device, const uint32_t *pm4,
                     uint32_t dwords, uint32_t queue_type) {
  pai_ps5_gc_state_t *st = (pai_ps5_gc_state_t *)device->state;

  if (!st->fd || !st->cb_buf || !pm4 || dwords == 0 ||
      dwords > 0xFFFFFu) {
    return PAI_ERR_INVALID_ARG;
  }

  if ((uint64_t)dwords * 4 > PAI_GC_CB_BUF_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }

  memcpy(st->cb_buf->cpu_addr, pm4, (size_t)dwords * 4);
  return pai_gc_submit(st, dwords, queue_type);
}

static pai_status_t
pai_gc_wait_label(pai_gpu_device_t *device, uint64_t label_addr,
                  uint32_t label_value, uint64_t timeout_ns) {
  uint64_t deadline;

  (void)device;

  if (label_addr == 0) {
    return PAI_ERR_INVALID_ARG;
  }

  /* Poll the label. Direct memory is GPU-coherent; the CPU observes the
   * write once the EOP event has completed. */
  deadline = pai_gc_now_ns() + timeout_ns;
  for (;;) {
    volatile uint32_t *label = (volatile uint32_t *)(uintptr_t)label_addr;
    if (*label == label_value) {
      return PAI_OK;
    }
    if (pai_gc_now_ns() >= deadline) {
      break;
    }
    pai_gc_sleep_us(100);
  }

  PAI_LOG_ERROR_(PAI_SUB_GPU, "label wait timed out (label=0x%llx value=0x%x)\n",
                 (unsigned long long)label_addr, label_value);
  return PAI_ERR_TIMEOUT;
}

static pai_status_t
pai_gc_init(pai_gpu_device_t *device) {
  pai_ps5_gc_state_t *st;
  pai_gpu_buffer_t *trailer;
  uint32_t query = 0;
  int r;

  st = (pai_ps5_gc_state_t *)calloc(1, sizeof(*st));
  if (!st) {
    return PAI_ERR_NOMEM;
  }
  st->fd = -1;
  device->state = st;

  st->fd = open(GC_DEVICE_PATH, O_RDWR);
  if (st->fd < 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "open(%s) failed: %s (errno %d)\n",
                   GC_DEVICE_PATH, strerror(errno), errno);
    return PAI_ERR_CAPABILITY;
  }

  r = ioctl(st->fd, AGC_GC_IOCTL_CONTEXT_QUERY, &query);
  if (r == 0) {
    st->ctx_caps = query;
    PAI_LOG_INFO_(PAI_SUB_GPU, "gc context query: caps=0x%08x\n", query);
  } else {
    PAI_LOG_WARN_(PAI_SUB_GPU,
                  "gc context query unavailable (errno %d); continuing\n",
                  errno);
  }

  /* SPRX-confirmed: when the context is not yet initialized (lower 16
   * bits of the capability are zero), the driver maps the GPU register
   * space on the gc fd at the fixed 0xFE0200000 address. Without this
   * step the kernel never runs compute dispatches for the process. */
  if ((st->ctx_caps & 0xFFFFu) == 0) {
    void *mmio = mmap((void *)AGC_GC_MMIO_BASE, AGC_GC_MMIO_SIZE,
                      AGC_GC_MMIO_PROT, MAP_SHARED, st->fd, 0);
    if (mmio == MAP_FAILED) {
      PAI_LOG_WARN_(PAI_SUB_GPU,
                    "gc register space mmap failed (errno %d); compute may "
                    "not execute\n",
                    errno);
    } else {
      st->mmio = mmio;
      sceKernelSetVirtualRangeName(mmio, AGC_GC_MMIO_SIZE, "SceGnmDingDong");
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "gc register space mapped at 0x%llx (context init)\n",
                    (unsigned long long)AGC_GC_MMIO_BASE);

      /* Re-query: an initialized context reports nonzero capabilities. */
      query = 0;
      if (ioctl(st->fd, AGC_GC_IOCTL_CONTEXT_QUERY, &query) == 0) {
        st->ctx_caps = query;
        PAI_LOG_INFO_(PAI_SUB_GPU, "gc context query (post-mmap): caps=0x%08x\n",
                      query);
      }
    }
  }

  st->cb_buf = (pai_gpu_buffer_t *)calloc(1, sizeof(*st->cb_buf));
  if (!st->cb_buf ||
      pai_gc_alloc_dmem(st->cb_buf, PAI_GC_CB_BUF_SIZE, "pai-cb") != PAI_OK) {
    return PAI_ERR_NOMEM;
  }

  /* Completion trailer: a 16-dword NOP indirect buffer that forces the
   * ring to execute the final descriptor in this frame. */
  trailer = (pai_gpu_buffer_t *)calloc(1, sizeof(*trailer));
  if (!trailer ||
      pai_gc_alloc_dmem(trailer, PAI_GPU_ALLOC_ALIGN, "pai-trailer") !=
          PAI_OK) {
    return PAI_ERR_NOMEM;
  }
  st->trailer_buf = trailer;
  {
    uint32_t *t = (uint32_t *)trailer->cpu_addr;
    t[0] = pai_pm4_header3(PAI_PM4_OP_NOP, 16);
    for (int i = 1; i < 16; i++) {
      t[i] = 0;
    }
  }

  return PAI_OK;
}

static pai_status_t
pai_gc_shutdown(pai_gpu_device_t *device) {
  pai_ps5_gc_state_t *st = (pai_ps5_gc_state_t *)device->state;

  if (st) {
    if (st->cb_buf) {
      pai_gc_buffer_free(device, st->cb_buf);
      free(st->cb_buf);
    }
    if (st->trailer_buf) {
      pai_gc_buffer_free(device, st->trailer_buf);
      free(st->trailer_buf);
    }
    if (st->mmio) {
      munmap(st->mmio, AGC_GC_MMIO_SIZE);
    }
    if (st->fd >= 0) {
      close(st->fd);
    }
    free(st);
    device->state = NULL;
  }
  return PAI_OK;
}

static pai_status_t
pai_gc_reset(pai_gpu_device_t *device) {
  pai_ps5_gc_state_t *st = (pai_ps5_gc_state_t *)device->state;
  uint32_t query = 0;

  if (!st) {
    return PAI_ERR_INVALID_ARG;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "gc reset: reopening device\n");

  if (st->mmio) {
    munmap(st->mmio, AGC_GC_MMIO_SIZE);
    st->mmio = NULL;
  }
  if (st->fd >= 0) {
    close(st->fd);
  }

  st->fd = open(GC_DEVICE_PATH, O_RDWR);
  if (st->fd < 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "gc reset: open failed (errno %d)\n", errno);
    return PAI_ERR_IO;
  }

  if (ioctl(st->fd, AGC_GC_IOCTL_CONTEXT_QUERY, &query) == 0) {
    st->ctx_caps = query;
  }

  if ((st->ctx_caps & 0xFFFFu) == 0) {
    void *mmio = mmap((void *)AGC_GC_MMIO_BASE, AGC_GC_MMIO_SIZE,
                      AGC_GC_MMIO_PROT, MAP_SHARED, st->fd, 0);
    if (mmio != MAP_FAILED) {
      st->mmio = mmio;
      sceKernelSetVirtualRangeName(mmio, AGC_GC_MMIO_SIZE, "SceGnmDingDong");
    }
  }

  return PAI_OK;
}

const pai_gpu_backend_ops_t pai_gpu_ops_ps5_gc = {
    .name = "ps5-gc (/dev/gc PM4)",
    .init = pai_gc_init,
    .shutdown = pai_gc_shutdown,
    .buffer_alloc = pai_gc_buffer_alloc,
    .buffer_free = pai_gc_buffer_free,
    .submit = pai_gc_submit_stream,
    .wait_label = pai_gc_wait_label,
    .reset = pai_gc_reset,
};

#endif /* PAI_PS5 */
