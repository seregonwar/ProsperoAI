/*
 * ProsperoAI — PS5 GPU backend (raw /dev/gc submission).
 *
 * Phase 0 bring-up path, no GNM/AGC driver: GPU-visible memory via the
 * direct-memory syscalls, submission via /dev/gc ioctl nr=0x02
 * (0xC0108102, identical on 3.20-12.70 per OpenAGC RE), completion via
 * an EOP fence label the CPU polls.
 *
 * ABI (see notes/re/940-gc-ioctl.md):
 *   AgcGcSubmitArgs    { u32 queue_type; u32 num_cbs; u64 cb_array }
 *   AgcGcCommandBuffer { u64 header([63:32]=ib_lo,[31:0]=0xC0023F00);
 *                        u64 ib_base([63:32]=ib_size,[31:0]=ib_hi)  }
 *
 * In payload contexts the ring can defer the final descriptor, so every
 * submit carries a trailing 16-dword NOP IB (FW 5.50-proven sequence).
 */

#include <hal/hal.h>
#include <ps5/gvmspace.h>
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
#define AGC_GC_IOCTL_QUEUE_CREATE 0xC0408121u /* nr=0x21, RW, 64 bytes */

/* GPU register space (SPRX-confirmed): mapped on the gc fd when the
 * context query reports an uninitialized context (caps lower 16 == 0). */
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
extern int sceKernelMapNamedSystemFlexibleMemory(void **addr, size_t size,
                                                 int type, int flags,
                                                 const char *name);
extern int sceKernelReleaseFlexibleMemory(void *addr, size_t len);

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
  void *acqrb;
  void *eop_fifo;
  pai_gpu_buffer_t *cb_buf;
  pai_gpu_buffer_t *trailer_buf;
  uint32_t cb_words;
} pai_ps5_gc_state_t;

/* QUEUE_CREATE arg (OpenAGC driver_prospero.c, SPRX layout):
 * 0x00 magics/token, 0x10 pipe_id u64, 0x18 caller_arg, 0x20 mmio_base,
 * 0x28 queue_id/flags, 0x30 ring_addr, 0x38 ring_size. */
typedef struct {
  uint32_t magic1;
  uint32_t magic2;
  uint32_t magic3;
  uint32_t token;
  uint64_t pipe_id;
  uint64_t caller_arg;
  uint64_t mmio_base;
  uint32_t queue_id;
  uint32_t flags;
  uint64_t ring_addr;
  uint64_t ring_size;
} pai_gc_queue_create_arg_t;

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
pai_gc_alloc_dmem(pai_gpu_buffer_t *buffer, uint64_t size, const char *name,
                  uint64_t *out_phys) {
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
  if (out_phys) {
    *out_phys = (uint64_t)phys;
  }
  return PAI_OK;
}

static pai_status_t
pai_gc_buffer_alloc(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer,
                    uint64_t size, uint32_t flags) {
  pai_status_t st;
  uint64_t phys = 0;

  (void)device;
  (void)flags;

  /* Direct memory gives us the physical address, which the GPU
   * page-table repair needs. */
  st = pai_gc_alloc_dmem(buffer, size, "pai-gpu", &phys);
  if (st != PAI_OK) {
    return st;
  }

  /* READ-ONLY diagnosis: is the kernel's GPU mapping of this VA
   * pointing at the SAME physical page as the dmem syscall? A mismatch
   * explains the zero reads (the GPU loads another page). */
  {
    uint64_t pml4_phys = 0;
    intptr_t dmap = 0;
    static int probed = 0;
    if (!probed && pai_gvmspace_layout(&pml4_phys, &dmap) == 0) {
      probed = 1;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "dmem buf: va=0x%llx syscall phys=0x%llx\n",
                    (unsigned long long)buffer->gpu_addr,
                    (unsigned long long)phys);
      (void)pai_gvmspace_probe(pml4_phys, buffer->gpu_addr, dmap);
    }
  }

  /* Repair the PDE physical frame: the dmem syscall hands out the
   * GPU-BUS address (aperture +0x2000000000) while the GPU MMU walks
   * CPU physicals - the kernel's PDE points at a different page, which
   * is exactly why every shader load returns 0. Existing flags kept. */
  (void)pai_gvmspace_repair(buffer->gpu_addr, phys);

  return PAI_OK;
}

static void
pai_gc_buffer_free(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer) {
  (void)device;
  if (buffer->cpu_addr) {
    if (sceKernelReleaseFlexibleMemory(buffer->cpu_addr,
                                       (size_t)buffer->size) != 0) {
      munmap(buffer->cpu_addr, (size_t)buffer->size);
    }
    buffer->cpu_addr = NULL;
  }
}

/* Submit the PM4 cb plus the 16-dword NOP trailer as two descriptors.
 * queue_type: 3 = GFX ring (IB 0x3F); 0x8000000C = special compute
 * queue pipe 0xc with const-IB (0x33) descriptors. */
static pai_status_t
pai_gc_submit(pai_ps5_gc_state_t *st, uint32_t cb_words, uint32_t queue_type) {
  pai_gc_cb_descriptor_t descs[2];
  pai_gc_submit_args_t args;
  uint64_t cb_addr = st->cb_buf->gpu_addr;
  uint64_t trailer_addr = st->trailer_buf->gpu_addr;
  uint32_t ib_header;
  int r;

  if ((queue_type & 0x80000000u) != 0) {
    queue_type &= 0x7FFFFFFFu;
    ib_header = 0xC0023300u; /* IT_INDIRECT_BUFFER_CNST (compute queue) */
  } else {
    ib_header = 0xC0023F00u; /* IT_INDIRECT_BUFFER (GFX ring) */
  }

  /* Descriptor layout: IB header, base, size. The kernel inserts the
   * VMID into ib_base[63:52] after copyin. */
  descs[0].header = ((cb_addr & 0xFFFFFFFFu) << 32) | ib_header;
  descs[0].ib_base = ((uint64_t)cb_words << 32) | ((cb_addr >> 32) & 0xFFFFu);
  descs[1].header = ((trailer_addr & 0xFFFFFFFFu) << 32) | ib_header;
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

  /* Poll the label; direct memory is GPU-coherent, so the CPU observes
   * the write once the EOP event completed. */
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

  /* SPRX-confirmed: when the context is not yet initialized (caps lower
   * 16 == 0), the driver maps the GPU register space on the gc fd at
   * 0xFE0200000. Without this, compute dispatches never run. */
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

  /* Driver-style context bring-up: ACQRB + EOP FIFO flexible-memory
   * regions (allocation only — SAFE, no ring). The queue creation is
   * disabled: the kernel wires the ring to the GPU compute engine,
   * which executes garbage and kills the UI. */
  if (sceKernelMapNamedSystemFlexibleMemory(&st->acqrb, 0x1E0000, 0x33, 0,
                                            "SceGnmACQRB") != 0 ||
      sceKernelMapNamedSystemFlexibleMemory(&st->eop_fifo, 0x3C000, 0x33, 0,
                                            "SceGnmEopFifo") != 0) {
    PAI_LOG_WARN_(PAI_SUB_GPU,
                  "gc internal memory allocation failed; shader loads may "
                  "not work\n");
  } else {
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "gc internal memory ready (acqrb=0x%llx eop=0x%llx)\n",
                  (unsigned long long)(uintptr_t)st->acqrb,
                  (unsigned long long)(uintptr_t)st->eop_fifo);
  }

  /* Read-only layout diagnostic for the GPU page-table work. */
  (void)pai_gvmspace_diag();

  /* Probe the LIVE GPU pml4 (from the diag) against the acqrb VA — the
   * kernel's own GPU mapping must be present there. Read-only. */
  {
    uint64_t pml4_phys = 0;
    intptr_t dmap = 0;
    uint64_t probe_va = st->acqrb
                            ? (uint64_t)(uintptr_t)st->acqrb
                            : 0x200400000ULL;
    if (pai_gvmspace_layout(&pml4_phys, &dmap) == 0) {
      (void)pai_gvmspace_probe(pml4_phys, probe_va, dmap);
    } else {
      PAI_LOG_WARN_(PAI_SUB_GPU, "gvm probe: no layout available\n");
    }
  }

  st->cb_buf = (pai_gpu_buffer_t *)calloc(1, sizeof(*st->cb_buf));
  if (!st->cb_buf ||
      pai_gc_alloc_dmem(st->cb_buf, PAI_GC_CB_BUF_SIZE, "pai-cb", NULL) != PAI_OK) {
    return PAI_ERR_NOMEM;
  }

  /* 16-dword NOP trailer: forces the ring to run the final descriptor. */
  trailer = (pai_gpu_buffer_t *)calloc(1, sizeof(*trailer));
  if (!trailer ||
      pai_gc_alloc_dmem(trailer, PAI_GPU_ALLOC_ALIGN, "pai-trailer", NULL) !=
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
