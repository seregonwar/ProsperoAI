/*
 * ProsperoAI — host reference GPU backend.
 *
 * PM4 interpreter running the exact streams the PS5 backend submits,
 * over identity-mapped host buffers: develops and validates the whole
 * submission path without hardware.
 */

#include <hal/hal.h>
#include <pm4/pm4.h>

#include "host_kernels.h"

#include <pai/log.h>

#include <stdlib.h>
#include <string.h>

#define PAI_HOST_REG_SH_BASE 0x0200
#define PAI_HOST_REG_SH_COUNT 0x0100

#define PAI_HOST_MAX_IB_DEPTH 4

typedef struct pai_host_shader_entry {
  uint64_t code_addr;
  pai_host_kernel_fn fn;
  void *ctx;
  struct pai_host_shader_entry *next;
} pai_host_shader_entry_t;

typedef struct pai_host_ref_state {
  uint32_t sh_regs[PAI_HOST_REG_SH_COUNT];
  pai_host_shader_entry_t *shaders;
} pai_host_ref_state_t;

pai_status_t
pai_gpu_host_register_shader(pai_gpu_device_t *device, uint64_t code_addr,
                             pai_host_kernel_fn fn, void *ctx) {
  pai_host_ref_state_t *st;
  pai_host_shader_entry_t *entry;
  pai_host_shader_entry_t **link;

  if (!device || device->backend != PAI_GPU_BACKEND_HOST_REF || !fn) {
    return PAI_ERR_INVALID_ARG;
  }

  st = (pai_host_ref_state_t *)device->state;

  /* Re-registering the same code address replaces the previous entry
   * (the harness reuses one code buffer across experiments). */
  link = &st->shaders;
  while (*link) {
    if ((*link)->code_addr == code_addr) {
      pai_host_shader_entry_t *old = *link;
      *link = old->next;
      free(old);
      break;
    }
    link = &(*link)->next;
  }

  entry = (pai_host_shader_entry_t *)calloc(1, sizeof(*entry));
  if (!entry) {
    return PAI_ERR_NOMEM;
  }

  entry->code_addr = code_addr;
  entry->fn = fn;
  entry->ctx = ctx;
  entry->next = st->shaders;
  st->shaders = entry;
  return PAI_OK;
}

static pai_status_t
pai_host_run_stream(pai_host_ref_state_t *st, const uint32_t *pm4,
                    uint32_t dwords, uint32_t depth) {
  uint32_t i = 0;

  if (depth > PAI_HOST_MAX_IB_DEPTH) {
    return PAI_ERR_INTERNAL;
  }

  while (i < dwords) {
    uint32_t header = pm4[i];
    uint32_t op = (header >> 8) & 0xFFu;
    uint32_t count;

    if ((header >> 30) != 3u) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "host-ref: not a type-3 packet (h=0x%08x)\n",
                     header);
      return PAI_ERR_UNSUPPORTED;
    }

    count = ((header >> 16) & 0x3FFFu) + 2;
    if (count < 1 || i + count > dwords) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "host-ref: malformed packet (h=0x%08x)\n",
                     header);
      return PAI_ERR_INTERNAL;
    }

    switch (op) {
    case PAI_PM4_OP_NOP:
      break;

    case PAI_PM4_OP_CONTEXT_CONTROL:
      /* Shadow-enable hint; the host interpreter has no shadow state. */
      break;

    case PAI_PM4_OP_SET_SH_REG: {
      uint32_t reg = pm4[i + 1] & 0xFFFFu;
      uint32_t n = count - 2;
      if (reg < PAI_HOST_REG_SH_BASE ||
          reg + n > PAI_HOST_REG_SH_BASE + PAI_HOST_REG_SH_COUNT) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "host-ref: SET_SH_REG out of range "
                                     "(0x%04x + %u)\n", reg, n);
        return PAI_ERR_INTERNAL;
      }
      memcpy(&st->sh_regs[reg - PAI_HOST_REG_SH_BASE], &pm4[i + 2],
             n * sizeof(uint32_t));
      break;
    }

    case PAI_PM4_OP_DISPATCH_DIRECT: {
      uint64_t code_addr;
      const uint32_t *ud;
      pai_host_shader_entry_t *e;
      uint32_t pgm_lo =
          st->sh_regs[PAI_REG_COMPUTE_PGM_LO - PAI_HOST_REG_SH_BASE];
      uint32_t pgm_hi =
          st->sh_regs[PAI_REG_COMPUTE_PGM_HI - PAI_HOST_REG_SH_BASE];
      uint32_t threads_x =
          st->sh_regs[PAI_REG_COMPUTE_NUM_THREAD_X - PAI_HOST_REG_SH_BASE];
      uint32_t group_x = pm4[i + 1];

      code_addr = ((uint64_t)pgm_lo << 8) | ((uint64_t)pgm_hi << 40);
      ud = &st->sh_regs[PAI_REG_COMPUTE_USER_DATA_0 - PAI_HOST_REG_SH_BASE];

      for (e = st->shaders; e; e = e->next) {
        if (e->code_addr == code_addr) {
          pai_status_t r = e->fn(e->ctx, ud, threads_x, group_x);
          if (r != PAI_OK) {
            PAI_LOG_ERROR_(PAI_SUB_GPU, "host-ref: kernel failed (%s)\n",
                           pai_status_str(r));
            return r;
          }
          break;
        }
      }
      if (!e) {
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "host-ref: no host kernel registered for code "
                       "addr 0x%llx\n",
                       (unsigned long long)code_addr);
        return PAI_ERR_UNSUPPORTED;
      }
      break;
    }

    case PAI_PM4_OP_RELEASE_MEM: {
      uint64_t addr;
      uint32_t data;

      if (count >= 8 && (pm4[i + 1] & 0xFFF000u) == 0x703000u) {
        /* Action-based EOP fence (OpenAGC runtime layout):
         * addr at [3..4], 32-bit value at [5]. */
        addr = (uint64_t)pm4[i + 3] | ((uint64_t)pm4[i + 4] << 32);
        data = pm4[i + 5];
      } else {
        /* Legacy SetEopFlip layout: addr at [2..3], value at [4]. */
        addr = (uint64_t)pm4[i + 2] | ((uint64_t)pm4[i + 3] << 32);
        data = pm4[i + 4];
      }
      *(volatile uint32_t *)(uintptr_t)addr = data;
      break;
    }

    case PAI_PM4_OP_WRITE_DATA: {
      uint64_t dst = (uint64_t)pm4[i + 2] | ((uint64_t)pm4[i + 3] << 32);
      memcpy((void *)(uintptr_t)dst, &pm4[i + 4],
             (count - 4) * sizeof(uint32_t));
      break;
    }

    case PAI_PM4_OP_DMA_DATA: {
      uint64_t src = (uint64_t)pm4[i + 2] | ((uint64_t)pm4[i + 3] << 32);
      uint64_t dst = (uint64_t)pm4[i + 4] | ((uint64_t)pm4[i + 5] << 32);
      uint32_t size = pm4[i + 6] & 0x1FFFFFu;
      memmove((void *)(uintptr_t)dst, (void *)(uintptr_t)src, size);
      break;
    }

    case PAI_PM4_OP_INDIRECT_BUFFER: {
      uint64_t ib = (uint64_t)pm4[i + 1] | ((uint64_t)pm4[i + 2] << 32);
      uint32_t ib_dwords = pm4[i + 3] & 0xFFFFFu;
      pai_status_t r = pai_host_run_stream(st, (const uint32_t *)(uintptr_t)ib,
                                           ib_dwords, depth + 1);
      if (r != PAI_OK) {
        return r;
      }
      break;
    }

    case PAI_PM4_OP_EVENT_WRITE:
      break;

    default:
      PAI_LOG_ERROR_(PAI_SUB_GPU, "host-ref: unsupported opcode 0x%02x\n",
                     op);
      return PAI_ERR_UNSUPPORTED;
    }

    i += count;
  }

  return PAI_OK;
}

static pai_status_t
pai_host_buffer_alloc(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer,
                      uint64_t size, uint32_t flags) {
  void *p;
  (void)device;

  if (size == 0) {
    return PAI_ERR_INVALID_ARG;
  }

#ifdef _WIN32
  p = _aligned_malloc((size_t)size, PAI_GPU_ALLOC_ALIGN);
#else
  if (posix_memalign(&p, PAI_GPU_ALLOC_ALIGN, (size_t)size) != 0) {
    p = NULL;
  }
#endif
  if (!p) {
    return PAI_ERR_NOMEM;
  }

  buffer->size = size;
  buffer->cpu_addr = p;
  buffer->gpu_addr = (uint64_t)(uintptr_t)p; /* identity mapping, like dmem */
  buffer->flags = flags | PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_CPU_COHERENT;
  return PAI_OK;
}

static void
pai_host_buffer_free(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer) {
  (void)device;
#ifdef _WIN32
  _aligned_free(buffer->cpu_addr);
#else
  free(buffer->cpu_addr);
#endif
  buffer->cpu_addr = NULL;
}

static pai_status_t
pai_host_submit(pai_gpu_device_t *device, const uint32_t *pm4,
                uint32_t dwords, uint32_t queue_type) {
  pai_host_ref_state_t *st = (pai_host_ref_state_t *)device->state;

  (void)queue_type;
  return pai_host_run_stream(st, pm4, dwords, 0);
}

static pai_status_t
pai_host_wait_label(pai_gpu_device_t *device, uint64_t label_addr,
                    uint32_t label_value, uint64_t timeout_ns) {
  /* The stream ran synchronously in submit(); the label was written by
   * the RELEASE_MEM handler inside the stream. */
  (void)device;
  (void)timeout_ns;
  if (label_addr != 0) {
    *(volatile uint32_t *)(uintptr_t)label_addr = label_value;
  }
  return PAI_OK;
}

static pai_status_t
pai_host_init(pai_gpu_device_t *device) {
  pai_host_ref_state_t *st = (pai_host_ref_state_t *)calloc(1, sizeof(*st));
  if (!st) {
    return PAI_ERR_NOMEM;
  }
  device->state = st;
  return PAI_OK;
}

static pai_status_t
pai_host_shutdown(pai_gpu_device_t *device) {
  pai_host_ref_state_t *st = (pai_host_ref_state_t *)device->state;
  pai_host_shader_entry_t *e;

  if (st) {
    e = st->shaders;
    while (e) {
      pai_host_shader_entry_t *next = e->next;
      free(e);
      e = next;
    }
    free(st);
    device->state = NULL;
  }
  return PAI_OK;
}

static void
pai_host_buffer_flush(pai_gpu_device_t *device, pai_gpu_buffer_t *buffer) {
  (void)device;
  (void)buffer;
}

static uint64_t
pai_host_aux_va(pai_gpu_device_t *device) {
  (void)device;
  return 0ULL;
}

static pai_status_t
pai_host_reset(pai_gpu_device_t *device) {
  pai_host_ref_state_t *st = (pai_host_ref_state_t *)device->state;
  memset(st->sh_regs, 0, sizeof(st->sh_regs));
  return PAI_OK;
}

const pai_gpu_backend_ops_t pai_gpu_ops_host_ref = {
    .name = "host-ref (PM4 interpreter)",
    .init = pai_host_init,
    .shutdown = pai_host_shutdown,
    .buffer_alloc = pai_host_buffer_alloc,
    .buffer_free = pai_host_buffer_free,
    .submit = pai_host_submit,
    .wait_label = pai_host_wait_label,
    .reset = pai_host_reset,
    .aux_va = pai_host_aux_va,
    .buffer_flush = pai_host_buffer_flush,
};
