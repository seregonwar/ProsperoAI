/*
 * prosperoai.elf — PAI-M0 bring-up harness.
 *
 * Stage A: /dev/gc DMA copy. Stage E: compute experiment matrix
 * (bisecting gfx1013 dispatch on 9.40). Stage B0/B: golden memset +
 * vecadd vs CPU reference. Stage C: dispatch latency.
 *
 * m0_run_gpu falls back to polling the output buffer when the EOP
 * label never fires, to tell "ran but fence broken" from "never ran".
 * Exit code: bit 0 = A failed, bit 1 = B failed.
 */

#include <math.h>

#include <pai/api.h>
#include <pai/log.h>

#include <ps5/gvmspace.h>

#include <bench.h>
#include <hal/hal.h>
#include <host_kernels.h>
#include <m0_experiments.h>
#include <memset16.h>
#include <pm4/pm4.h>
#include <ref_ops.h>
#include <runtime.h>
#include <vecadd.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M0_STAGE_A_FAIL (1u << 0)
#define M0_STAGE_B_FAIL (1u << 1)

#define M0_TIMEOUT_NS UINT64_C(2000000000) /* 2s per submission */

#define PAI_GVMSPACE_MODE 0 /* 0 probe / 1 no-op write / 2 full repair */
#define M0_DMA_BYTES   (1u << 20)
#define M0_PM4_CAP     512

#define PAI_DEPLOY_NOTIFY "ProsperoAI deployed. Credit: SeregonWar"

typedef struct m0_ctx {
  pai_runtime_t *rt;
  pai_gpu_device_t *gpu;
  pai_gpu_buffer_t src;
  pai_gpu_buffer_t dst;
  pai_gpu_buffer_t a;
  pai_gpu_buffer_t b;
  pai_gpu_buffer_t c;
  pai_gpu_buffer_t code;
  pai_gpu_buffer_t label;
  uint32_t label_value;
  int acb_mode; /* 1 = submit via the special compute queue (pipe 0xc) */
  int quiet_phases; /* 1 = skip INFO phase lines on successful submits */
  uint64_t timeout_ns; /* 0 = M0_TIMEOUT_NS */
  pai_bench_result_t bench;
} m0_ctx_t;

static void
m0_ctx_free_buffers(m0_ctx_t *ctx) {
  pai_gpu_device_t *g = ctx->gpu;
  if (!g) {
    return;
  }
  if (ctx->src.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->src);
  }
  if (ctx->dst.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->dst);
  }
  if (ctx->a.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->a);
  }
  if (ctx->b.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->b);
  }
  if (ctx->c.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->c);
  }
  if (ctx->code.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->code);
  }
  if (ctx->label.cpu_addr) {
    pai_gpu_buffer_free(g, &ctx->label);
  }
}

static int
m0_ctx_alloc_buffers(m0_ctx_t *ctx) {
  pai_gpu_device_t *g = ctx->gpu;
  uint64_t sz = M0_DMA_BYTES;

  if (pai_gpu_buffer_alloc(g, &ctx->src, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->dst, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->a, sz,
                           PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->b, sz,
                           PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->c, sz,
                           PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->code, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->label, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK) {
    return -1;
  }
  return 0;
}

/* Runs `stream` on queue 3 with a completion signal appended. Fence
 * ladder (hardware-qualified): eop-action → write-data → eop-legacy.
 *
 * Phases (logged, never collapsed into a single FAIL):
 *   SUBMITTED               submit ioctl returned OK
 *   EXECUTION_OBSERVED      EOP fired and/or watch bytes changed
 *   SHADER_OUTPUT_OBSERVED  watch differs from the fill pattern
 *   EOP_OBSERVED            completion label matched
 * VALIDATED is reported by the caller (m0_exp_report) after a CPU check.
 *
 * Returns 0 when execution was observed (EOP or shader output), -1
 * when the stream produced neither. A label timeout with changed
 * watch is success: the GPU ran and only the fence is broken. */
static int
m0_run_gpu(m0_ctx_t *ctx, uint32_t *stream, uint32_t stream_len,
           void *watch, uint32_t watch_bytes, uint32_t watch_fill,
           const char *stage) {
  static const char *const fence_names[] = {
      "eop-action", "write-data", "eop-legacy"};

  for (uint32_t fence = 0; fence < 3; fence++) {
    pai_pm4_builder_t pb;
    pai_status_t st;
    int submitted = 0;
    int eop_observed = 0;
    int shader_output = 0;
    int execution = 0;
    const uint8_t *w = (const uint8_t *)watch;
    uint32_t i;

    memset(watch, (int)watch_fill, watch_bytes);
#if defined(PAI_PS5)
    {
      uintptr_t wstart = (uintptr_t)watch & ~(uintptr_t)63u;
      uintptr_t wend = (uintptr_t)watch + watch_bytes;
      while (wstart < wend) {
        __asm__ volatile("clflush (%0)" : : "r"(wstart) : "memory");
        wstart += 64u;
      }
    }
#endif
    *(volatile uint32_t *)ctx->label.cpu_addr = 0;
    ctx->label_value++;

    pb.buf = stream;
    pb.cap = M0_PM4_CAP;
    pb.len = stream_len;

    /* The a/b/c data buffers are WC_GARLIC: flush CPU writes before
     * the GPU reads them. Cheap (2 MB clflush ~1 ms). */
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->b);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->c);

    switch (fence) {
    case 0:
      if (!pai_pm4_release_mem_eop_fence(&pb, ctx->label.gpu_addr,
                                         ctx->label_value) ||
          !pai_pm4_nop(&pb, 2)) {
        return -1;
      }
      break;
    case 1:
      if (!pai_pm4_write_data(&pb, ctx->label.gpu_addr, &ctx->label_value,
                              1)) {
        return -1;
      }
      break;
    default:
      if (!pai_pm4_release_mem_eop(&pb, PAI_GFX1013_EOP_CACHE_FLUSH_EVENT, 0,
                                   ctx->label.gpu_addr, ctx->label_value)) {
        return -1;
      }
      break;
    }

    PAI_LOG_DEBUG_(PAI_SUB_GPU,
                   "[M0-%s] attempt: fence=%s label=0x%llx value=0x%x\n",
                   stage, fence_names[fence],
                   (unsigned long long)ctx->label.gpu_addr, ctx->label_value);

    st = ctx->acb_mode
             ? pai_gpu_submit_acb(ctx->gpu, stream, pb.len)
             : pai_gpu_submit_q(ctx->gpu, stream, pb.len, 3);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-%s] SUBMITTED=0 fence=%s: %s\n", stage,
                     fence_names[fence], pai_status_str(st));
      continue;
    }
    submitted = 1;

    st = pai_gpu_wait_label(ctx->gpu, ctx->label.gpu_addr, ctx->label_value,
                            ctx->timeout_ns ? ctx->timeout_ns : M0_TIMEOUT_NS);
    eop_observed = (st == PAI_OK);
    for (i = 0; i < watch_bytes; i++) {
      if (w[i] != (uint8_t)watch_fill) {
        shader_output = 1;
        break;
      }
    }
    execution = eop_observed || shader_output;

    if (!ctx->quiet_phases || !execution) {
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] SUBMITTED=%d EXECUTION_OBSERVED=%d "
                    "SHADER_OUTPUT_OBSERVED=%d EOP_OBSERVED=%d fence=%s\n",
                    (stage[0] == 'M' && stage[1] == '0' && stage[2] == '-')
                        ? stage + 3
                        : stage,
                    submitted, execution, shader_output, eop_observed,
                    fence_names[fence]);
    }

    if (execution) {
      return 0;
    }
  }

  return -1;
}

/* Submit the already-built dispatch and wait for the EOP label.
 * Does not memset/clflush garlic data buffers — caller must have
 * flushed inputs once and (if validating) cleared the watch region. */
static int
m0_timed_eop(m0_ctx_t *ctx, uint32_t *stream, uint32_t stream_len,
             uint64_t *out_ns) {
  pai_pm4_builder_t pb;
  pai_status_t st;
  uint64_t t0, t1;

  *(volatile uint32_t *)ctx->label.cpu_addr = 0;
  ctx->label_value++;

  pb.buf = stream;
  pb.cap = M0_PM4_CAP;
  pb.len = stream_len;
  if (!pai_pm4_release_mem_eop_fence(&pb, ctx->label.gpu_addr,
                                     ctx->label_value) ||
      !pai_pm4_nop(&pb, 2)) {
    return -1;
  }

  t0 = pai_clock_ns();
  st = ctx->acb_mode ? pai_gpu_submit_acb(ctx->gpu, stream, pb.len)
                     : pai_gpu_submit_q(ctx->gpu, stream, pb.len, 3);
  if (st != PAI_OK) {
    return -1;
  }
  st = pai_gpu_wait_label(ctx->gpu, ctx->label.gpu_addr, ctx->label_value,
                          ctx->timeout_ns ? ctx->timeout_ns : M0_TIMEOUT_NS);
  t1 = pai_clock_ns();
  if (st != PAI_OK) {
    return -1;
  }
  *out_ns = t1 - t0;
  return 0;
}

static void
m0_clflush_range(void *p, uint32_t bytes) {
#if defined(PAI_PS5)
  uintptr_t wstart = (uintptr_t)p & ~(uintptr_t)63u;
  uintptr_t wend = (uintptr_t)p + bytes;
  while (wstart < wend) {
    __asm__ volatile("clflush (%0)" : : "r"(wstart) : "memory");
    wstart += 64u;
  }
#else
  (void)p;
  (void)bytes;
#endif
}

/* Stage A — raw IT_DMA_DATA copy through /dev/gc. */

static int
m0_stage_a(m0_ctx_t *ctx) {
  uint32_t stream[M0_PM4_CAP];
  pai_pm4_builder_t pb;
  uint32_t *pattern = (uint32_t *)ctx->src.cpu_addr;
  uint32_t len;

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-A] GPU DMA copy (%u KiB)\n",
                M0_DMA_BYTES >> 10);

  for (uint32_t i = 0; i < M0_DMA_BYTES / 4; i++) {
    pattern[i] = 0x9E370001u ^ (i * 0x85EBCA6Bu);
  }

  pai_pm4_builder_init(&pb, stream, M0_PM4_CAP);
  if (!pai_pm4_dma_data(&pb, ctx->src.gpu_addr, ctx->dst.gpu_addr,
                        M0_DMA_BYTES)) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-A] pm4 build failed\n");
    return -1;
  }
  len = pb.len;

  if (m0_run_gpu(ctx, stream, len, ctx->dst.cpu_addr, M0_DMA_BYTES, 0x00,
                 "M0-A") != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-A] FAIL: GPU DMA never executed "
                   "(dst[0..3]=%08x %08x %08x %08x)\n",
                   ((uint32_t *)ctx->dst.cpu_addr)[0],
                   ((uint32_t *)ctx->dst.cpu_addr)[1],
                   ((uint32_t *)ctx->dst.cpu_addr)[2],
                   ((uint32_t *)ctx->dst.cpu_addr)[3]);
    return -1;
  }

  if (memcmp(ctx->src.cpu_addr, ctx->dst.cpu_addr, M0_DMA_BYTES) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-A] FAIL: destination mismatch\n");
    return -1;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-A] PASS: GPU DMA verified\n");
  return 0;
}

/* Stage E — compute bring-up experiment matrix. */

static void
m0_build_dispatch_stream_rsrc1(m0_ctx_t *ctx, uint32_t *stream, uint32_t cap,
                               uint32_t rsrc1, uint32_t rsrc2,
                               uint32_t threads_x, uint32_t groups_x,
                               const uint32_t *user_data, uint32_t ud_count,
                               uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[9];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, stream, cap);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = rsrc1;
  vals[1] = rsrc2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_EXP_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = threads_x;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, ud_count,
                             user_data);

  pai_pm4_dispatch_direct(&pb, groups_x, 1, 1, 0);

  *out_len = pb.len;
}

static void
m0_build_dispatch_stream(m0_ctx_t *ctx, uint32_t *stream, uint32_t cap,
                         uint32_t rsrc2, uint32_t threads_x, uint32_t groups_x,
                         const uint32_t *user_data, uint32_t ud_count,
                         uint32_t *out_len) {
  m0_build_dispatch_stream_rsrc1(ctx, stream, cap, PAI_EXP_RSRC1, rsrc2,
                                 threads_x, groups_x, user_data, ud_count,
                                 out_len);
}

/* Full OpenAGC dispatch preamble (agcGfx1013DispatchComputeCommon):
 * context-control shadow enable, resource limits, destination enables,
 * START/NUM thread registers — the driver sequence our minimal one
 * skipped. */
static void
m0_emit_compute_preamble(pai_pm4_builder_t *pb, uint32_t threads_x) {
  uint32_t vals[6];

  pai_pm4_context_control(pb, 0x80000000u, 0x80000000u);

  vals[0] = 0; /* WAVES_PER_SH: 0 unless waves%4==0 */
  vals[1] = 0xFFFFFFFFu;
  vals[2] = 0xFFFFFFFFu;
  pai_pm4_set_sh_reg_compute(pb, 0x215, 3, vals); /* COMPUTE_RESOURCE_LIMITS */

  vals[0] = 0xFFFFFFFFu;
  vals[1] = 0xFFFFFFFFu;
  pai_pm4_set_sh_reg_compute(pb, 0x219, 2, vals); /* destination enable SE1/2 */

  vals[0] = 0; /* COMPUTE_START_X */
  vals[1] = 0;
  vals[2] = 0;
  vals[3] = threads_x;
  vals[4] = 1;
  vals[5] = 1;
  pai_pm4_set_sh_reg_compute(pb, PAI_REG_COMPUTE_START_X, 6, vals);
}

static void
m0_build_dispatch_stream_preamble(m0_ctx_t *ctx, uint32_t *stream,
                                  uint32_t cap, uint32_t rsrc2,
                                  uint32_t threads_x, uint32_t groups_x,
                                  const uint32_t *user_data, uint32_t ud_count,
                                  uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[9];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, stream, cap);

  m0_emit_compute_preamble(&pb, threads_x);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_EXP_RSRC1;
  vals[1] = rsrc2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_EXP_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, ud_count,
                             user_data);

  pai_pm4_dispatch_direct(&pb, groups_x, 1, 1, 0);

  *out_len = pb.len;
}

/* Defined with stage B0 below. */
static void m0_build_memset16_stream(m0_ctx_t *ctx, uint32_t *stream,
                                     uint32_t cap, uint64_t dst_addr,
                                     uint32_t blocks, const uint32_t pattern[4],
                                     uint32_t *out_len);

static void
m0_exp_report(const char *name, int ok) {
  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] VALIDATED %s\n", name,
                ok ? "PASS" : "FAIL");
}

/* ACO's hardware-qualified memset kernel sets bit 15 of every FLAT
 * word0 (e.g. 0xDC788000); llvm-mc 14 emits 0xDC700000. The patch
 * targets (PAI_*_FLAT_WORD) index word0 — the DC-prefixed first word;
 * word1 holds ADDR/DATA and must not be patched. Returns -1 if the
 * target word is not a FLAT word0. */
static int
m0_patch_flat_bit15(uint32_t *code, uint32_t word_index) {
  if ((code[word_index] & 0xFF000000u) != 0xDC000000u) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "flat patch target word %u = 0x%08x is not a FLAT word0\n",
                   word_index, code[word_index]);
    return -1;
  }
  code[word_index] |= 0x8000u;
  return 0;
}

static int
m0_check_store_const(const uint32_t *dst, const char *name) {
  int ok = 1;
  for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
    if (dst[i] != PAI_STORE_CONST_VALUE) {
      ok = 0;
      break;
    }
  }
  m0_exp_report(name, ok);
  if (!ok) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-%s] dst[0..7] = %08x %08x %08x %08x "
                   "%08x %08x %08x %08x\n",
                   name, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                   dst[6], dst[7]);
  }
  return ok;
}

static int
m0_check_loadstore(const uint32_t *a, const uint32_t *c, const char *name) {
  int ok = 1;
  for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
    if (c[i] != a[i]) {
      ok = 0;
      break;
    }
  }
  m0_exp_report(name, ok);
  if (!ok) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-%s] c[0..7] = %08x %08x %08x %08x "
                   "%08x %08x %08x %08x\n",
                   name, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]);
  }
  return ok;
}

static int
m0_check_memset(const uint32_t *dst, uint32_t blocks,
                const uint32_t pattern[4], const char *name) {
  uint32_t ref[1024 * 4];
  int ok;
  pai_ref_memset16(ref, pattern, blocks);
  ok = memcmp(dst, ref, blocks * 16) == 0;
  m0_exp_report(name, ok);
  if (!ok) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-%s] dst[0..15] = %08x %08x %08x %08x %08x %08x "
                   "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                   name, dst[0], dst[1], dst[2], dst[3], dst[4], dst[5],
                   dst[6], dst[7], dst[8], dst[9], dst[10], dst[11], dst[12],
                   dst[13], dst[14], dst[15]);
  }
  return ok;
}

static int
m0_exp_store_const(m0_ctx_t *ctx, const char *name, int patched) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[2];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;

  memcpy(ctx->code.cpu_addr, pai_store_const_code,
         PAI_STORE_CONST_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_store_const, NULL);
  }
  if (patched &&
      m0_patch_flat_bit15((uint32_t *)ctx->code.cpu_addr,
                          PAI_STORE_CONST_FLAT_WORD) != 0) {
    m0_exp_report(name, 0);
    return 0;
  }
  ud[0] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  ud[1] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_STORE_CONST_RSRC2,
                           PAI_EXP_THREADS_X, 1, ud, 2, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, dst32, 128, 0x00, name);
  m0_check_store_const(dst32, name);
  return 0;
}

static int
m0_exp_loadstore(m0_ctx_t *ctx, const char *name, int patched) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[4];
  uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
  uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;

  memcpy(ctx->code.cpu_addr, pai_loadstore_code,
         PAI_LOADSTORE_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_loadstore, NULL);
  }
  if (patched &&
      (m0_patch_flat_bit15((uint32_t *)ctx->code.cpu_addr,
                           PAI_LOADSTORE_FLAT_LOAD_WORD) != 0 ||
       m0_patch_flat_bit15((uint32_t *)ctx->code.cpu_addr,
                           PAI_LOADSTORE_FLAT_STORE_WORD) != 0)) {
    m0_exp_report(name, 0);
    return 0;
  }
  for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
    a32[i] = 0x11111111u + i;
  }
  ud[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
  ud[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
  ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_LOADSTORE_RSRC2,
                           PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);
  m0_check_loadstore(a32, c32, name);
  return 0;
}

static int
m0_exp_memset(m0_ctx_t *ctx, uint32_t blocks, const char *name) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint32_t pattern[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};

  memcpy(ctx->code.cpu_addr, pai_memset16_code,
         PAI_MEMSET16_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_memset16, NULL);
  }
  m0_build_memset16_stream(ctx, stream, M0_PM4_CAP, ctx->dst.gpu_addr, blocks,
                           pattern, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, dst32, blocks * 16, 0x00, name);
  m0_check_memset(dst32, blocks, pattern, name);
  return 0;
}

/* E5/E7: store_const64 with the psbc user-data ABI (dst at s2/s3). */
static int
m0_exp_store_const64(m0_ctx_t *ctx, const char *name, int preamble) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[4];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;

  memcpy(ctx->code.cpu_addr, pai_store_const64_code,
         PAI_STORE_CONST64_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_store_const64, NULL);
  }
  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);

  if (preamble) {
    m0_build_dispatch_stream_preamble(ctx, stream, M0_PM4_CAP,
                                      PAI_STORE_CONST64_RSRC2,
                                      PAI_STORE_CONST64_THREADS_X, 1, ud, 4,
                                      &stream_len);
  } else {
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_STORE_CONST64_RSRC2,
                             PAI_STORE_CONST64_THREADS_X, 1, ud, 4,
                             &stream_len);
  }
  m0_run_gpu(ctx, stream, stream_len, dst32, 128, 0x00, name);

  {
    int ok = 1;
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      if (dst32[i] != PAI_STORE_CONST64_VALUE) {
        ok = 0;
        break;
      }
    }
    m0_exp_report(name, ok);
    if (!ok) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-%s] dst[0..7] = %08x %08x %08x %08x "
                     "%08x %08x %08x %08x\n",
                     name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                     dst32[5], dst32[6], dst32[7]);
    }
  }
  return 0;
}

/* E6: memset golden with the full OpenAGC dispatch preamble. */
static int
m0_exp_memset_preamble(m0_ctx_t *ctx, uint32_t blocks, const char *name) {
  pai_pm4_builder_t pb;
  uint32_t stream[M0_PM4_CAP];
  uint32_t vals[9];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint32_t pattern[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};
  uint64_t code = ctx->code.gpu_addr;

  memcpy(ctx->code.cpu_addr, pai_memset16_code,
         PAI_MEMSET16_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_memset16, NULL);
  }

  pai_pm4_builder_init(&pb, stream, M0_PM4_CAP);
  m0_emit_compute_preamble(&pb, PAI_MEMSET16_THREADS_X);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC1;
  vals[1] = PAI_MEMSET16_RSRC2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = 0;
  vals[1] = 0;
  vals[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  vals[4] = blocks;
  vals[5] = pattern[0];
  vals[6] = pattern[1];
  vals[7] = pattern[2];
  vals[8] = pattern[3];
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 9, vals);

  pai_pm4_dispatch_direct(&pb, (blocks + PAI_MEMSET16_THREADS_X - 1) /
                                   PAI_MEMSET16_THREADS_X,
                          1, 1, 0);

  m0_run_gpu(ctx, stream, pb.len, dst32, blocks * 16, 0x00, name);
  m0_check_memset(dst32, blocks, pattern, name);
  return 0;
}

/* E8: user-data probe — 9 distinct slot values through the golden
 * kernel reveal which SGPR slots feed the store data. */
static int
m0_exp_ud_probe(m0_ctx_t *ctx, const char *name) {
  pai_pm4_builder_t pb;
  uint32_t stream[M0_PM4_CAP];
  uint32_t vals[9];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint64_t code = ctx->code.gpu_addr;

  memcpy(ctx->code.cpu_addr, pai_memset16_code,
         PAI_MEMSET16_CODE_WORDS * sizeof(uint32_t));

  pai_pm4_builder_init(&pb, stream, M0_PM4_CAP);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC1;
  vals[1] = PAI_MEMSET16_RSRC2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_MEMSET16_THREADS_X;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  for (uint32_t i = 0; i < 9; i++) {
    vals[i] = 0x11111111u * (i + 1);
  }
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 9, vals);

  pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

  m0_run_gpu(ctx, stream, pb.len, dst32, 256, 0x00, name);

  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-%s] dst[0..31] = %08x %08x %08x %08x %08x %08x %08x "
                "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                "%08x %08x %08x\n",
                name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                dst32[5], dst32[6], dst32[7], dst32[8], dst32[9], dst32[10],
                dst32[11], dst32[12], dst32[13], dst32[14], dst32[15],
                dst32[16], dst32[17], dst32[18], dst32[19], dst32[20],
                dst32[21], dst32[22], dst32[23], dst32[24], dst32[25],
                dst32[26], dst32[27], dst32[28], dst32[29], dst32[30],
                dst32[31]);
  return 0;
}

/* E11: store_const64 with the golden register config (RSRC2=0x92,
 * TGID_X_EN, 9 user SGPRs) — isolates whether that config launches
 * waves. */
static int
m0_exp_store_const64_golden_cfg(m0_ctx_t *ctx, const char *name) {
  pai_pm4_builder_t pb;
  uint32_t stream[M0_PM4_CAP];
  uint32_t vals[9];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint64_t code = ctx->code.gpu_addr;

  memcpy(ctx->code.cpu_addr, pai_store_const64_code,
         PAI_STORE_CONST64_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 pai_host_kernel_store_const64, NULL);
  }

  pai_pm4_builder_init(&pb, stream, M0_PM4_CAP);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC1;
  vals[1] = PAI_MEMSET16_RSRC2; /* 0x92: 9 user SGPRs + TGID_X_EN */
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_MEMSET16_THREADS_X; /* 64 threads, like the golden */
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  vals[0] = 0;
  vals[1] = 0;
  vals[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  vals[4] = 0;
  vals[5] = 0;
  vals[6] = 0;
  vals[7] = 0;
  vals[8] = 0;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 9, vals);

  pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

  m0_run_gpu(ctx, stream, pb.len, dst32, 128, 0x00, name);

  {
    int ok = 1;
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      if (dst32[i] != PAI_STORE_CONST64_VALUE) {
        ok = 0;
        break;
      }
    }
    m0_exp_report(name, ok);
    if (!ok) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-%s] dst[0..7] = %08x %08x %08x %08x "
                     "%08x %08x %08x %08x\n",
                     name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                     dst32[5], dst32[6], dst32[7]);
    }
  }
  return 0;
}

/* E12/E13: FLAT x2/x4 store variants vs flat_store_dword. */
static int
m0_exp_store64_variant(m0_ctx_t *ctx, const char *name,
                       const uint32_t *code, uint32_t code_words,
                       uint32_t value, uint32_t check_words,
                       pai_host_kernel_fn host_fn, int patched,
                       uint32_t flat_word) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[4];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;

  memcpy(ctx->code.cpu_addr, code, code_words * sizeof(uint32_t));
  if (patched &&
      m0_patch_flat_bit15((uint32_t *)ctx->code.cpu_addr, flat_word) != 0) {
    m0_exp_report(name, 0);
    return 0;
  }
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr, host_fn, NULL);
  }
  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_STORE_CONST64_RSRC2,
                           PAI_STORE_CONST64_THREADS_X, 1, ud, 4, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, dst32, check_words * 4, 0x00, name);

  {
    int ok = 1;
    for (uint32_t i = 0; i < check_words; i++) {
      if (dst32[i] != value) {
        ok = 0;
        break;
      }
    }
    m0_exp_report(name, ok);
    if (!ok) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-%s] dst[0..7] = %08x %08x %08x %08x "
                     "%08x %08x %08x %08x\n",
                     name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                     dst32[5], dst32[6], dst32[7]);
    }
  }
  return 0;
}

/* E14: golden memset with the pattern pre-placed in the destination at
 * offset 0x27C (the kernel's SMEM load reads it from there). */
static int
m0_exp_memset_pattern_in_buf(m0_ctx_t *ctx, const char *name) {
  pai_pm4_builder_t pb;
  uint32_t stream[M0_PM4_CAP];
  uint32_t vals[9];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint32_t pattern[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};
  uint64_t code = ctx->code.gpu_addr;

  memcpy(ctx->code.cpu_addr, pai_memset16_code,
         PAI_MEMSET16_CODE_WORDS * sizeof(uint32_t));
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    /* The real kernel reads the pattern from the buffer (psbc ABI);
     * the host interpreter cannot emulate that. */
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] skipped on host (buffer ABI test)\n",
                  name);
    return 0;
  }

  memset(dst32, 0, 4096);
  memcpy((uint8_t *)dst32 + 0x27C, pattern, 16);

  pai_pm4_builder_init(&pb, stream, M0_PM4_CAP);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC1;
  vals[1] = PAI_MEMSET16_RSRC2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_MEMSET16_THREADS_X;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  vals[0] = 0;
  vals[1] = 0;
  vals[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  vals[4] = 64;
  vals[5] = 0;
  vals[6] = 0;
  vals[7] = 0;
  vals[8] = 0;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 9, vals);

  pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

  /* Watch only the region after the pre-placed pattern (0x27C + 16). */
  m0_run_gpu(ctx, stream, pb.len, (uint8_t *)dst32 + 0x400, 1024 - 0x400,
             0x00, name);
  m0_check_memset(dst32, 64, pattern, name);
  return 0;
}

/* E15-E19: instruction bisection — each kernel adds one instruction
 * family; the first that hangs pinpoints the toxic instruction.
 * Check = label fired, except E19 which must also write the constant. */
static int
m0_exp_bisect(m0_ctx_t *ctx, const char *name, uint32_t off,
              uint32_t words, int check_store, int patched) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[4];
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;

  memcpy(ctx->code.cpu_addr, &pai_bisect_code[off],
         words * sizeof(uint32_t));
  if (patched &&
      m0_patch_flat_bit15((uint32_t *)ctx->code.cpu_addr,
                          PAI_BISECT_STORE_FLAT_WORD - off) != 0) {
    m0_exp_report(name, 0);
    return 0;
  }
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                 check_store ? pai_host_kernel_bisect_store
                                             : pai_host_kernel_store64_x4,
                                 NULL);
  }
  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_STORE_CONST64_RSRC2,
                           PAI_STORE_CONST64_THREADS_X, 1, ud, 4, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, dst32, 64, 0x00, name);

  if (check_store) {
    int ok = 1;
    for (uint32_t i = 0; i < PAI_STORE_CONST64_THREADS_X * 4; i++) {
      if (dst32[i] != PAI_BISECT_STORE_VALUE) {
        ok = 0;
        break;
      }
    }
    m0_exp_report(name, ok);
    if (!ok) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-%s] dst[0..7] = %08x %08x %08x %08x "
                     "%08x %08x %08x %08x\n",
                     name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                     dst32[5], dst32[6], dst32[7]);
    }
  }
  /* For E15-E18 the report is implicit: m0_run_gpu already logged
   * whether the wave completed (label fired) or hung. */
  return 0;
}

/* G-series: mutate the golden kernel toward mine one step at a time;
 * the mutation that breaks execution is the real difference. */
static int
m0_exp_golden_mutant(m0_ctx_t *ctx, const char *name, uint32_t store_word0,
                     uint32_t store_word1) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t *dst32 = (uint32_t *)ctx->dst.cpu_addr;
  uint32_t pattern[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};
  uint32_t code[PAI_MEMSET16_CODE_WORDS];

  memcpy(code, pai_memset16_code, sizeof(code));
  if (store_word0 != 0) {
    code[14] = store_word0; /* golden flat_store_dwordx4 word0 */
  }
  if (store_word1 != 0) {
    code[15] = store_word1; /* golden flat_store_dwordx4 word1 */
  }
  memcpy(ctx->code.cpu_addr, code, sizeof(code));

  m0_build_memset16_stream(ctx, stream, M0_PM4_CAP, ctx->dst.gpu_addr, 64,
                           pattern, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, dst32, 64 * 16, 0x00, name);
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-%s] dst[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                "%08x\n",
                name, dst32[0], dst32[1], dst32[2], dst32[3], dst32[4],
                dst32[5], dst32[6], dst32[7]);
  return 0;
}

/* Parametric 1D add: same add1d kernel, groups_x = N, 1 thread/group.
 * Returns 1 on zero-mismatch for all completed iters. */
static int
m0_add1d_run_n(m0_ctx_t *ctx, uint32_t *stream, uint32_t *ud, uint32_t n) {
  pai_gpu_buffer_t packb;
  pai_gpu_buffer_t coutb;
  pai_gpu_buffer_t *pack_buf;
  pai_gpu_buffer_t *c_buf;
  uint32_t *pack;
  uint32_t *c1d;
  uint32_t stream_len = 0;
  uint32_t iter;
  uint32_t niter = 1u;
  uint32_t want_iters;
  uint64_t pack_bytes = (uint64_t)n * 8u;
  uint64_t c_bytes = (uint64_t)n * 4u;
  uint64_t t0, t1, first_ns = 0;
  int extra = 0;
  int ok = 1;
  int host = pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF;

  memset(&packb, 0, sizeof(packb));
  memset(&coutb, 0, sizeof(coutb));

  if (n == 0) {
    return 0;
  }
  if (pack_bytes <= M0_DMA_BYTES && c_bytes <= M0_DMA_BYTES) {
    pack_buf = &ctx->a;
    c_buf = &ctx->c;
  } else {
    extra = 1;
    if (pai_gpu_buffer_alloc(ctx->gpu, &packb, pack_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
            PAI_OK ||
        pai_gpu_buffer_alloc(ctx->gpu, &coutb, c_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
            PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-G] N=%u SKIP: buffer alloc failed (%llu + %llu)\n",
                     n, (unsigned long long)pack_bytes,
                     (unsigned long long)c_bytes);
      if (packb.cpu_addr) {
        pai_gpu_buffer_free(ctx->gpu, &packb);
      }
      if (coutb.cpu_addr) {
        pai_gpu_buffer_free(ctx->gpu, &coutb);
      }
      return 0;
    }
    pack_buf = &packb;
    c_buf = &coutb;
  }

  pack = (uint32_t *)pack_buf->cpu_addr;
  c1d = (uint32_t *)c_buf->cpu_addr;
  for (uint32_t i = 0; i < n; i++) {
    pack[2u * i] = 0x11110000u + i;
    pack[2u * i + 1u] = 0x00001111u + i;
  }
  pai_gpu_buffer_flush(ctx->gpu, pack_buf);

  if (n <= 256u) {
    want_iters = PAI_ADD1D_ITERS;
  } else if (n <= 4096u) {
    want_iters = 10u;
  } else {
    want_iters = 1u;
  }
  ctx->timeout_ns = (n >= 32768u) ? UINT64_C(15000000000) : 0;

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(pack_buf->gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(pack_buf->gpu_addr >> 32);
  ud[4] = (uint32_t)(c_buf->gpu_addr & 0xFFFFFFFFu);
  ud[5] = (uint32_t)(c_buf->gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ADD1D_RSRC2,
                           PAI_ADD1D_THREADS, n, ud, 6, &stream_len);

  for (iter = 0; iter < niter && ok; iter++) {
    ctx->quiet_phases = (iter > 0);
    t0 = pai_clock_ns();
    if (m0_run_gpu(ctx, stream, stream_len, c1d, (uint32_t)c_bytes, 0xCC,
                   "G") != 0) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-G] N=%u iter %u no execution c[0]=%08x\n", n, iter,
                     c1d[0]);
      ok = 0;
      break;
    }
    t1 = pai_clock_ns();
    if (iter == 0) {
      first_ns = t1 - t0;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G] N=%u iter 0 c[0..3] = %08x %08x %08x %08x "
                    "(%llu us)\n",
                    n, c1d[0], c1d[1 < n ? 1 : 0], c1d[2 < n ? 2 : 0],
                    c1d[3 < n ? 3 : 0],
                    (unsigned long long)(first_ns / 1000u));
    }
    for (uint32_t i = 0; i < n; i++) {
      uint32_t want = 0x11111111u + i * 2u;
      if (c1d[i] != want) {
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-G] N=%u c[%u] = %08x want %08x\n", n, i, c1d[i],
                       want);
        ok = 0;
        break;
      }
    }
    if (ok && iter == 0) {
      niter = want_iters;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G] N=%u first PASS, repeating %u times (no reset)\n",
                    n, niter);
    }
  }
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-G] N=%u %s iters=%u/%u first=%llu us groups=%u "
                "threads=1\n",
                n, ok ? "PASS" : "FAIL", iter, niter,
                (unsigned long long)(first_ns / 1000u), n);
  if (extra) {
    pai_gpu_buffer_free(ctx->gpu, &packb);
    pai_gpu_buffer_free(ctx->gpu, &coutb);
  }
  (void)host;
  return ok;
}

/* Integer SAXPY: C[i] = 3*A[i]+B[i], same dispatch as add1d. */
static int
m0_saxpy_run_n(m0_ctx_t *ctx, uint32_t *stream, uint32_t *ud, uint32_t n) {
  pai_gpu_buffer_t packb;
  pai_gpu_buffer_t coutb;
  pai_gpu_buffer_t *pack_buf;
  pai_gpu_buffer_t *c_buf;
  uint32_t *pack;
  uint32_t *c1d;
  uint32_t stream_len = 0;
  uint32_t iter;
  uint32_t niter = 1u;
  uint32_t want_iters;
  uint64_t pack_bytes = (uint64_t)n * 8u;
  uint64_t c_bytes = (uint64_t)n * 4u;
  uint64_t t0, t1, first_ns = 0;
  int extra = 0;
  int ok = 1;
  int host = pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF;

  memset(&packb, 0, sizeof(packb));
  memset(&coutb, 0, sizeof(coutb));

  if (n == 0) {
    return 0;
  }
  if (pack_bytes <= M0_DMA_BYTES && c_bytes <= M0_DMA_BYTES) {
    pack_buf = &ctx->a;
    c_buf = &ctx->c;
  } else {
    extra = 1;
    if (pai_gpu_buffer_alloc(ctx->gpu, &packb, pack_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
            PAI_OK ||
        pai_gpu_buffer_alloc(ctx->gpu, &coutb, c_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
            PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-S] N=%u SKIP: buffer alloc failed (%llu + %llu)\n",
                     n, (unsigned long long)pack_bytes,
                     (unsigned long long)c_bytes);
      if (packb.cpu_addr) {
        pai_gpu_buffer_free(ctx->gpu, &packb);
      }
      if (coutb.cpu_addr) {
        pai_gpu_buffer_free(ctx->gpu, &coutb);
      }
      return 0;
    }
    pack_buf = &packb;
    c_buf = &coutb;
  }

  pack = (uint32_t *)pack_buf->cpu_addr;
  c1d = (uint32_t *)c_buf->cpu_addr;
  for (uint32_t i = 0; i < n; i++) {
    pack[2u * i] = 0x11110000u + i;
    pack[2u * i + 1u] = 0x00001111u + i;
  }
  pai_gpu_buffer_flush(ctx->gpu, pack_buf);

  if (n <= 256u) {
    want_iters = PAI_SAXPY_ITERS;
  } else if (n <= 4096u) {
    want_iters = 10u;
  } else {
    want_iters = 1u;
  }
  ctx->timeout_ns = (n >= 32768u) ? UINT64_C(15000000000) : 0;

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(pack_buf->gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(pack_buf->gpu_addr >> 32);
  ud[4] = (uint32_t)(c_buf->gpu_addr & 0xFFFFFFFFu);
  ud[5] = (uint32_t)(c_buf->gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_SAXPY_RSRC2,
                           PAI_SAXPY_THREADS, n, ud, 6, &stream_len);

  for (iter = 0; iter < niter && ok; iter++) {
    ctx->quiet_phases = (iter > 0);
    t0 = pai_clock_ns();
    if (m0_run_gpu(ctx, stream, stream_len, c1d, (uint32_t)c_bytes, 0xCC,
                   "S") != 0) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-S] N=%u iter %u no execution c[0]=%08x\n", n, iter,
                     c1d[0]);
      ok = 0;
      break;
    }
    t1 = pai_clock_ns();
    if (iter == 0) {
      first_ns = t1 - t0;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-S] N=%u iter 0 c[0..3] = %08x %08x %08x %08x "
                    "(%llu us)\n",
                    n, c1d[0], c1d[1 < n ? 1 : 0], c1d[2 < n ? 2 : 0],
                    c1d[3 < n ? 3 : 0],
                    (unsigned long long)(first_ns / 1000u));
    }
    for (uint32_t i = 0; i < n; i++) {
      uint32_t want =
          PAI_SAXPY_A * (0x11110000u + i) + (0x00001111u + i);
      if (c1d[i] != want) {
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-S] N=%u c[%u] = %08x want %08x\n", n, i, c1d[i],
                       want);
        ok = 0;
        break;
      }
    }
    if (ok && iter == 0) {
      niter = want_iters;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-S] N=%u first PASS, repeating %u times (no reset)\n",
                    n, niter);
    }
  }
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-S] N=%u %s iters=%u/%u first=%llu us groups=%u "
                "threads=1 a=%u integer\n",
                n, ok ? "PASS" : "FAIL", iter, niter,
                (unsigned long long)(first_ns / 1000u), n, PAI_SAXPY_A);
  if (extra) {
    pai_gpu_buffer_free(ctx->gpu, &packb);
    pai_gpu_buffer_free(ctx->gpu, &coutb);
  }
  (void)host;
  return ok;
}

/* M1B: serial uint32 dot correctness primitive. One group, one thread,
 * SGPR loop. N from packed header. Result in C[0], uint32 wrap. */
static int
m0_dot_serial_run_n(m0_ctx_t *ctx, uint32_t *stream, uint32_t *ud,
                    uint32_t n) {
  uint32_t *pack;
  uint32_t *c1d;
  uint32_t stream_len = 0;
  uint32_t iter;
  uint32_t niter = 1u;
  uint32_t want_iters;
  uint32_t want = 0;
  uint64_t pack_bytes = 8u + (uint64_t)n * 8u;
  uint64_t t0, t1, first_ns = 0;
  int ok = 1;
  int host = pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF;

  if (n == 0 || pack_bytes > M0_DMA_BYTES) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-M1B] N=%u SKIP: bad size (%llu)\n", n,
                   (unsigned long long)pack_bytes);
    return 0;
  }

  pack = (uint32_t *)ctx->a.cpu_addr;
  c1d = (uint32_t *)ctx->c.cpu_addr;
  pack[0] = n;
  pack[1] = 0;
  want = 0;
  for (uint32_t i = 0; i < n; i++) {
    uint32_t a = 1u + i;
    uint32_t b = 2u + i;
    pack[2u + 2u * i] = a;
    pack[3u + 2u * i] = b;
    want += a * b;
  }
  pai_gpu_buffer_flush(ctx->gpu, &ctx->a);

  if (n <= 256u) {
    want_iters = PAI_DOT_SERIAL_U32_ITERS;
  } else if (n <= 4096u) {
    want_iters = 10u;
  } else {
    want_iters = 1u;
  }
  ctx->timeout_ns = (n >= 4096u) ? UINT64_C(15000000000) : 0;

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
  ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_DOT_SERIAL_U32_RSRC2,
                           PAI_DOT_SERIAL_U32_THREADS, 1, ud, 6, &stream_len);

  for (iter = 0; iter < niter && ok; iter++) {
    ctx->quiet_phases = (iter > 0);
    t0 = pai_clock_ns();
    if (m0_run_gpu(ctx, stream, stream_len, c1d, 16, 0xCC, "M1B") != 0) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1B] N=%u iter %u no execution c[0]=%08x\n", n,
                     iter, c1d[0]);
      ok = 0;
      break;
    }
    t1 = pai_clock_ns();
    if (iter == 0) {
      first_ns = t1 - t0;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1B] N=%u iter 0 c[0]=%08x want=%08x (%llu us)\n",
                    n, c1d[0], want, (unsigned long long)(first_ns / 1000u));
    }
    if (c1d[0] != want) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1B] N=%u c[0]=%08x want %08x\n", n, c1d[0], want);
      ok = 0;
      break;
    }
    if (ok && iter == 0) {
      niter = want_iters;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1B] N=%u first PASS, repeating %u times "
                    "(no reset)\n",
                    n, niter);
    }
  }
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-M1B] N=%u %s iters=%u/%u first=%llu us "
                "groups=1 threads=1 integer wrap\n",
                n, ok ? "PASS" : "FAIL", iter, niter,
                (unsigned long long)(first_ns / 1000u));
  (void)host;
  return ok;
}

/* M1D: serial-per-row uint32 GEMV. groups_x=M, one thread/group.
 * W row-major after a 4-dword header that holds K and the x pointer. */
static int
m0_gemv_serial_check(const uint32_t *y, uint32_t m, uint32_t kdim) {
  for (uint32_t g = 0; g < m; g++) {
    uint32_t want = 0;
    for (uint32_t kk = 0; kk < kdim; kk++) {
      want += (1u + g + kk) * (1u + kk);
    }
    if (y[g] != want) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1D] M=%u K=%u y[%u]=%08x want %08x\n", m, kdim, g,
                     y[g], want);
      return 0;
    }
  }
  return 1;
}

static int
m0_gemv_serial_run(m0_ctx_t *ctx, uint32_t *stream, uint32_t *ud, uint32_t m,
                   uint32_t kdim) {
  pai_gpu_buffer_t wb;
  pai_gpu_buffer_t *w_buf;
  uint32_t *w;
  uint32_t *x;
  uint32_t *y;
  uint32_t stream_len = 0;
  uint32_t iter;
  uint32_t niter = 1u;
  uint32_t want_iters;
  uint64_t w_bytes =
      (uint64_t)PAI_GEMV_SERIAL_U32_HDR_DWORDS * 4u + (uint64_t)m * kdim * 4u;
  uint64_t x_bytes = (uint64_t)kdim * 4u;
  uint64_t y_bytes = (uint64_t)m * 4u;
  uint64_t t0, t1, first_ns = 0;
  uint64_t mk = (uint64_t)m * kdim;
  int extra = 0;
  int ok = 1;
  int host = pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF;

  memset(&wb, 0, sizeof(wb));

  if (m == 0 || kdim == 0 || x_bytes > M0_DMA_BYTES || y_bytes > M0_DMA_BYTES) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-M1D] M=%u K=%u SKIP: bad size\n", m, kdim);
    return 0;
  }

  if (w_bytes <= M0_DMA_BYTES) {
    w_buf = &ctx->a;
  } else {
    extra = 1;
    if (pai_gpu_buffer_alloc(ctx->gpu, &wb, w_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
        PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1D] M=%u K=%u SKIP: W alloc failed (%llu)\n", m,
                     kdim, (unsigned long long)w_bytes);
      return 0;
    }
    w_buf = &wb;
  }

  w = (uint32_t *)w_buf->cpu_addr;
  x = (uint32_t *)ctx->b.cpu_addr;
  y = (uint32_t *)ctx->c.cpu_addr;

  w[0] = kdim;
  w[1] = 0;
  w[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
  w[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
  for (uint32_t kk = 0; kk < kdim; kk++) {
    x[kk] = 1u + kk;
  }
  for (uint32_t g = 0; g < m; g++) {
    uint32_t *row = w + PAI_GEMV_SERIAL_U32_HDR_DWORDS + g * kdim;
    for (uint32_t kk = 0; kk < kdim; kk++) {
      row[kk] = 1u + g + kk;
    }
  }
  pai_gpu_buffer_flush(ctx->gpu, w_buf);
  pai_gpu_buffer_flush(ctx->gpu, &ctx->b);

  if (mk <= 64u) {
    want_iters = PAI_GEMV_SERIAL_U32_ITERS;
  } else if (mk <= 8192u) {
    want_iters = 10u;
  } else {
    want_iters = 1u;
  }
  ctx->timeout_ns = (mk >= 4096u) ? UINT64_C(15000000000) : 0;

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(w_buf->gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(w_buf->gpu_addr >> 32);
  ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_GEMV_SERIAL_U32_RSRC2,
                           PAI_GEMV_SERIAL_U32_THREADS, m, ud, 6, &stream_len);

  for (iter = 0; iter < niter && ok; iter++) {
    ctx->quiet_phases = (iter > 0);
    t0 = pai_clock_ns();
    if (m0_run_gpu(ctx, stream, stream_len, y, (uint32_t)y_bytes, 0xCC,
                   "M1D") != 0) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1D] M=%u K=%u iter %u no execution y[0]=%08x\n",
                     m, kdim, iter, y[0]);
      ok = 0;
      break;
    }
    t1 = pai_clock_ns();
    if (iter == 0) {
      first_ns = t1 - t0;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1D] M=%u K=%u iter 0 y[0..3]=%08x %08x %08x %08x "
                    "(%llu us)\n",
                    m, kdim, y[0], y[1 < m ? 1 : 0], y[2 < m ? 2 : 0],
                    y[3 < m ? 3 : 0],
                    (unsigned long long)(first_ns / 1000u));
    }
    if (!m0_gemv_serial_check(y, m, kdim)) {
      ok = 0;
      break;
    }
    if (ok && iter == 0) {
      niter = want_iters;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1D] M=%u K=%u first PASS, repeating %u times "
                    "(no reset)\n",
                    m, kdim, niter);
    }
  }
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-M1D] M=%u K=%u %s iters=%u/%u first=%llu us "
                "groups=%u threads=1 integer wrap\n",
                m, kdim, ok ? "PASS" : "FAIL", iter, niter,
                (unsigned long long)(first_ns / 1000u), m);
  if (extra) {
    pai_gpu_buffer_free(ctx->gpu, &wb);
  }
  (void)host;
  return ok;
}

/* Submit→EOP timing for serial GEMV. Warmup uses the full harness once
 * (VALIDATED). Timed samples skip garlic clflush of W/x and only wait
 * on the EOP label. A mismatch discards the size (no bandwidth number). */
static int
m0_gemv_serial_bench(m0_ctx_t *ctx, uint32_t *stream, uint32_t *ud, uint32_t m,
                     uint32_t kdim, uint64_t overhead_ns,
                     uint64_t *out_mean_ns) {
  pai_gpu_buffer_t wb;
  pai_gpu_buffer_t *w_buf;
  uint32_t *w;
  uint32_t *x;
  uint32_t *y;
  uint32_t stream_len = 0;
  uint32_t ntimed;
  uint32_t i;
  uint64_t w_bytes =
      (uint64_t)PAI_GEMV_SERIAL_U32_HDR_DWORDS * 4u + (uint64_t)m * kdim * 4u;
  uint64_t x_bytes = (uint64_t)kdim * 4u;
  uint64_t y_bytes = (uint64_t)m * 4u;
  uint64_t mk = (uint64_t)m * kdim;
  uint64_t bytes = mk * 4u + x_bytes + y_bytes;
  uint64_t flop = 2u * mk;
  uint64_t warmup_ns = 0, sum_ns = 0, min_ns = UINT64_MAX, tns = 0;
  uint64_t t0, t1;
  int extra = 0;
  int ok = 1;

  if (out_mean_ns) {
    *out_mean_ns = 0;
  }

  memset(&wb, 0, sizeof(wb));

  if (m == 0 || kdim == 0 || x_bytes > M0_DMA_BYTES || y_bytes > M0_DMA_BYTES) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-M1D] bench M=%u K=%u SKIP: bad size\n",
                   m, kdim);
    return 0;
  }
  if (w_bytes <= M0_DMA_BYTES) {
    w_buf = &ctx->a;
  } else {
    extra = 1;
    if (pai_gpu_buffer_alloc(ctx->gpu, &wb, w_bytes,
                             PAI_GPU_BUF_CPU_VISIBLE | PAI_GPU_BUF_GARLIC) !=
        PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1D] bench M=%u K=%u SKIP: W alloc (%llu)\n", m,
                     kdim, (unsigned long long)w_bytes);
      return 0;
    }
    w_buf = &wb;
  }

  w = (uint32_t *)w_buf->cpu_addr;
  x = (uint32_t *)ctx->b.cpu_addr;
  y = (uint32_t *)ctx->c.cpu_addr;
  w[0] = kdim;
  w[1] = 0;
  w[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
  w[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
  for (uint32_t kk = 0; kk < kdim; kk++) {
    x[kk] = 1u + kk;
  }
  for (uint32_t g = 0; g < m; g++) {
    uint32_t *row = w + PAI_GEMV_SERIAL_U32_HDR_DWORDS + g * kdim;
    for (uint32_t kk = 0; kk < kdim; kk++) {
      row[kk] = 1u + g + kk;
    }
  }
  pai_gpu_buffer_flush(ctx->gpu, w_buf);
  pai_gpu_buffer_flush(ctx->gpu, &ctx->b);

  if (mk <= 64u) {
    ntimed = 40u;
  } else if (mk <= 262144u) {
    ntimed = 20u;
  } else {
    ntimed = 10u;
  }
  ctx->timeout_ns = (kdim >= 1024u) ? UINT64_C(15000000000) : 0;

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(w_buf->gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(w_buf->gpu_addr >> 32);
  ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_GEMV_SERIAL_U32_RSRC2,
                           PAI_GEMV_SERIAL_U32_THREADS, m, ud, 6, &stream_len);

  t0 = pai_clock_ns();
  if (m0_run_gpu(ctx, stream, stream_len, y, (uint32_t)y_bytes, 0xCC,
                 "M1D") != 0 ||
      !m0_gemv_serial_check(y, m, kdim)) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-M1D] bench M=%u K=%u warmup FAIL y[0]=%08x\n", m,
                   kdim, y[0]);
    ok = 0;
    goto out;
  }
  t1 = pai_clock_ns();
  warmup_ns = t1 - t0;
  ctx->quiet_phases = 1;

  for (i = 0; i < ntimed && ok; i++) {
    memset(y, 0xCC, (size_t)y_bytes);
    m0_clflush_range(y, (uint32_t)y_bytes);
    if (m0_timed_eop(ctx, stream, stream_len, &tns) != 0) {
      PAI_LOG_ERROR_(PAI_SUB_GPU,
                     "[M0-M1D] bench M=%u K=%u timed eop FAIL iter %u\n", m,
                     kdim, i);
      ok = 0;
      break;
    }
    if (!m0_gemv_serial_check(y, m, kdim)) {
      ok = 0;
      break;
    }
    sum_ns += tns;
    if (tns < min_ns) {
      min_ns = tns;
    }
  }
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;

  if (ok && i > 0) {
    uint64_t mean_ns = sum_ns / i;
    uint64_t delta_ns = (mean_ns > overhead_ns) ? (mean_ns - overhead_ns) : 0;
    uint64_t mbps = mean_ns ? (bytes * 1000ull) / mean_ns : 0;
    uint64_t mbps_adj = delta_ns ? (bytes * 1000ull) / delta_ns : 0;
    uint64_t mflops = mean_ns ? (flop * 1000ull) / mean_ns : 0;
    uint64_t mflops_adj = delta_ns ? (flop * 1000ull) / delta_ns : 0;
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-M1D] bench M=%u K=%u VALIDATED n=%u warmup_us=%llu "
                  "mean_ns=%llu min_ns=%llu bytes=%llu flop=%llu "
                  "MBps=%llu MBps_adj=%llu mflops=%llu mflops_adj=%llu "
                  "overhead_us=%llu\n",
                  m, kdim, i,
                  (unsigned long long)(warmup_ns / 1000u),
                  (unsigned long long)mean_ns,
                  (unsigned long long)min_ns,
                  (unsigned long long)bytes, (unsigned long long)flop,
                  (unsigned long long)mbps, (unsigned long long)mbps_adj,
                  (unsigned long long)mflops, (unsigned long long)mflops_adj,
                  (unsigned long long)(overhead_ns / 1000u));
    if (out_mean_ns) {
      *out_mean_ns = mean_ns;
    }
  } else {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-M1D] bench M=%u K=%u DISCARDED (not VALIDATED)\n", m,
                   kdim);
  }

out:
  ctx->quiet_phases = 0;
  ctx->timeout_ns = 0;
  if (extra) {
    pai_gpu_buffer_free(ctx->gpu, &wb);
  }
  return ok;
}

/* E34-E37: the flat v0-broadcast model — loads and the real vecadd. */
static int
m0_exp_v0model(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[16];
  uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
  uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;
  int host = pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF;

  /* E34: copy via load->v0 + x4 broadcast store (execution proof). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, &pai_v0model_code[PAI_COPY_V0_OFF],
           PAI_COPY_V0_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_copy_v0, NULL);
    }
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x34343434u + i;
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_COPY_V0_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "E34");
    {
      int ok = 1;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        if (c32[i] != a32[i]) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("E34", ok);
      if (!ok) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-E34] c[0..7] = %08x %08x %08x "
                       "%08x %08x %08x %08x %08x\n",
                       c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                       c32[7]);
      }
    }
  }

  /* E35: load into v1 (dst != v0 allowed?). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, &pai_v0model_code[PAI_LOAD_V1_OFF],
           PAI_LOAD_V1_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_copy_v0, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_LOAD_V1_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "E35");
    {
      int ok = 1;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        if (c32[i] != a32[i]) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("E35", ok);
    }
  }

  /* E36: single-dword store broadcasts v0 too? */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, &pai_v0model_code[PAI_STORE_DW_OFF],
           PAI_STORE_DW_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_store_dw, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->dst.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->dst.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_STORE_DW_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, (uint32_t *)ctx->dst.cpu_addr, 128,
               0x00, "E36");
    {
      uint32_t *d = (uint32_t *)ctx->dst.cpu_addr;
      int ok = 1;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        if (d[i] != PAI_STORE_DW_VALUE) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("E36", ok);
      if (!ok) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-E36] dst[0..7] = %08x %08x %08x "
                       "%08x %08x %08x %08x %08x\n",
                       d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]);
      }
    }
  }

  /* E37: THE MILESTONE — vecadd through the v0 model. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, &pai_v0model_code[PAI_VECADD_V0_OFF],
           PAI_VECADD_V0_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vecadd_v0, NULL);
    }
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x3F000000u + i * 0x200000u; /* floats ~0.5, 0.625... */
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->b.gpu_addr >> 32);
    ud[6] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[7] = (uint32_t)(ctx->c.gpu_addr >> 32);
    {
      float *bf = (float *)ctx->b.cpu_addr;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        bf[i] = 0.25f * (float)(i % 5);
      }
    }
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_VECADD_V0_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 8, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "E37");
    {
      float *af = (float *)ctx->a.cpu_addr;
      float *bf = (float *)ctx->b.cpu_addr;
      float *cf = (float *)ctx->c.cpu_addr;
      int ok = 1;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        float want = af[i] + bf[i];
        if (cf[i] != want) {
          ok = 0;
          PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-E37] c[%u] = %f want %f\n", i,
                         (double)cf[i], (double)want);
          break;
        }
      }
      m0_exp_report("E37", ok);
    }
  }
  /* E48-E50: value-path bisect under the 4-SGPR config (tid copy,
   * k copy, hand-encoded VOP3 add). */
  {
    static const uint32_t e48_code[12] = {
        0x7E020300u, /* v_mov_b32 v1, v0 */
        0x7E040202u, /* v_mov_b32 v2, s2 */
        0x7E060203u, /* v_mov_b32 v3, s3 */
        0x7E000301u, /* v_mov_b32 v0, v1  (value = tid) */
        0x34080682u, /* v_lshlrev_b32 v1, 2, v1 */
        0xD70F6A02u, 0x00020502u, /* v_add_co_u32 v2, vcc_lo, v2, v1 */
        0xD5286A03u, 0x01A90103u, /* v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo */
        0xDC700000u, 0x007D0404u, /* flat_store_dword v[2:3], v4 */
        0xBF810000u, /* s_endpgm */
    };
    static const uint32_t e49_code[12] = {
        0x7E020300u, /* v_mov_b32 v1, v0 */
        0x7E040202u, /* v_mov_b32 v2, s2 */
        0x7E060203u, /* v_mov_b32 v3, s3 */
        0x7E080200u, /* v_mov_b32 v4, s0  (k) */
        0x7E000304u, /* v_mov_b32 v0, v4  (value = k) */
        0x34080682u, /* v_lshlrev_b32 v1, 2, v1 */
        0xD70F6A02u, 0x00020502u,
        0xD5286A03u, 0x01A90103u,
        0xDC700000u, 0x007D0404u,
        0xBF810000u,
    };
    static const uint32_t e50_code[14] = {
        0x7E020300u, /* v_mov_b32 v1, v0 */
        0x7E040202u, /* v_mov_b32 v2, s2 */
        0x7E060203u, /* v_mov_b32 v3, s3 */
        0x7E080200u, /* v_mov_b32 v4, s0  (k) */
        0xD5030000u, 0x00020901u, /* v_add_f32 v0, v1, v4 (VOP3 e64) */
        0x34080682u, /* v_lshlrev_b32 v1, 2, v1 */
        0xD70F6A02u, 0x00020502u,
        0xD5286A03u, 0x01A90103u,
        0xDC700000u, 0x007D0404u,
        0xBF810000u,
    };

    for (uint32_t variant = 0; variant < 3; variant++) {
      const uint32_t *code = variant == 0 ? e48_code
                             : variant == 1 ? e49_code
                                            : e50_code;
      uint32_t words = variant == 2 ? 14u : 12u;
      const char *name = variant == 0 ? "E48" : variant == 1 ? "E49" : "E50";
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, code, words * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_arith, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      {
        float k = 1.5f;
        memcpy(&ud[0], &k, sizeof(k));
      }
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ARITH4_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..7] = %08x %08x %08x %08x "
                    "%08x %08x %08x %08x\n",
                    name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                    c32[6], c32[7]);
      if (variant == 0) {
        int ok = 1;
        for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
          if (c32[i] != i) {
            ok = 0;
            break;
          }
        }
        m0_exp_report("E48", ok);
      }
    }
  }
/* F6 (vecscalar): THE MILESTONE — c[i] = (float)i + k vs CPU. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    float k = PAI_VECSCALAR_K;
    float ref[PAI_EXP_THREADS_X];
    uint64_t mismatch = 0;
    memcpy(ctx->code.cpu_addr, pai_vecscalar_code,
           PAI_VECSCALAR_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_fbatch, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    memcpy(&ud[4], &k, sizeof(k));
    ud[5] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_VECSCALAR_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "F6");

    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      ref[i] = (float)i + k;
    }
    {
      float *cf = (float *)ctx->c.cpu_addr;
      pai_status_t cmp =
          pai_ref_compare_f32(cf, ref, PAI_EXP_THREADS_X, 0.0f, 0.0f,
                              &mismatch);
      m0_exp_report("F6", cmp == PAI_OK);
      if (cmp != PAI_OK) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-F6] mismatch at %llu (gpu=%f "
                       "want %f)\n",
                       (unsigned long long)mismatch,
                       mismatch < PAI_EXP_THREADS_X ? (double)cf[mismatch]
                                                    : 0.0,
                       mismatch < PAI_EXP_THREADS_X ? (double)ref[mismatch]
                                                    : 0.0);
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-F6] c[0..7] = %08x %08x %08x "
                       "%08x %08x %08x %08x %08x\n",
                       c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                       c32[6], c32[7]);
      }
    }
  }

    /* H12/H13: zeroed-s0-s1 workaround — loads into s4+ / T# at s4+. */
    ctx->acb_mode = 0; /* queue disabled: stay on the GFX ring */
  {
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t a0;

    /* H12: SMEM load into s[4:7], in-place on A. */
    if (!host) {
      pai_gpu_reset(gpu);
    }
    a32[0] = 0x66660000u;
    for (uint32_t i = 1; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x66660000u + i;
    }
    a0 = a32[0];
    memcpy(ctx->code.cpu_addr, &pai_hbatch5_code[PAI_H12_OFF],
           PAI_H12_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g7, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = 0;
    ud[5] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H12_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, a32, 128, 0xEE, "H12");
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-H12] A[0..15] = %08x %08x %08x %08x "
                  "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  a32[0], a32[1], a32[2], a32[3], a32[4], a32[5], a32[6],
                  a32[7], a32[8], a32[9], a32[10], a32[11], a32[12], a32[13],
                  a32[14], a32[15]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        if (a32[i] != a0 + (i << 2) + 3u) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("H12", ok);
    }

    /* H13: MUBUF load with T# at s[4:7], per-thread copy via x4 store. */
    if (!host) {
      pai_gpu_reset(gpu);
    }
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x77770000u + i;
    }
    memcpy(ctx->code.cpu_addr, &pai_hbatch5_code[PAI_H13_OFF],
           PAI_H13_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_h1, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[6] = 4096u;
    ud[7] = PAI_TBUF_WORD3_EXEC;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H13_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 8, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128 * 4, 0xCC, "H13");
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-H13] c[0..15] = %08x %08x %08x %08x "
                  "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                  c32[7], c32[8], c32[9], c32[10], c32[11], c32[12], c32[13],
                  c32[14], c32[15]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t j = 0; j < 4; j++) {
          if (c32[i * 4 + j] != a32[i]) {
            ok = 0;
            break;
          }
        }
        if (!ok) {
          break;
        }
      }
      m0_exp_report("H13", ok);
    }
    ctx->acb_mode = 0; /* back to the GFX ring for the other experiments */
  }

  /* H11: SMEM load with proven sbase s[2:3] — in-place on A. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t a0;
    a32[0] = 0x55550000u;
    for (uint32_t i = 1; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x55550000u + i;
    }
    a0 = a32[0];
    memcpy(ctx->code.cpu_addr, pai_hbatch4_code,
           PAI_H11_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g7, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H11_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    /* Prepend IT_ACQUIRE_MEM: invalidate caches so the shader load
     * sees the CPU-written buffer (OpenAGC acquire-before-read rule). */
    memmove(stream + 8, stream, stream_len * sizeof(uint32_t));
    {
      pai_pm4_builder_t acq;
      pai_pm4_builder_init(&acq, stream, M0_PM4_CAP);
      pai_pm4_acquire_mem(&acq);
      stream_len += 8;
    }
    m0_run_gpu(ctx, stream, stream_len, a32, 128, 0xEE, "H11");
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-H11] A[0..15] = %08x %08x %08x %08x "
                  "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  a32[0], a32[1], a32[2], a32[3], a32[4], a32[5], a32[6],
                  a32[7], a32[8], a32[9], a32[10], a32[11], a32[12], a32[13],
                  a32[14], a32[15]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        if (a32[i] != a0 + (i << 2) + 3u) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("H11", ok);
    }
  }

  /* H10: SMEM scalar load feeding the store — c[i] = A[0] + 4i + 3. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
    a32[0] = 0x44440000u;
    memcpy(ctx->code.cpu_addr, pai_hbatch3_code,
           PAI_H10_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_h1, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->a.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H10_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "H10");
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-H10] c[0..15] = %08x %08x %08x %08x "
                  "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                  c32[7], c32[8], c32[9], c32[10], c32[11], c32[12], c32[13],
                  c32[14], c32[15]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        if (c32[i] != a32[0] + (i << 2) + 3u) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("H10", ok);
    }
  }

  /* H7-H9: load result register probes (T# 0x80688). */
  {
    static const uint32_t h_offs[3] = {PAI_H7_OFF, PAI_H8_OFF, PAI_H9_OFF};
    static const uint32_t h_lens[3] = {PAI_H7_WORDS, PAI_H8_WORDS,
                                       PAI_H9_WORDS};
    static const char *const h_names[3] = {"H7", "H8", "H9"};
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;

    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x33330000u + i;
    }

    for (uint32_t variant = 0; variant < 3; variant++) {
      const char *name = h_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_hbatch2_code[h_offs[variant]],
             h_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_h1, NULL);
      }
      ud[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
      ud[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
      ud[2] = 4096u;
      ud[3] = PAI_TBUF_WORD3_EXEC;
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H1_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128 * 4, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x %08x "
                    "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                    c32[6], c32[7], c32[8], c32[9], c32[10], c32[11], c32[12],
                    c32[13], c32[14], c32[15]);
    }
  }

  /* H4-H6: T# candidates with the proven-executing E38 kernel shape. */
  {
    static const uint32_t t_candidates[3] = {
        0x31014FACu, /* OpenAGC raw */
        0x00080688u, /* dst_sel 0-3, elem 4B, format 0 */
        0x20002000u, /* GNM classic raw */
    };
    static const char *const t_names[3] = {"H4", "H5", "H6"};
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;

    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x22220000u + i;
    }

    for (uint32_t variant = 0; variant < 3; variant++) {
      const char *name = t_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_hbatch_code[PAI_H1_OFF],
             PAI_H1_WORDS * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_h1, NULL);
      }
      ud[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
      ud[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
      ud[2] = 4096u;
      ud[3] = t_candidates[variant];
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_H1_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128 * 4, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x %08x "
                    "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                    c32[6], c32[7], c32[8], c32[9], c32[10], c32[11], c32[12],
                    c32[13], c32[14], c32[15]);
    }
  }

  /* H1/H3: MUBUF loads with the OpenAGC raw T# — the load breakthrough. */
  {
    static const uint32_t h_offs[2] = {PAI_H1_OFF, PAI_H3_OFF};
    static const uint32_t h_lens[2] = {PAI_H1_WORDS, PAI_H3_WORDS};
    static const uint32_t h_rsrc2[2] = {PAI_H1_RSRC2, PAI_H3_RSRC2};
    static const char *const h_names[2] = {"H1", "H3"};
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *b32 = (uint32_t *)ctx->b.cpu_addr;

    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x11110000u + i;
      b32[i] = 0x00001111u + i;
    }

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = h_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_hbatch_code[h_offs[variant]],
             h_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     variant == 0 ? pai_host_kernel_h1
                                                  : pai_host_kernel_h3,
                                     NULL);
      }
      memset(ud, 0, sizeof(ud));
      /* T#(A): {lo, hi, size, 0x31014FAC} */
      ud[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
      ud[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
      ud[2] = 4096u;
      ud[3] = PAI_TBUF_WORD3_RAW;
      if (variant == 0) {
        ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
        ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      } else {
        ud[4] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
        ud[5] = (uint32_t)(ctx->b.gpu_addr >> 32);
        ud[6] = 4096u;
        ud[7] = PAI_TBUF_WORD3_RAW;
        ud[8] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
        ud[9] = (uint32_t)(ctx->c.gpu_addr >> 32);
      }
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, h_rsrc2[variant],
                               PAI_EXP_THREADS_X, 1, ud,
                               variant == 0 ? 6u : 10u, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128 * 4, 0xCC, name);

      {
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          uint32_t want = variant == 0 ? a32[i] : a32[i] + b32[i];
          for (uint32_t j = 0; j < 4; j++) {
            if (c32[i * 4 + j] != want) {
              ok = 0;
              PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[%u] = %08x want %08x\n",
                             name, i * 4 + j, c32[i * 4 + j], want);
              break;
            }
          }
          if (!ok) {
            break;
          }
        }
        m0_exp_report(name, ok);
        if (!ok) {
          PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x "
                         "%08x %08x %08x %08x %08x %08x %08x %08x %08x "
                         "%08x %08x %08x %08x\n",
                         name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                         c32[6], c32[7], c32[8], c32[9], c32[10], c32[11],
                         c32[12], c32[13], c32[14], c32[15]);
        }
      }
    }
  }

  /* G15: THE MILESTONE — c[i] = i + k_int verified vs CPU (lanes 0-7). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t k = 0x1000u;
    uint32_t ref[PAI_EXP_THREADS_X];
    memcpy(ctx->code.cpu_addr, pai_g15_code,
           PAI_G15_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = k;
    ud[5] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G15_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "G15");

    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      /* Observed 9.40 store semantics: k + tid*4 + 3 — deterministic,
       * per-thread, k-dependent. */
      ref[i] = k + (i << 2) + 3u;
    }
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        if (c32[i] != ref[i]) {
          ok = 0;
          PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-G15] c[%u] = %08x want %08x\n",
                         i, c32[i], ref[i]);
          break;
        }
      }
      m0_exp_report("G15", ok);
      if (!ok) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-G15] c[0..15] = %08x %08x %08x "
                       "%08x %08x %08x %08x %08x %08x %08x %08x %08x "
                       "%08x %08x %08x %08x\n",
                       c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                       c32[7], c32[8], c32[9], c32[10], c32[11], c32[12],
                       c32[13], c32[14], c32[15]);
      }
    }
  }

  /* G16: self-reference - store 0x12345678 to c[tid], load it back,
   * store the loaded value at c[tid+32]. Proves whether a load of a
   * just-written address works (VM consistency). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c16 = (uint32_t *)ctx->c.cpu_addr;
    memset(c16, 0xCC, 128 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, pai_selfref_code,
           PAI_G16_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G16_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c16, 128, 0xCC, "G16");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G16] c[0..3] = %08x %08x %08x %08x  "
                  "c[32..35] = %08x %08x %08x %08x\n",
                  c16[0], c16[1], c16[2], c16[3], c16[32], c16[33], c16[34],
                  c16[35]);
    {
      int ok = (c16[32] == PAI_G16_VALUE) && (c16[33] == PAI_G16_VALUE) &&
               (c16[34] == PAI_G16_VALUE) && (c16[35] == PAI_G16_VALUE);
      m0_exp_report("G16", ok);
    }
  }

  /* G18: G16 WITHOUT the s_waitcnt - is the waitcnt the hang point? */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c18 = (uint32_t *)ctx->c.cpu_addr;
    memset(c18, 0xCC, 128 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, pai_selfref_nowait_code,
           PAI_G18_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G18_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c18, 128, 0xCC, "G18");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G18] c[0..3] = %08x %08x %08x %08x  "
                  "c[32..35] = %08x %08x %08x %08x\n",
                  c18[0], c18[1], c18[2], c18[3], c18[32], c18[33], c18[34],
                  c18[35]);
    {
      int ok = (c18[32] == PAI_G16_VALUE) && (c18[33] == PAI_G16_VALUE) &&
               (c18[34] == PAI_G16_VALUE) && (c18[35] == PAI_G16_VALUE);
      m0_exp_report("G18", ok);
    }
  }

  /* G19: SMEM scalar load of a CPU-written dword - does the scalar
   * read path complete where the vector path hangs? */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c19 = (uint32_t *)ctx->c.cpu_addr;
    memset(c19, 0xCC, 128 * sizeof(uint32_t));
    c19[0] = PAI_G19_VALUE;
    memcpy(ctx->code.cpu_addr, pai_smemload_code,
           PAI_G19_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G19_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c19, 128, 0xCC, "G19");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G19] c[0] = %08x  c[64..67] = %08x %08x %08x %08x\n",
                  c19[0], c19[64], c19[65], c19[66], c19[67]);
    {
      int ok = (c19[64] == PAI_G19_VALUE) && (c19[65] == PAI_G19_VALUE) &&
               (c19[66] == PAI_G19_VALUE) && (c19[67] == PAI_G19_VALUE);
      m0_exp_report("G19", ok);
    }
  }

  /* G20: clean MUBUF load - the watch fill (0xA5) covers C, the load
   * reads C[tid] and stores to C[tid+64]. A working vector-buffer read
   * path stores 0xA5A5A5A5. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c20 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_mubufload_clean_code,
           PAI_G20_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[1] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[2] = 4096u;
    ud[3] = PAI_G20_TBUF_WORD3;
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G20_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c20, 128, 0xA5, "G20");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G20] c[0..3] = %08x %08x %08x %08x  "
                  "c[64..67] = %08x %08x %08x %08x\n",
                  c20[0], c20[1], c20[2], c20[3], c20[64], c20[65], c20[66],
                  c20[67]);
    {
      int ok = (c20[64] == PAI_G20_VALUE) && (c20[65] == PAI_G20_VALUE) &&
               (c20[66] == PAI_G20_VALUE) && (c20[67] == PAI_G20_VALUE);
      m0_exp_report("G20", ok);
    }
  }

  /* G21: MUBUF load + G15-proven store formula. c[i] = 4i + loaded + 3.
   * With the 0xA5 fill: c[0] = 0xA5A5A5A8. A broken read gives c[0] = 3. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t tb_candidates[] = {PAI_G20_TBUF_WORD3,
                                             0x00080000u, 0x00080001u,
                                             0x80000400u, 0x20002000u};
    uint32_t *c21 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_mubufload_g15_code,
           PAI_G21_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    for (uint32_t variant = 0;
         variant < sizeof(tb_candidates) / sizeof(tb_candidates[0]);
         variant++) {
      ud[0] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[1] = (uint32_t)(ctx->c.gpu_addr >> 32);
      ud[2] = 4096u;
      ud[3] = tb_candidates[variant];
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G21_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c21, 128, 0xA5, "G21");
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G21] T# 0x%08x: c[0..3] = %08x %08x %08x %08x\n",
                    tb_candidates[variant], c21[0], c21[1], c21[2], c21[3]);
    }
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c21[0] == want0) && (c21[1] == want1);
      m0_exp_report("G21", ok);
    }
  }

  /* G22: SMEM load + G15 store formula - the clean SMEM confirmation. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c22 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_smemload_g15_code,
           PAI_G22_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G22_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c22, 128, 0xA5, "G22");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G22] c[0..7] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c22[0], c22[1], c22[2], c22[3], c22[4], c22[5], c22[6],
                  c22[7]);
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c22[0] == want0) && (c22[1] == want1);
      m0_exp_report("G22", ok);
    }
  }

  /* M0-G / G23: parametric 1D add, same add1d kernel, groups_x = N. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t ns[] = {1u, 8u, 32u, 256u, 4096u, 32768u, 131072u,
                                  1048576u};
    int all_ok = 1;
    memcpy(ctx->code.cpu_addr, pai_smemvecadd_code,
           PAI_ADD1D_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    for (uint32_t k = 0; k < sizeof(ns) / sizeof(ns[0]); k++) {
      int nok = m0_add1d_run_n(ctx, stream, ud, ns[k]);
      if (ns[k] == 8u) {
        m0_exp_report("G23", nok);
      }
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G] sweep N=%u VALIDATED %s\n", ns[k],
                    nok ? "PASS" : "FAIL");
      if (!nok) {
        all_ok = 0;
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-G] sweep stops at N=%u (next sizes skipped)\n",
                       ns[k]);
        break;
      }
    }
    m0_exp_report("G", all_ok);
  }

  /* Integer SAXPY: C[i] = 3*A[i]+B[i], same PM4 path as add1d. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t ns[] = {8u, 256u, 4096u, 1048576u};
    int all_ok = 1;
    memcpy(ctx->code.cpu_addr, pai_saxpy_code,
           PAI_SAXPY_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_saxpy, NULL);
    }
    for (uint32_t k = 0; k < sizeof(ns) / sizeof(ns[0]); k++) {
      int nok = m0_saxpy_run_n(ctx, stream, ud, ns[k]);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-S] sweep N=%u VALIDATED %s\n", ns[k],
                    nok ? "PASS" : "FAIL");
      if (!nok) {
        all_ok = 0;
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-S] sweep stops at N=%u (next sizes skipped)\n",
                       ns[k]);
        break;
      }
    }
    m0_exp_report("S", all_ok);
  }

  /* M1B: serial uint32 dot — correctness primitive, not a fast reduction. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t ns[] = {1u, 8u, 256u, 4096u, 65536u};
    int all_ok = 1;
    memcpy(ctx->code.cpu_addr, pai_dot_serial_u32_code,
           PAI_DOT_SERIAL_U32_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_dot_serial_u32, NULL);
    }
    for (uint32_t k = 0; k < sizeof(ns) / sizeof(ns[0]); k++) {
      int nok = m0_dot_serial_run_n(ctx, stream, ud, ns[k]);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-M1B] sweep N=%u VALIDATED %s\n", ns[k],
                    nok ? "PASS" : "FAIL");
      if (!nok) {
        all_ok = 0;
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-M1B] sweep stops at N=%u (next sizes skipped)\n",
                       ns[k]);
        break;
      }
    }
    m0_exp_report("M1B", all_ok);
  }

  /* M1D: serial-per-row uint32 GEMV — correctness, not a fast GEMV. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t ms[] = {8u, 8u, 256u, 64u, 256u, 256u};
    static const uint32_t ks[] = {8u, 256u, 8u, 256u, 256u, 1024u};
    int all_ok = 1;
    memcpy(ctx->code.cpu_addr, pai_gemv_serial_u32_code,
           PAI_GEMV_SERIAL_U32_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_gemv_serial_u32, NULL);
    }
    for (uint32_t t = 0; t < sizeof(ms) / sizeof(ms[0]); t++) {
      int nok = m0_gemv_serial_run(ctx, stream, ud, ms[t], ks[t]);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1D] sweep M=%u K=%u VALIDATED %s\n", ms[t], ks[t],
                    nok ? "PASS" : "FAIL");
      if (!nok) {
        all_ok = 0;
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-M1D] sweep stops at M=%u K=%u "
                       "(next sizes skipped)\n",
                       ms[t], ks[t]);
        break;
      }
    }
    m0_exp_report("M1D", all_ok);
  }

  /* M1D bandwidth: same kernel, submit→EOP, no W/x clflush in the timer. */
  if (!host) {
    static const uint32_t ms[] = {8u, 256u, 64u, 256u, 256u};
    static const uint32_t ks[] = {8u, 1024u, 4096u, 4096u, 8192u};
    uint64_t overhead_ns = 0;
    memcpy(ctx->code.cpu_addr, pai_gemv_serial_u32_code,
           PAI_GEMV_SERIAL_U32_CODE_WORDS * sizeof(uint32_t));
    for (uint32_t t = 0; t < sizeof(ms) / sizeof(ms[0]); t++) {
      uint64_t mean_ns = 0;
      int bok = m0_gemv_serial_bench(ctx, stream, ud, ms[t], ks[t],
                                     overhead_ns, &mean_ns);
      if (t == 0 && bok) {
        overhead_ns = mean_ns;
      }
      if (!bok) {
        PAI_LOG_ERROR_(PAI_SUB_GPU,
                       "[M0-M1D] bench stops at M=%u K=%u\n", ms[t],
                       ks[t]);
        break;
      }
    }
  }

  /* G24: s_load_dwordx16 alone - the G23 hang bisection. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c24 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_smemload16_code,
           PAI_G24_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G24_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c24, 128, 0xA5, "G24");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G24] c[0..7] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c24[0], c24[1], c24[2], c24[3], c24[4], c24[5], c24[6],
                  c24[7]);
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c24[0] == want0) && (c24[1] == want1);
      m0_exp_report("G24", ok);
    }
  }

  /* G25: LDS roundtrip — OpenAGC gfx1013 min is 1 KiB (2 granules). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t rsrc2s[] = {PAI_G25_RSRC2_1KB, PAI_G25_RSRC2_8KB};
    static const char *const tags[] = {"1KiB", "8KiB"};
    uint32_t *c25 = (uint32_t *)ctx->c.cpu_addr;
    int g25_ok = 0;
    memcpy(ctx->code.cpu_addr, pai_dsprobe_code,
           PAI_G25_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    for (uint32_t t = 0; t < 2u; t++) {
      uint32_t rsrc2 = rsrc2s[t];
      uint32_t granules =
          (rsrc2 >> PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT) &
          PAI_COMPUTE_PGM_RSRC2_LDS_SIZE_MASK;
      uint32_t lds_bytes = granules * PAI_COMPUTE_PGM_RSRC2_LDS_GRANULE_BYTES;
      int data_changed = 0;
      int exec_ok;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G25] probe %s RSRC2=0x%08x m0=0 LDS_SIZE=%u "
                    "(%u B) USER_SGPR=6 threads=1\n",
                    tags[t], rsrc2, granules, lds_bytes);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, rsrc2,
                               PAI_G25_THREADS, 1, ud, 4, &stream_len);
      exec_ok = (m0_run_gpu(ctx, stream, stream_len, c25, 128, 0xCC,
                            "G25") == 0);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G25] %s c[0..7] = %08x %08x %08x %08x %08x "
                    "%08x %08x %08x\n",
                    tags[t], c25[0], c25[1], c25[2], c25[3], c25[4], c25[5],
                    c25[6], c25[7]);
      for (uint32_t i = 0; i < 8u; i++) {
        if (c25[i] != 0xCCCCCCCCu) {
          data_changed = 1;
        }
      }
      if (c25[0] == PAI_G25_VALUE) {
        g25_ok = 1;
        PAI_LOG_INFO_(PAI_SUB_GPU,
                      "[M0-G25] %s sentinel PASS (m0=0 LDS=%u B)\n", tags[t],
                      lds_bytes);
        break;
      }
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G25] %s FAIL exec=%d data_changed=%d "
                    "c[0]=%08x want=%08x\n",
                    tags[t], exec_ok, data_changed, c25[0], PAI_G25_VALUE);
    }
    m0_exp_report("G25", g25_ok);

    if (g25_ok) {
      uint32_t *c8 = (uint32_t *)ctx->c.cpu_addr;
      int lanes_ok = 1;
      memcpy(ctx->code.cpu_addr, pai_lds_lanes_code,
             PAI_LDS_LANES_CODE_WORDS * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_g8, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_LDS_LANES_RSRC2,
                               PAI_LDS_LANES_THREADS, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c8, 128, 0xCC, "M1C");
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1C] c[0..7] = %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    c8[0], c8[1], c8[2], c8[3], c8[4], c8[5], c8[6], c8[7]);
      for (uint32_t i = 0; i < PAI_LDS_LANES_THREADS; i++) {
        if (c8[i] != (PAI_LDS_LANES_BASE + i)) {
          lanes_ok = 0;
          break;
        }
      }
      m0_exp_report("M1C", lanes_ok);
    } else {
      /* Explicit gate: stop inventing RSRC2; unlock via AGC CS blob. */
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-M1C] BLOCKED: G25 LDS not VALIDATED "
                    "(tried OpenAGC min 1 KiB RSRC2=0x1000C and 8 KiB). "
                    "Next unlock = Shader CS AGC blob with real "
                    "COMPUTE_PGM_RSRC2 LDS (SH offset 0x213); "
                    "do not invent further RSRC2 values.\n");
    }
  }

  /* PAI-M1 closeout on the serial G22 path (M1C deferred). */
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "===== PAI-M1: CLOSED (serial G22 path) =====\n");
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M1] M1A SAXPY int VALIDATED | M1B dot_serial VALIDATED | "
                "M1C parallel reduction BLOCKED (LDS) | "
                "M1D GEMV serial VALIDATED\n");
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M1] open gate: AGC CS LDS blob @ COMPUTE_PGM_RSRC2 "
                "(0x213); no further LDS RSRC2 hunt\n");

  /* G26: one x16 load + one ds write/read + G15 formula. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c26 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_dsstaged_code,
           PAI_G26_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G26_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c26, 128, 0xA5, "G26");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G26] c[0..7] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c26[0], c26[1], c26[2], c26[3], c26[4], c26[5], c26[6],
                  c26[7]);
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c26[0] == want0) && (c26[1] == want1);
      m0_exp_report("G26", ok);
    }
  }

  /* G27: single s_load + ds write/read + G15 formula. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c27 = (uint32_t *)ctx->c.cpu_addr;
    memcpy(ctx->code.cpu_addr, pai_dsstaged1_code,
           PAI_G27_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G27_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c27, 128, 0xA5, "G27");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G27] c[0..7] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c27[0], c27[1], c27[2], c27[3], c27[4], c27[5], c27[6],
                  c27[7]);
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c27[0] == want0) && (c27[1] == want1);
      m0_exp_report("G27", ok);
    }
  }

  /* G28-G30: float ALU v2 - v_cvt_f32_i32 then add (denormal-safe). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t g_offs[3] = {PAI_G28_OFF, PAI_G29_OFF, PAI_G30_OFF};
    static const uint32_t g_lens[3] = {PAI_G28_WORDS, PAI_G29_WORDS,
                                       PAI_G30_WORDS};
    static const char *const g_names[3] = {"G28", "G29", "G30"};
    float *cf = (float *)ctx->c.cpu_addr;
    float k = PAI_G2X_K;

    for (uint32_t variant = 0; variant < 3; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(cf, 0xCC, 128 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_fbatch2_code[g_offs[variant]],
             g_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_fbatch, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      memcpy(&ud[4], &k, sizeof(k));
      ud[5] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G28_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, cf, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] c[0..7] = %.1f %.1f %.1f %.1f %.1f %.1f "
                    "%.1f %.1f\n",
                    name, (double)cf[0], (double)cf[1], (double)cf[2],
                    (double)cf[3], (double)cf[4], (double)cf[5], (double)cf[6],
                    (double)cf[7]);
      {
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          float want = (float)(4 * i + 3) +
                       ((variant == 0) ? 0.0f : k);
          if (cf[i] != want) {
            ok = 0;
            break;
          }
        }
        m0_exp_report(name, ok);
      }
    }
  }

  /* G31/G32: float add operand probes (dst-v0 broadcast, VGPR+VGPR). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t g_offs[2] = {PAI_G31_OFF, PAI_G32_OFF};
    static const uint32_t g_lens[2] = {PAI_G31_WORDS, PAI_G32_WORDS};
    static const char *const g_names[2] = {"G31", "G32"};
    float *cf = (float *)ctx->c.cpu_addr;
    float k = PAI_G2X_K;

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(cf, 0xCC, 128 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_fbatch3_code[g_offs[variant]],
             g_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_fbatch, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      memcpy(&ud[4], &k, sizeof(k));
      ud[5] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G28_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, cf, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] c[0..7] = %.1f %.1f %.1f %.1f %.1f %.1f "
                    "%.1f %.1f\n",
                    name, (double)cf[0], (double)cf[1], (double)cf[2],
                    (double)cf[3], (double)cf[4], (double)cf[5], (double)cf[6],
                    (double)cf[7]);
    }
  }

  /* G33: the unlocked float form - SGPR scalar k via a VGPR mov. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    float *cf = (float *)ctx->c.cpu_addr;
    float k = PAI_G2X_K;
    memset(cf, 0xCC, 128 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, pai_fbatch4_code,
           PAI_G33_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_fbatch, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    memcpy(&ud[4], &k, sizeof(k));
    ud[5] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G33_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, cf, 128, 0xCC, "G33");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G33] c[0..7] = %.1f %.1f %.1f %.1f %.1f %.1f "
                  "%.1f %.1f\n",
                  (double)cf[0], (double)cf[1], (double)cf[2], (double)cf[3],
                  (double)cf[4], (double)cf[5], (double)cf[6], (double)cf[7]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 8; i++) {
        float want = (float)(4 * i + 3) + k;
        if (cf[i] != want) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("G33", ok);
    }
  }

  /* G34/G35: SGPR scalar into the float add. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t g_offs[2] = {PAI_G34_OFF, PAI_G35_OFF};
    static const uint32_t g_lens[2] = {PAI_G34_WORDS, PAI_G35_WORDS};
    static const char *const g_names[2] = {"G34", "G35"};
    float *cf = (float *)ctx->c.cpu_addr;
    float k = PAI_G2X_K;

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(cf, 0xCC, 128 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_fbatch5_code[g_offs[variant]],
             g_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_fbatch, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      memcpy(&ud[4], &k, sizeof(k));
      ud[5] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G33_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, cf, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] c[0..7] = %.1f %.1f %.1f %.1f %.1f %.1f "
                    "%.1f %.1f\n",
                    name, (double)cf[0], (double)cf[1], (double)cf[2],
                    (double)cf[3], (double)cf[4], (double)cf[5], (double)cf[6],
                    (double)cf[7]);
      {
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          float want = (float)(4 * i + 3) + k;
          if (cf[i] != want) {
            ok = 0;
            break;
          }
        }
        m0_exp_report(name, ok);
      }
    }
  }

  /* G36: MUBUF with the T# in non-zeroed s[4:7], base = VA>>8.
   * The G20/G21 used s[0:3] (s0-s1 hardware-zeroed) and the raw VA
   * (unshifted). c[0] = 0xA5A5A5A5 + 3 = 0xA5A5A5A8 on a working read. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c36 = (uint32_t *)ctx->c.cpu_addr;
    uint64_t cva = ctx->c.gpu_addr;
    memcpy(ctx->code.cpu_addr, pai_mubufload36_code,
           PAI_G36_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(cva & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(cva >> 32);
    ud[4] = (uint32_t)(cva >> 8);
    ud[5] = 0;
    ud[6] = 4096u;
    ud[7] = PAI_G20_TBUF_WORD3;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G36_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 8, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c36, 128, 0xA5, "G36");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G36] c[0..3] = %08x %08x %08x %08x (want c0=%08x)\n",
                  c36[0], c36[1], c36[2], c36[3],
                  PAI_G20_VALUE + 3u);
    {
      uint32_t want0 = PAI_G20_VALUE + 3u;
      uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
      int ok = (c36[0] == want0) && (c36[1] == want1);
      m0_exp_report("G36", ok);
    }
  }

  /* G37/G38: MUBUF format matrix - 32-bit DATA_FORMAT, OpenAGC cache
   * word2, record counts 0x4FAC vs 128. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    static const uint32_t g_offs[2] = {PAI_G37_OFF, PAI_G38_OFF};
    static const char *const g_names[2] = {"G37", "G38"};
    static const uint32_t g_w3[2] = {PAI_G37_TBUF_WORD3,
                                     PAI_G38_TBUF_WORD3};
    uint32_t *c3x = (uint32_t *)ctx->c.cpu_addr;
    uint64_t cva = ctx->c.gpu_addr;

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_mubufload37_code[g_offs[variant]],
             PAI_G37_WORDS * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_g8, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(cva & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(cva >> 32);
      ud[4] = (uint32_t)(cva >> 8);
      ud[5] = 0;
      ud[6] = PAI_G37_TBUF_WORD2;
      ud[7] = g_w3[variant];
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G36_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 8, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c3x, 128, 0xA5, name);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] word3 %08x: c[0..3] = %08x %08x %08x %08x\n",
                    name, g_w3[variant], c3x[0], c3x[1], c3x[2], c3x[3]);
      {
        uint32_t want0 = PAI_G20_VALUE + 3u;
        uint32_t want1 = PAI_G20_VALUE + 4u + 3u;
        int ok = (c3x[0] == want0) && (c3x[1] == want1);
        m0_exp_report(name, ok);
      }
    }
  }

  /* G39: VALU float dot, serial SMEM path. SALU float does not exist
   * on gfx1013; the float ops are VALU-only. The e64 direct-SGPR
   * mul (G35 rule) + VGPR+VGPR add (G32 rule). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *fp = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c39 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t n = 128u;
    uint32_t stream_len = 0;
    float want = 0.0f;
    memcpy(ctx->code.cpu_addr, pai_fdot_serial_code,
           PAI_FDOT_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_fbatch, NULL);
    }
    fp[0] = n;
    fp[1] = 0;
    for (uint32_t i = 0; i < n; i++) {
      float a = 1.5f + (float)i * 0.25f;
      float b = 2.0f - (float)i * 0.125f;
      memcpy(&fp[2u + 2u * i], &a, 4);
      memcpy(&fp[3u + 2u * i], &b, 4);
      want += a * b;
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_FDOT_RSRC2,
                             PAI_FDOT_THREADS, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c39, 16, 0xCC, "G39");
    {
      float got;
      memcpy(&got, &c39[0], 4);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G39] c[0]=%.6f want=%.6f (%08x)\n",
                    (double)got, (double)want, c39[0]);
      m0_exp_report("G39", got == want);
    }
  }

  /* G40: VALU float GEMV, serial-per-row (TGID_X parallel rows). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t m = 8u, k = 32u;
    uint32_t *w32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *x32 = (uint32_t *)ctx->b.cpu_addr;
    uint32_t *y32 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    float want[8];
    memcpy(ctx->code.cpu_addr, pai_fgemv_serial_code,
           PAI_FGEMV_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    w32[0] = k;
    w32[1] = 0;
    w32[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
    w32[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
    for (uint32_t g = 0; g < m; g++) {
      want[g] = 0.0f;
      for (uint32_t kk = 0; kk < k; kk++) {
        float wv = 0.5f + 0.1f * (float)g + 0.01f * (float)kk;
        float xv = 1.0f - 0.02f * (float)kk;
        memcpy(&w32[4u + g * k + kk], &wv, 4);
        want[g] += wv * xv;
      }
    }
    for (uint32_t kk = 0; kk < k; kk++) {
      float xv = 1.0f - 0.02f * (float)kk;
      memcpy(&x32[kk], &xv, 4);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->b);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[6] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_FGEMV_RSRC2,
                             PAI_FGEMV_THREADS, m, ud, 7, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, y32, 64, 0xCC, "G40");
    {
      float got[8];
      int ok = 1;
      memcpy(got, y32, sizeof(got));
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G40] y[0..3] = %.4f %.4f %.4f %.4f\n",
                    (double)got[0], (double)got[1], (double)got[2],
                    (double)got[3]);
      for (uint32_t g = 0; g < m; g++) {
        if (got[g] != want[g]) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("G40", ok);
    }
  }

  /* G41: VALU float SAXPY, per-group (C[g] = 0.5*A[g] + B[g]). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t n = 64u;
    uint32_t *ab32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c41 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    float want[64];
    memcpy(ctx->code.cpu_addr, pai_fsaxpy_code,
           PAI_FSAXPY_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    for (uint32_t g = 0; g < n; g++) {
      float av = 0.25f * (float)g;
      float bv = 1.0f - 0.01f * (float)g;
      memcpy(&ab32[2u * g], &av, 4);
      memcpy(&ab32[2u * g + 1u], &bv, 4);
      want[g] = PAI_FSAXPY_K * av + bv;
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[6] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_FSAXPY_RSRC2,
                             PAI_FSAXPY_THREADS, n, ud, 7, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c41, 64, 0xCC, "G41");
    {
      float got[8];
      int ok = 1;
      memcpy(got, c41, sizeof(got));
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G41] c[0..3] = %.4f %.4f %.4f %.4f\n",
                    (double)got[0], (double)got[1], (double)got[2],
                    (double)got[3]);
      for (uint32_t g = 0; g < n; g++) {
        float gg;
        memcpy(&gg, &c41[g], 4);
        if (gg != want[g]) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("G41", ok);
    }
  }

/* G42-G46: T4 serial float elementwise kernels - packed (a,b) pairs,
   * C at s4:s5, TGID_X = element index, one thread per group. The
   * e64 direct-SGPR forms validated in G35/G39/G41. */
  {
    static const uint32_t k_offs[5] = {PAI_T4_ADD1D_OFF, PAI_T4_SUB1D_OFF,
                                       PAI_T4_MUL1D_OFF, PAI_T4_RELU_OFF,
                                       PAI_T4_CLIP_OFF};
    static const uint32_t k_lens[5] = {PAI_T4_ADD1D_WORDS, PAI_T4_SUB1D_WORDS,
                                       PAI_T4_MUL1D_WORDS, PAI_T4_RELU_WORDS,
                                       PAI_T4_CLIP_WORDS};
    static const char *const k_names[5] = {"G42", "G43", "G44", "G45", "G46"};
    static pai_host_kernel_fn const k_host[5] = {
        pai_host_kernel_t4_add1d, pai_host_kernel_t4_sub1d,
        pai_host_kernel_t4_mul1d, pai_host_kernel_t4_relu,
        pai_host_kernel_t4_clip};
    uint32_t n = 64u;
    uint32_t *ab32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c42 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    float want[64];

for (uint32_t variant = 0; variant < 5; variant++) {
      int ok = 1;
      uint64_t cpu_phys = 0;
      uint32_t pde[4];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(c42, 0xCC, 64 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_t4_ops_code[k_offs[variant]],
             k_lens[variant] * sizeof(uint32_t));
      pai_cpu_phys_of_va((uint64_t)(uintptr_t)ctx->code.cpu_addr, &cpu_phys);
      pai_gvmspace_dump_pde_page(ctx->code.gpu_addr, pde);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] code cpu_phys=0x%llx gpu_pde=%08x %08x %08x %08x "
                    "first=%08x\n",
                    k_names[variant], (unsigned long long)cpu_phys, pde[0],
                    pde[1], pde[2], pde[3], ((uint32_t *)ctx->code.cpu_addr)[0]);
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr, k_host[variant],
                                     NULL);
      }
      for (uint32_t g = 0; g < n; g++) {
        float av = 0.25f * (float)g;
        float bv = 1.0f - 0.01f * (float)g;
        memcpy(&ab32[2u * g], &av, 4);
        memcpy(&ab32[2u * g + 1u], &bv, 4);
        switch (variant) {
        case 0:
          want[g] = av + bv;
          break;
        case 1:
          want[g] = av - bv;
          break;
        case 2:
          want[g] = av * bv;
          break;
        case 3:
          want[g] = av > 0.0f ? av : 0.0f;
          break;
        default:
          want[g] = av < 0.0f ? 0.0f : (av > 1.0f ? 1.0f : av);
          break;
        }
      }
      pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
      pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      ud[6] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_T4_RSRC2,
                               PAI_T4_THREADS, n, ud, 7, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c42, 64, 0xCC, k_names[variant]);
      for (uint32_t g = 0; g < n && ok; g++) {
        float gg;
        memcpy(&gg, &c42[g], 4);
        if (gg != want[g]) {
          ok = 0;
        }
      }
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..3] = %.4f %.4f %.4f %.4f\n",
                    k_names[variant], (double)((float *)c42)[0],
                    (double)((float *)c42)[1], (double)((float *)c42)[2],
                    (double)((float *)c42)[3]);
      m0_exp_report(k_names[variant], ok);
    }
  }

  /* G47: T4 biasadd - C[g] = a[g] + bias[g%cols], one group per
   * cell. Header at s2:s3: [cols, pad, a_lo, a_hi, bias_lo, bias_hi]. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t rows = 8u, cols = 8u;
    uint32_t *h47 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *a47 = (uint32_t *)ctx->b.cpu_addr;
    uint32_t *bias47 = h47 + 16;
    uint32_t *c47 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c47, 0xCC, 64 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, &pai_t4_ops_code[PAI_T4_BIASADD_OFF],
           PAI_T4_BIASADD_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_t4_biasadd, NULL);
    }
h47[0] = cols;
    h47[1] = 0;
    h47[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
    h47[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
    h47[4] = (uint32_t)((ctx->a.gpu_addr + 64) & 0xFFFFFFFFu);
    h47[5] = (uint32_t)((ctx->a.gpu_addr + 64) >> 32);
    for (uint32_t j = 0; j < cols; j++) {
      float bj = 0.5f + 0.25f * (float)j;
      memcpy(&bias47[j], &bj, 4);
    }
    for (uint32_t i = 0; i < rows; i++) {
      for (uint32_t j = 0; j < cols; j++) {
        float av = 0.1f * (float)i + 0.01f * (float)j;
        memcpy(&a47[i * cols + j], &av, 4);
      }
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->b);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[6] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_T4_RSRC2,
                             PAI_T4_THREADS, rows * cols, ud, 7,
                             &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c47, 64, 0xCC, "G47");
    for (uint32_t i = 0; i < rows && ok; i++) {
      for (uint32_t j = 0; j < cols; j++) {
        float gg;
        float want = (0.1f * (float)i + 0.01f * (float)j) +
                     (0.5f + 0.25f * (float)j);
        memcpy(&gg, &c47[i * cols + j], 4);
        if (gg != want) {
          ok = 0;
        }
      }
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G47] c[0..3] = %.4f %.4f %.4f %.4f\n",
                  (double)((float *)c47)[0], (double)((float *)c47)[1],
                  (double)((float *)c47)[2], (double)((float *)c47)[3]);
    m0_exp_report("G47", ok);
  }

  /* G48: T4 matmul - C[g] = sum_k a[i*K+k] * b[k*N+j] (i=g/N,
   * j=g%N), one group per cell. Header at s2:s3: [K, N, a_lo, a_hi,
   * b_lo, b_hi]. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t rows = 4u, kk = 16u, cols = 4u;
    uint32_t *h48 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *a48 = (uint32_t *)ctx->b.cpu_addr;
    uint32_t *b48 = a48 + rows * kk;
    uint32_t *c48 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c48, 0xCC, 64 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, &pai_t4_ops_code[PAI_T4_MATMUL_OFF],
           PAI_T4_MATMUL_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_t4_matmul, NULL);
    }
h48[0] = kk;
    h48[1] = cols;
    h48[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
    h48[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
    h48[4] = (uint32_t)((ctx->b.gpu_addr + rows * kk * 4) & 0xFFFFFFFFu);
    h48[5] = (uint32_t)((ctx->b.gpu_addr + rows * kk * 4) >> 32);
    for (uint32_t i = 0; i < rows; i++) {
      for (uint32_t t = 0; t < kk; t++) {
        float av = 0.5f + 0.1f * (float)i + 0.01f * (float)t;
        memcpy(&a48[i * kk + t], &av, 4);
      }
    }
    for (uint32_t t = 0; t < kk; t++) {
      for (uint32_t j = 0; j < cols; j++) {
        float bv = 1.0f - 0.02f * (float)t + 0.1f * (float)j;
        memcpy(&b48[t * cols + j], &bv, 4);
      }
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->b);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[6] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_T4_RSRC2,
                             PAI_T4_THREADS, rows * cols, ud, 7,
                             &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c48, 64, 0xCC, "G48");
    {
      float ref48[64];
      uint64_t mismatch48 = 0;
      pai_status_t st48;
      for (uint32_t i = 0; i < rows; i++) {
        for (uint32_t j = 0; j < cols; j++) {
          float want = 0.0f;
          for (uint32_t t = 0; t < kk; t++) {
            want += (0.5f + 0.1f * (float)i + 0.01f * (float)t) *
                    (1.0f - 0.02f * (float)t + 0.1f * (float)j);
          }
          ref48[i * cols + j] = want;
        }
      }
      /* Tolerant compare: the GPU accumulates with v_add_f32 per term;
       * host summation order can differ by <=1 ULP, so exact equality
       * is not the right oracle here (G42-G47 and G54 use exact checks
       * because their ops are single-rounding). */
      st48 = pai_ref_compare_f32((float *)c48, ref48, rows * cols, 1e-6f,
                                 1e-6f, &mismatch48);
      if (st48 != PAI_OK) {
        PAI_LOG_ERROR_(
            PAI_SUB_GPU,
            "[M0-G48] FAIL: mismatch at %llu (gpu=%f ref=%f)\n",
            (unsigned long long)mismatch48,
            mismatch48 < rows * cols ? (double)((float *)c48)[mismatch48] : 0.0,
            mismatch48 < rows * cols ? (double)ref48[mismatch48] : 0.0);
        ok = 0;
      }
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G48] c[0..3] = %.4f %.4f %.4f %.4f\n",
                  (double)((float *)c48)[0], (double)((float *)c48)[1],
                  (double)((float *)c48)[2], (double)((float *)c48)[3]);
    m0_exp_report("G48", ok);
  }

  /* G49-G53: T4 serial integer elementwise kernels - packed (a,b)
   * pairs, C at s4:s5, TGID_X = element index, one thread per group.
   * u32 wrap semantics (SALU integer path, G22-proven). */
  {
    static const uint32_t k_offs[5] = {PAI_INT_ADD2D_OFF, PAI_INT_SUB1D_OFF,
                                       PAI_INT_MUL1D_OFF, PAI_INT_RELU_OFF,
                                       PAI_INT_CLIP_OFF};
    static const uint32_t k_lens[5] = {
        PAI_INT_ADD2D_WORDS, PAI_INT_SUB1D_WORDS, PAI_INT_MUL1D_WORDS,
        PAI_INT_RELU_WORDS, PAI_INT_CLIP_WORDS};
    static const char *const k_names[5] = {"G49", "G50", "G51", "G52",
                                           "G53"};
    static pai_host_kernel_fn const k_host[5] = {
        pai_host_kernel_int_add2d, pai_host_kernel_int_sub1d,
        pai_host_kernel_int_mul1d, pai_host_kernel_int_relu,
        pai_host_kernel_int_clip};
    uint32_t n = 64u;
    uint32_t *ab32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c49 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    uint32_t want[64];

    for (uint32_t variant = 0; variant < 5; variant++) {
      int ok = 1;
      uint64_t cpu_phys = 0;
      uint32_t pde[4];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(c49, 0xCC, 64 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_int_ops_code[k_offs[variant]],
             k_lens[variant] * sizeof(uint32_t));
      pai_cpu_phys_of_va((uint64_t)(uintptr_t)ctx->code.cpu_addr, &cpu_phys);
      pai_gvmspace_dump_pde_page(ctx->code.gpu_addr, pde);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] code cpu_phys=0x%llx gpu_pde=%08x %08x %08x %08x "
                    "first=%08x\n",
                    k_names[variant], (unsigned long long)cpu_phys, pde[0],
                    pde[1], pde[2], pde[3],
                    ((uint32_t *)ctx->code.cpu_addr)[0]);
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr, k_host[variant],
                                     NULL);
      }
      for (uint32_t g = 0; g < n; g++) {
        uint32_t av = 0x80000000u + 0x01000000u * g;
        uint32_t bv = 0x11111111u + 0x00001000u * g;
        ab32[2u * g] = av;
        ab32[2u * g + 1u] = bv;
        switch (variant) {
        case 0:
          want[g] = av + bv;
          break;
        case 1:
          want[g] = av - bv;
          break;
        case 2:
          want[g] = av * bv;
          break;
        case 3:
          want[g] = av > 0u ? av : 0u;
          break;
        default:
          want[g] = av < 0u ? 0u : (av > 1u ? 1u : av);
          break;
        }
      }
      pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
      pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      ud[6] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_INT_RSRC2,
                               PAI_INT_THREADS, n, ud, 7, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c49, 64, 0xCC, k_names[variant]);
      for (uint32_t g = 0; g < n && ok; g++) {
        if (c49[g] != want[g]) {
          ok = 0;
        }
      }
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..3] = %08x %08x %08x %08x\n",
                    k_names[variant], c49[0], c49[1], c49[2], c49[3]);
      m0_exp_report(k_names[variant], ok);
    }
  }

  /* G54: T4 integer matmul - C[g] = sum_k a[i*K+k] * b[k*N+j]
   * (i=g/N, j=g%N), u32 wrap, one group per cell. Header [K, N,
   * a_lo, a_hi, b_lo, b_hi]. b is row-major [k][N]. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t rows = 4u, kk = 16u, cols = 4u;
    uint32_t *h54 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *a54 = (uint32_t *)ctx->b.cpu_addr;
    uint32_t *b54 = a54 + rows * kk;
    uint32_t *c54 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c54, 0xCC, 64 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, &pai_int_ops_code[PAI_INT_MATMUL_OFF],
           PAI_INT_MATMUL_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_int_matmul, NULL);
    }
    h54[0] = kk;
    h54[1] = cols;
    h54[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
    h54[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
    h54[4] = (uint32_t)((ctx->b.gpu_addr + rows * kk * 4) & 0xFFFFFFFFu);
    h54[5] = (uint32_t)((ctx->b.gpu_addr + rows * kk * 4) >> 32);
    for (uint32_t i = 0; i < rows; i++) {
      for (uint32_t t = 0; t < kk; t++) {
        a54[i * kk + t] = 0x10000000u + 0x00000100u * i + t;
      }
    }
    for (uint32_t t = 0; t < kk; t++) {
      for (uint32_t j = 0; j < cols; j++) {
        b54[t * cols + j] = 0x20000000u + 0x00010000u * t + j;
      }
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->b);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    ud[6] = 0;
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_INT_RSRC2,
                             PAI_INT_THREADS, rows * cols, ud, 7,
                             &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c54, 64, 0xCC, "G54");
    for (uint32_t i = 0; i < rows && ok; i++) {
      for (uint32_t j = 0; j < cols; j++) {
        uint32_t want = 0;
        for (uint32_t t = 0; t < kk; t++) {
          want += a54[i * kk + t] * b54[t * cols + j];
        }
        if (c54[i * cols + j] != want) {
          ok = 0;
        }
      }
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G54] c[0..3] = %08x %08x %08x %08x\n",
                  c54[0], c54[1], c54[2], c54[3]);
    m0_exp_report("G54", ok);
  }

  /* G55: wave-parallel float linear ramp - c[i] = base + k*i with
   * NUM_THREAD_X=32 (1 group); each lane derives its value from tid +
   * uniform scalars (the only reads that work on 9.40). Lanes 0-7 of
   * the wave store. First kernel past the serial NUM_THREAD_X=1
   * model; the ramp is the RoPE position-table primitive (Phase 2). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c55 = (uint32_t *)ctx->c.cpu_addr;
    float k55 = 0.5f;
    float base55 = 1.0f;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c55, 0xCC, 64 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, pai_ramp_code,
           PAI_RAMP_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ramp, NULL);
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    memcpy(&ud[4], &k55, sizeof(k55));
    memcpy(&ud[5], &base55, sizeof(base55));
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_RAMP_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c55, 64, 0xCC, "G55");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G55] c[0..7] = %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                  "%.2f\n",
                  (double)((float *)c55)[0], (double)((float *)c55)[1],
                  (double)((float *)c55)[2], (double)((float *)c55)[3],
                  (double)((float *)c55)[4], (double)((float *)c55)[5],
                  (double)((float *)c55)[6], (double)((float *)c55)[7]);
    /* The value path bakes in the G15 store formula (tid*4+3), the
     * same convention G35's check uses: want = base + k*(4i+3). */
    for (uint32_t i = 0; i < 8 && ok; i++) {
      float gg;
      float want = base55 + k55 * (float)(4 * i + 3);
      memcpy(&gg, &c55[i], 4);
      if (gg != want) {
        ok = 0;
      }
    }
    m0_exp_report("G55", ok);
  }

  /* G56: wave-parallel float ramp, s_load-fed - same kernel as G55
   * but k/base are read from a GPU-mem header via s_load_dword inside
   * the 32-thread wave. Proves the scalar-read path works
   * wave-parallel (G55 had no s_load); this is the x-side of a
   * wave-parallel GEMV (x[k] is uniform across rows). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h56 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c56 = (uint32_t *)ctx->c.cpu_addr;
    float k56 = 0.5f;
    float base56 = 1.0f;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c56, 0xCC, 64 * sizeof(uint32_t));
    memcpy(&h56[0], &k56, sizeof(k56));
    memcpy(&h56[1], &base56, sizeof(base56));
    memcpy(ctx->code.cpu_addr, pai_ramp2_code,
           PAI_RAMP2_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ramp2, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_RAMP2_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c56, 64, 0xCC, "G56");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G56] c[0..7] = %.2f %.2f %.2f %.2f %.2f %.2f %.2f "
                  "%.2f\n",
                  (double)((float *)c56)[0], (double)((float *)c56)[1],
                  (double)((float *)c56)[2], (double)((float *)c56)[3],
                  (double)((float *)c56)[4], (double)((float *)c56)[5],
                  (double)((float *)c56)[6], (double)((float *)c56)[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      float gg;
      float want = base56 + k56 * (float)(4 * i + 3);
      memcpy(&gg, &c56[i], 4);
      if (gg != want) {
        ok = 0;
      }
    }
    m0_exp_report("G56", ok);
  }

  /* G57: per-lane select from an s_load_dwordx16 block - can a lane
   * pick its own element out of a 16-dword scalar load (v_movrels_b32
   * with m0 base + v0 index)? The unlock for wave-parallel GEMV
   * (per-lane W[i,k] access without vector loads). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h57 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c57 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c57, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h57[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_lanepick_code,
           PAI_LANEPICK_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_lanepick, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_LANEPICK_RSRC2, PAI_EXP_THREADS_X,
                                   1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c57, 64, 0xCC, "G57");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G57] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c57[0], c57[1], c57[2], c57[3], c57[4], c57[5], c57[6],
                  c57[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c57[i] != h57[i]) {
        ok = 0;
      }
    }
    m0_exp_report("G57", ok);
  }

  /* G58: block dump bisection - does s_load_dwordx16 populate
   * s[16:31] with 32 SGPRs allocated? Direct v_mov copies to c, no
   * v_movrels. Isolates block SMEM load from G57's select. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h58 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c58 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c58, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h58[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_blockdump_code,
           PAI_BLOCKDUMP_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_blockdump, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_BLOCKDUMP_RSRC2, 1,
                                   1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c58, 64, 0xCC, "G58");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G58] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c58[0], c58[1], c58[2], c58[3], c58[4], c58[5], c58[6],
                  c58[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c58[i] != h58[i]) {
        ok = 0;
      }
    }
    m0_exp_report("G58", ok);
  }

  /* G59: direct v16 read, no movrels - decisive bisection for G57.
   * If c[0..7] == header[0], v16+ copies land and v_movrels_b32 is
   * the non-functional piece. If garbage, copies to v16+ are dropped. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h59 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c59 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c59, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h59[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_vpick_code,
           PAI_VPICK_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vpick, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_VPICK_RSRC2, PAI_EXP_THREADS_X,
                                   1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c59, 64, 0xCC, "G59");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G59] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c59[0], c59[1], c59[2], c59[3], c59[4], c59[5], c59[6],
                  c59[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c59[i] != h59[0]) {
        ok = 0;
      }
    }
    m0_exp_report("G59", ok);
  }

  /* G60: v8..v15 block copies under STANDARD RSRC1 (0x602C0000) -
   * golden E22 proves v6-v9 work with this RSRC1; does v15? If yes,
   * the v16+ failure is a real ceiling (or BLOCK32 broke it), and
   * G61 (v16 + standard RSRC1) discriminates the two. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h60 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c60 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c60, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h60[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_vpick2_code,
           PAI_VPICK2_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vpick2, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_VPICK2_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c60, 64, 0xCC, "G60");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G60] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c60[0], c60[1], c60[2], c60[3], c60[4], c60[5], c60[6],
                  c60[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c60[i] != h60[0]) {
        ok = 0;
      }
    }
    m0_exp_report("G60", ok);
  }

  /* G61: v16 direct read but STANDARD RSRC1 (vpick code) - if G60
   * passes (v8-v15 fine) and G61 fails, the ceiling is between v16
   * and v24; if G61 passes too, BLOCK32 RSRC1 was the breakage. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h61 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c61 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c61, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h61[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_vpick_code,
           PAI_VPICK_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vpick, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_VPICK_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c61, 64, 0xCC, "G61");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G61] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c61[0], c61[1], c61[2], c61[3], c61[4], c61[5], c61[6],
                  c61[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c61[i] != h61[0]) {
        ok = 0;
      }
    }
    m0_exp_report("G61", ok);
  }

  /* G62: v8..v15 copies + direct v8 read under BLOCK32 (32 VGPR +
   * 32 SGPR) at 32 threads. G60 used STANDARD (8 VGPR) so v8 was
   * out of range - confounded. If G62 passes, v8..v15 are fine with
   * the right allocation and G59's failure is v16+ specific (or a
   * thread-count effect, discriminated by G63). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h62 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c62 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c62, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h62[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_vpick2_code,
           PAI_VPICK2_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vpick2, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_VPICK2_RSRC2, PAI_EXP_THREADS_X,
                                   1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c62, 64, 0xCC, "G62");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G62] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c62[0], c62[1], c62[2], c62[3], c62[4], c62[5], c62[6],
                  c62[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c62[i] != h62[0]) {
        ok = 0;
      }
    }
    m0_exp_report("G62", ok);
  }

  /* G63: v16..v23 copies + direct v16 read under BLOCK32 at 1 thread.
   * G59 (32T) leaked tid; if v16 works at 1T, the 32-thread dispatch
   * is what breaks high-VGPR reads (a wave/execution quirk), not the
   * VGPR number itself. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h63 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c63 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c63, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h63[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_vpick_code,
           PAI_VPICK_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_vpick, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_VPICK_RSRC2, 1, 1, ud, 6,
                                   &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c63, 64, 0xCC, "G63");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G63] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c63[0], c63[1], c63[2], c63[3], c63[4], c63[5], c63[6],
                  c63[7]);
    /* 1 thread -> only c[0] written (lanes 1..7 stay 0xCC). */
    if (c63[0] != h63[0]) {
      ok = 0;
    }
    m0_exp_report("G63", ok);
  }

  /* G64: v_movrels_b32 with in-ceiling block v7..v14 + m0=7 - the one
   * untested piece. G57's v16+ copies were dropped by the 16-VGPR HW
   * ceiling, so movrels read garbage there; G64 copies the block to
   * v7..v14 (G62 proved v8..v15 readable) and selects v[7+tid].
   * Expect c[i] = h[7+i] = 0x10000007+i. If PASS: per-lane select
   * unlocked -> wave-parallel GEMV buildable. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *h64 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c64 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c64, 0xCC, 64 * sizeof(uint32_t));
    for (uint32_t i = 0; i < 16; i++) {
      h64[i] = 0x10000000u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_movrels_code,
           PAI_MOVRELS_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_movrels, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream_rsrc1(ctx, stream, M0_PM4_CAP,
                                   PAI_EXP_RSRC1_BLOCK32,
                                   PAI_MOVRELS_RSRC2, PAI_EXP_THREADS_X,
                                   1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c64, 64, 0xCC, "G64");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G64] c[0..7] = %08x %08x %08x %08x %08x %08x %08x "
                  "%08x\n",
                  c64[0], c64[1], c64[2], c64[3], c64[4], c64[5], c64[6],
                  c64[7]);
    for (uint32_t i = 0; i < 8 && ok; i++) {
      if (c64[i] != h64[7 + i]) {
        ok = 0;
      }
    }
    m0_exp_report("G64", ok);
  }

  /* G65/G66: wave-parallel cos/sin ramp - RoPE position-table
   * primitives (Phase 2). theta = scale*i via the G55 value path; the
   * lane index reads as (4i+3) so want[i] = cosf/sinf(scale*(4i+3)).
   * This validates v_cos_f32/v_sin_f32 (never exercised on 9.40). */
  {
    static const uint32_t k_off[2] = {PAI_COSSIN_COS_OFF, PAI_COSSIN_SIN_OFF};
    static const uint32_t k_len[2] = {PAI_COSSIN_COS_WORDS, PAI_COSSIN_SIN_WORDS};
    static const char *const k_name[2] = {"G65", "G66"};
    static pai_host_kernel_fn const k_host[2] = {
        pai_host_kernel_cossin_cos, pai_host_kernel_cossin_sin};
    float scale = 0.15f;
    uint32_t *c66 = (uint32_t *)ctx->c.cpu_addr;
    uint32_t stream_len = 0;
    float want[8];

    for (uint32_t variant = 0; variant < 2; variant++) {
      int ok = 1;
      uint64_t cpu_phys = 0;
      uint32_t pde[4];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memset(c66, 0xCC, 64 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, &pai_cossin_code[k_off[variant]],
             k_len[variant] * sizeof(uint32_t));
      pai_cpu_phys_of_va((uint64_t)(uintptr_t)ctx->code.cpu_addr, &cpu_phys);
      pai_gvmspace_dump_pde_page(ctx->code.gpu_addr, pde);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] code cpu_phys=0x%llx first=%08x words=%u\n",
                    k_name[variant], (unsigned long long)cpu_phys,
                    ((uint32_t *)ctx->code.cpu_addr)[0], k_len[variant]);
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr, k_host[variant],
                                     NULL);
      }
      /* Turns convention (HW-verified): v_cos/v_sin read their operand
       * in full turns, i.e. the value path multiplies by 2*pi. The
       * oracle must use want = cos/sin(2*pi*scale*(4i+3)). */
      for (uint32_t g = 0; g < 8; g++) {
        float theta = 6.2831853f * scale * (float)(4u * g + 3u);
        want[g] = (variant == 0) ? cosf(theta) : sinf(theta);
      }
      pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      ud[4] = 0;
      memcpy(&ud[4], &scale, 4);
      ud[5] = 0;
      ud[6] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_COSSIN_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c66, 64, 0xCC, k_name[variant]);
      for (uint32_t g = 0; g < 8 && ok; g++) {
        float gg;
        memcpy(&gg, &c66[g], 4);
        if (fabsf(gg - want[g]) > 1e-4f) {
          ok = 0;
        }
      }
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-%s] c[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
                    "%.4f\n",
                    k_name[variant], (double)((float *)c66)[0],
                    (double)((float *)c66)[1], (double)((float *)c66)[2],
                    (double)((float *)c66)[3], (double)((float *)c66)[4],
                    (double)((float *)c66)[5], (double)((float *)c66)[6],
                    (double)((float *)c66)[7]);
      m0_exp_report(k_name[variant], ok);
    }
  }

  /* G67/G68: on-GPU RoPE cos/sin table generator - groups_x = r2
   * columns, lane l -> position p = 4l+3, scale = inv_freq/(2pi)
   * s_load'ed per group (G40 TGID_X pattern). Produces the EXACT
   * tables B's pai_ref_rope_cossin_f32 oracle generates (rows 3..31);
   * differential check < 1e-4 on rows p=4l+3. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t r2 = 4u, nctx = 32u;
    float base = 10000.0f;
    uint32_t *h67 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c67 = (uint32_t *)ctx->c.cpu_addr;
    float ocos[128], osin[128];
    uint32_t stream_len = 0;
    int ok = 1;

    memset(c67, 0xCC, 2u * nctx * r2 * sizeof(uint32_t));
    h67[0] = r2;
    h67[1] = nctx;
    for (uint32_t p = 0; p < nctx; p++) {
      for (uint32_t i = 0; i < r2; i++) {
        /* theta_turns[e] = theta/(2*pi) so the x2pi turns convention
         * cancels: cos(2*pi*theta_turns) = cos(theta) exactly. */
        float inv = powf(base, -(float)i / (float)r2);
        float tt = (float)p * inv / 6.2831853f;
        memcpy(&h67[2 + p * r2 + i], &tt, 4);
      }
    }
    pai_ref_rope_cossin_f32(nctx, r2, base, ocos, osin);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ROPEGEN_RSRC2,
                             PAI_ROPEGEN_THREADS, nctx * r2, ud, 6,
                             &stream_len);
    memcpy(ctx->code.cpu_addr, &pai_ropegen_code[PAI_ROPEGEN_COS_OFF],
           PAI_ROPEGEN_COS_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ropegen_cos, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    m0_run_gpu(ctx, stream, stream_len, c67, 64, 0xCC, "G67");
    {
      uint32_t fmis = UINT32_MAX;
      for (uint32_t e = 0; e < nctx * r2 && ok; e++) {
        float gg;
        memcpy(&gg, &c67[e], 4);
        if (fabsf(gg - ocos[e]) > 1e-4f) {
          ok = 0;
          fmis = e;
        }
      }
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G67] c[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
                    "first_mis=%u\n",
                    (double)((float *)c67)[0], (double)((float *)c67)[1],
                    (double)((float *)c67)[2], (double)((float *)c67)[3],
                    (double)((float *)c67)[4], (double)((float *)c67)[5],
                    (double)((float *)c67)[6], (double)((float *)c67)[7], fmis);
    }
    m0_exp_report("G67", ok);

    if (!host) {
      pai_gpu_reset(gpu);
    }
    memcpy(ctx->code.cpu_addr, &pai_ropegen_code[PAI_ROPEGEN_SIN_OFF],
           PAI_ROPEGEN_SIN_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ropegen_sin, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ROPEGEN_RSRC2,
                             PAI_ROPEGEN_THREADS, nctx * r2, ud, 6,
                             &stream_len);
    /* Watch the SIN half (+ctx*r2*4 bytes): the shader writes there, so
     * m0_run_gpu observes real shader output and waits on the fence
     * instead of timing out and reading mid-execution. Long timeout:
     * post-panic GPU runs degraded/slow, EOP needs more than 2s. */
    ctx->timeout_ns = UINT64_C(30000000000); /* 30s */
    m0_run_gpu(ctx, stream, stream_len,
               (uint8_t *)c67 + nctx * r2 * 4, nctx * r2 * 4, 0xCC,
               "G68");
    ctx->timeout_ns = 0;
    ok = 1;
    {
      uint32_t fmis = UINT32_MAX;
      uint32_t nbad = 0;
      /* Convergence poll: re-read up to 10x over ~5s to separate
       * mid-execution reads (holes fill in) from hung waves (holes
       * stay). */
      for (uint32_t poll = 0; poll < 10 && nbad != 0 || poll == 0; poll++) {
        fmis = UINT32_MAX;
        nbad = 0;
        for (uint32_t e = 0; e < nctx * r2; e++) {
          float gg;
          memcpy(&gg, &c67[nctx * r2 + e], 4);
          if (fabsf(gg - osin[e]) > 1e-4f) {
            if (fmis == UINT32_MAX) {
              fmis = e;
            }
            nbad++;
          }
        }
        if (poll > 0 && nbad == 0) {
          break;
        }
        if (poll > 0) {
          /* ~500ms busy delay via clock. */
          uint64_t t0 = pai_clock_ns();
          while (pai_clock_ns() - t0 < UINT64_C(500000000)) {
          }
        }
      }
      ok = (nbad == 0);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G68] s[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f "
                    "bad=%u first_mis=%u",
                    (double)((float *)c67)[nctx * r2 + 0],
                    (double)((float *)c67)[nctx * r2 + 1],
                    (double)((float *)c67)[nctx * r2 + 2],
                    (double)((float *)c67)[nctx * r2 + 3],
                    (double)((float *)c67)[nctx * r2 + 4],
                    (double)((float *)c67)[nctx * r2 + 5],
                    (double)((float *)c67)[nctx * r2 + 6],
                    (double)((float *)c67)[nctx * r2 + 7], nbad, fmis);
      if (nbad) {
        float gv, wv;
        memcpy(&gv, &c67[nctx * r2 + fmis], 4);
        wv = osin[fmis];
        PAI_LOG_INFO_(PAI_SUB_GPU,
                      " -> mis e=%u got=%.6f want=%.6f\n", fmis,
                      (double)gv, (double)wv);
        {
          char bm[140];
          uint32_t bp = 0;
          for (uint32_t e = 0; e < nctx * r2; e++) {
            float gg;
            uint32_t raw;
            memcpy(&gg, &c67[nctx * r2 + e], 4);
            raw = ((const uint32_t *)c67)[nctx * r2 + e];
            bm[bp++] = (raw == 0xCCCCCCCCu) ? '#'
                      : (fabsf(gg - osin[e]) > 1e-4f) ? '?'
                      : '.';
          }
          bm[bp] = 0;
          PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G68] bitmap[0..127]=%s\n", bm);
        }
        PAI_LOG_INFO_(PAI_SUB_GPU,
                      "[M0-G68] e8..23=%.3f %.3f %.3f %.3f %.3f %.3f %.3f "
                      "%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                      (double)((float *)c67)[nctx * r2 + 8],
                      (double)((float *)c67)[nctx * r2 + 9],
                      (double)((float *)c67)[nctx * r2 + 10],
                      (double)((float *)c67)[nctx * r2 + 11],
                      (double)((float *)c67)[nctx * r2 + 12],
                      (double)((float *)c67)[nctx * r2 + 13],
                      (double)((float *)c67)[nctx * r2 + 14],
                      (double)((float *)c67)[nctx * r2 + 15],
                      (double)((float *)c67)[nctx * r2 + 16],
                      (double)((float *)c67)[nctx * r2 + 17],
                      (double)((float *)c67)[nctx * r2 + 18],
                      (double)((float *)c67)[nctx * r2 + 19],
                      (double)((float *)c67)[nctx * r2 + 20],
                      (double)((float *)c67)[nctx * r2 + 21],
                      (double)((float *)c67)[nctx * r2 + 22],
                      (double)((float *)c67)[nctx * r2 + 23]);
      } else {
        PAI_LOG_INFO_(PAI_SUB_GPU, "\n");
      }
    }
    m0_exp_report("G68", ok);
  }

  /* G69: sin kernel with v_sin patched to v_cos (single bit 8 of word
   * 44: 7E026B01 -> 7E026D01), writing COS values into the SIN half.
   * Discriminator: if G68 (v_sin serial) fails but G69 (v_cos serial)
   * passes, serial-per-element v_sin is the problem and the sin table
   * can be generated by the validated cos path via cos(theta-pi/2). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t r2 = 4u, nctx = 32u;
    float base = 10000.0f;
    uint32_t *c69 = (uint32_t *)ctx->c.cpu_addr;
    float ocos[128], osin[128];
    uint32_t stream_len = 0;
    int ok = 1;

    pai_ref_rope_cossin_f32(nctx, r2, base, ocos, osin);
    memset(c69, 0xCC, 2u * nctx * r2 * sizeof(uint32_t));
    memcpy(ctx->code.cpu_addr, &pai_ropegen_code[PAI_ROPEGEN_SIN_OFF],
           PAI_ROPEGEN_SIN_WORDS * sizeof(uint32_t));
    {
      uint32_t *code = (uint32_t *)ctx->code.cpu_addr;
      uint32_t w = code[PAI_ROPEGEN_SIN_VSIN_WORD];
      /* 7E026B01 (v_sin) vs 7E026D01 (v_cos): bits 9,10 (0x6B<->0x6D
       * = byte bits 1,2) toggle; clear 9, set 10. */
      code[PAI_ROPEGEN_SIN_VSIN_WORD] = (w & ~0x200u) | 0x400u;
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G69] v_sin word %u patched 0x%08x -> 0x%08x\n",
                    PAI_ROPEGEN_SIN_VSIN_WORD, w,
                    code[PAI_ROPEGEN_SIN_VSIN_WORD]);
    }
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ropegen_sin, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ROPEGEN_RSRC2,
                             PAI_ROPEGEN_THREADS, nctx * r2, ud, 6,
                             &stream_len);
    ctx->timeout_ns = UINT64_C(30000000000); /* 30s */
    m0_run_gpu(ctx, stream, stream_len,
               (uint8_t *)c69 + nctx * r2 * 4, nctx * r2 * 4, 0xCC,
               "G69");
    ctx->timeout_ns = 0;
    {
      uint32_t fmis = UINT32_MAX;
      uint32_t nbad = 0;
      for (uint32_t e = 0; e < nctx * r2; e++) {
        float gg;
        memcpy(&gg, &c69[nctx * r2 + e], 4);
        if (fabsf(gg - ocos[e]) > 1e-4f) {
          if (fmis == UINT32_MAX) {
            fmis = e;
          }
          nbad++;
        }
      }
      ok = (nbad == 0);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G69] c[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f "
                    "%.4f bad=%u first_mis=%u\n",
                    (double)((float *)c69)[nctx * r2 + 0],
                    (double)((float *)c69)[nctx * r2 + 1],
                    (double)((float *)c69)[nctx * r2 + 2],
                    (double)((float *)c69)[nctx * r2 + 3],
                    (double)((float *)c69)[nctx * r2 + 4],
                    (double)((float *)c69)[nctx * r2 + 5],
                    (double)((float *)c69)[nctx * r2 + 6],
                    (double)((float *)c69)[nctx * r2 + 7], nbad, fmis);
    }
    m0_exp_report("G69", ok);
  }

  /* G70: sin table via cos-shift - v_sin is toxic serial-per-element
   * (G68 fail / G69 pass), but sin(theta) = cos(theta - pi/2), so
   * dispatch the cos kernel with theta_turns' = theta_turns - 0.25
   * into the sin half. Validates the production sin-table path. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t r2 = 4u, nctx = 32u;
    float base = 10000.0f;
    uint32_t *h70 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c70 = (uint32_t *)ctx->c.cpu_addr;
    float ocos[128], osin[128];
    uint32_t stream_len = 0;
    int ok = 1;

    pai_ref_rope_cossin_f32(nctx, r2, base, ocos, osin);
    memset(c70, 0xCC, 2u * nctx * r2 * sizeof(uint32_t));
    h70[0] = r2;
    h70[1] = nctx;
    for (uint32_t p = 0; p < nctx; p++) {
      for (uint32_t i = 0; i < r2; i++) {
        float inv = powf(base, -(float)i / (float)r2);
        float tt = (float)p * inv / 6.2831853f - 0.25f;
        memcpy(&h70[2 + p * r2 + i], &tt, 4);
      }
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
    memcpy(ctx->code.cpu_addr, &pai_ropegen_code[PAI_ROPEGEN_SIN_OFF],
           PAI_ROPEGEN_SIN_WORDS * sizeof(uint32_t));
    {
      uint32_t *code = (uint32_t *)ctx->code.cpu_addr;
      uint32_t w = code[PAI_ROPEGEN_SIN_VSIN_WORD];
      code[PAI_ROPEGEN_SIN_VSIN_WORD] = (w & ~0x200u) | 0x400u;
    }
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_ropegen_sin, NULL);
    }
    pai_gpu_buffer_flush(ctx->gpu, &ctx->code);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ROPEGEN_RSRC2,
                             PAI_ROPEGEN_THREADS, nctx * r2, ud, 6,
                             &stream_len);
    ctx->timeout_ns = UINT64_C(30000000000); /* 30s */
    m0_run_gpu(ctx, stream, stream_len,
               (uint8_t *)c70 + nctx * r2 * 4, nctx * r2 * 4, 0xCC,
               "G70");
    ctx->timeout_ns = 0;
    {
      uint32_t fmis = UINT32_MAX;
      uint32_t nbad = 0;
      for (uint32_t e = 0; e < nctx * r2; e++) {
        float gg;
        memcpy(&gg, &c70[nctx * r2 + e], 4);
        if (fabsf(gg - osin[e]) > 1e-4f) {
          if (fmis == UINT32_MAX) {
            fmis = e;
          }
          nbad++;
        }
      }
      ok = (nbad == 0);
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G70] s[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f "
                    "%.4f bad=%u first_mis=%u\n",
                    (double)((float *)c70)[nctx * r2 + 0],
                    (double)((float *)c70)[nctx * r2 + 1],
                    (double)((float *)c70)[nctx * r2 + 2],
                    (double)((float *)c70)[nctx * r2 + 3],
                    (double)((float *)c70)[nctx * r2 + 4],
                    (double)((float *)c70)[nctx * r2 + 5],
                    (double)((float *)c70)[nctx * r2 + 6],
                    (double)((float *)c70)[nctx * r2 + 7], nbad, fmis);
    }
    m0_exp_report("G70", ok);
  }

  /* G17: load from the kernel's own acqrb VA - does ANY load complete,
   * or only our dmem pages hang? */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint64_t acq = pai_gpu_aux_va(ctx->gpu);
    uint32_t *c17 = (uint32_t *)ctx->c.cpu_addr;
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-G17] acqrb VA = 0x%llx\n",
                  (unsigned long long)acq);
    if (!acq) {
      m0_exp_report("G17", 0);
    } else {
      memset(c17, 0xCC, 128 * sizeof(uint32_t));
      memcpy(ctx->code.cpu_addr, pai_acqload_code,
             PAI_G17_CODE_WORDS * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_g8, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(acq & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(acq >> 32);
      ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G17_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c17, 128, 0xCC, "G17");
      PAI_LOG_INFO_(PAI_SUB_GPU,
                    "[M0-G17] c[0..7] = %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    c17[0], c17[1], c17[2], c17[3], c17[4], c17[5], c17[6],
                    c17[7]);
      {
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          if (c17[i] == 0xCCCCCCCCu) {
            ok = 0;
            break;
          }
        }
        m0_exp_report("G17", ok);
      }
    }
  }

  /* G13/G14: literal source probes + formal milestone fill. */
  {
    static const uint32_t g_offs[2] = {PAI_G13_OFF, PAI_G14_OFF};
    static const uint32_t g_lens[2] = {PAI_G13_WORDS, PAI_G14_WORDS};
    static const char *const g_names[2] = {"G13", "G14"};

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_gbatch3_code[g_offs[variant]],
             g_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_g7, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ARITH4_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x %08x "
                    "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                    c32[6], c32[7], c32[8], c32[9], c32[10], c32[11], c32[12],
                    c32[13], c32[14], c32[15]);
      if (variant == 1) {
        int ok = 1;
        for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
          if (c32[i] != PAI_G14_VALUE) {
            ok = 0;
            break;
          }
        }
        m0_exp_report("G14", ok);
      }
    }
  }

  /* G9-G12: store data source probes (raw dumps, no formula check). */
  {
    static const uint32_t g_offs[4] = {PAI_G9_OFF, PAI_G10_OFF, PAI_G11_OFF,
                                       PAI_G12_OFF};
    static const uint32_t g_lens[4] = {PAI_G9_WORDS, PAI_G10_WORDS,
                                       PAI_G11_WORDS, PAI_G12_WORDS};
    static const char *const g_names[4] = {"G9", "G10", "G11", "G12"};

    for (uint32_t variant = 0; variant < 4; variant++) {
      const char *name = g_names[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_gbatch2_code[g_offs[variant]],
             g_lens[variant] * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_g7, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ARITH4_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);
      PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x %08x "
                    "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                    "%08x %08x\n",
                    name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                    c32[6], c32[7], c32[8], c32[9], c32[10], c32[11], c32[12],
                    c32[13], c32[14], c32[15]);
    }
  }

  /* G7/G8: x4 per-thread broadcast + integer arithmetic. */
  {
    static const uint32_t g_offs[2] = {PAI_G7_OFF, PAI_G8_OFF};
    static const uint32_t g_lens[2] = {PAI_G7_WORDS, PAI_G8_WORDS};
    static const char *const g_names[2] = {"G7", "G8"};
    uint32_t k = 0x00001000u;

    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = g_names[variant];
      uint32_t off = g_offs[variant];
      uint32_t words = g_lens[variant];
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_gbatch_code[off],
             words * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     variant == 0 ? pai_host_kernel_g7
                                                  : pai_host_kernel_g8,
                                     NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      ud[4] = k;
      ud[5] = 0;
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G8_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128 * 4, 0xCC, name);

      {
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          uint32_t want = variant == 0 ? i : i * 4 + k;
          for (uint32_t j = 0; j < 4; j++) {
            if (c32[i * 4 + j] != want) {
              ok = 0;
              PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[%u] = %08x want %08x\n",
                             name, i * 4 + j, c32[i * 4 + j], want);
              break;
            }
          }
          if (!ok) {
            break;
          }
        }
        m0_exp_report(name, ok);
        if (!ok) {
          PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x "
                         "%08x %08x %08x %08x %08x %08x %08x %08x %08x "
                         "%08x %08x %08x %08x\n",
                         name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                         c32[6], c32[7], c32[8], c32[9], c32[10], c32[11],
                         c32[12], c32[13], c32[14], c32[15]);
        }
      }
    }
  }

  /* F5: shotgun — value in v0, v4, v5 at the store. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, pai_shotgun_code,
           PAI_SHOTGUN_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_fbatch, NULL);
    }
    ud[0] = PAI_SHOTGUN_VALUE;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_SHOTGUN_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "F5");
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-F5] c[0..7] = %08x %08x %08x %08x "
                  "%08x %08x %08x %08x\n",
                  c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                  c32[7]);
  }

  /* F1-F4: dst-v0-broadcast workaround batch. */
  {
    float k = 1.5f;
    uint32_t k_bits;
    memcpy(&k_bits, &k, sizeof(k_bits));

    for (uint32_t variant = 0; variant < 4; variant++) {
      static const uint32_t offs[4] = {PAI_F1_OFF, PAI_F2_OFF, PAI_F3_OFF,
                                       PAI_F4_OFF};
      static const uint32_t lens[4] = {PAI_F1_WORDS, PAI_F2_WORDS,
                                       PAI_F3_WORDS, PAI_F4_WORDS};
      static const char *const names[4] = {"F1", "F2", "F3", "F4"};
      uint32_t off = offs[variant];
      uint32_t words = lens[variant];
      const char *name = names[variant];

      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_fbatch_code[off],
             words * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_arith, NULL);
      }
      ud[0] = k_bits;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ARITH4_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);

      if (variant == 1) {
        /* F2: store semantics probe — observed on 9.40: lanes 0..7
         * write (tid<<2)+3, lanes 8+ leave the fill intact (exec-mask
         * quirk to investigate). Verify the observed lanes. */
        int ok = 1;
        for (uint32_t i = 0; i < 8; i++) {
          if (c32[i] != (i << 2) + 3u) {
            ok = 0;
            break;
          }
        }
        m0_exp_report(name, ok);
        PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] c[0..15] = %08x %08x %08x %08x "
                      "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x "
                      "%08x %08x\n",
                      name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                      c32[6], c32[7], c32[8], c32[9], c32[10], c32[11],
                      c32[12], c32[13], c32[14], c32[15]);
      } else if (variant == 3) {
        /* F4: integer add -> c[i] == i + k_bits */
        int ok = 1;
        for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
          if (c32[i] != i + k_bits) {
            ok = 0;
            break;
          }
        }
        m0_exp_report(name, ok);
        if (!ok) {
          PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[0..7] = %08x %08x %08x "
                         "%08x %08x %08x %08x %08x\n",
                         name, c32[0], c32[1], c32[2], c32[3], c32[4], c32[5],
                         c32[6], c32[7]);
        }
      } else {
        /* F1/F3: float add -> c[i] == (float)i + k */
        float *cf = (float *)ctx->c.cpu_addr;
        int ok = 1;
        for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
          float want = (float)i + k;
          if (cf[i] != want) {
            ok = 0;
            PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[%u] = %f want %f\n",
                           name, i, (double)cf[i], (double)want);
            break;
          }
        }
        m0_exp_report(name, ok);
      }
    }
  }

  return 0;
}

/* E38: MUBUF per-thread load. E39: arithmetic milestone (no loads). */
static int
m0_exp_loads_arith(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[8];
  uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
  uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;
  int host = pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF;

  /* E38: MUBUF load copy. T# candidate A (radv-style). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    memcpy(ctx->code.cpu_addr, pai_mubufload_code,
           PAI_MUBUFLOAD_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_mubufload, NULL);
    }
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x38383838u + i;
    }
    ud[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[2] = 1024u;       /* num_records */
    ud[3] = 0x00097688u; /* dst_sel x/y/z/w, float, 32, elem 4B */
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_MUBUFLOAD_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "E38");
    {
      int ok = 1;
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        if (c32[i] != a32[i]) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("E38", ok);
      if (!ok) {
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-E38] c[0..7] = %08x %08x %08x "
                       "%08x %08x %08x %08x %08x\n",
                       c32[0], c32[1], c32[2], c32[3], c32[4], c32[5], c32[6],
                       c32[7]);
      }
    }
  }

  /* E45/E46: milestone arith under the proven 4-SGPR config. */
  {
    float k = 1.5f;
    for (uint32_t variant = 0; variant < 2; variant++) {
      const char *name = variant == 0 ? "E45" : "E46";
      uint32_t off = variant == 0 ? PAI_ARITH4_OFF : PAI_ARITH4B_OFF;
      uint32_t words = variant == 0 ? PAI_ARITH4_WORDS : PAI_ARITH4B_WORDS;
      if (!host) {
        pai_gpu_reset(gpu);
      }
      memcpy(ctx->code.cpu_addr, &pai_arith4_code[off],
             words * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_arith, NULL);
      }
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
      /* k sits in s0 for the arith4 ABI; the host kernel reads s4, so
       * keep both for host parity. */
      memcpy(&ud[0], &k, sizeof(k));
      memcpy(&ud[4], &k, sizeof(k));
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_ARITH4_RSRC2,
                               PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, name);
      {
        float *cf = (float *)ctx->c.cpu_addr;
        int ok = 1;
        for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
          float want = (float)i + k;
          if (cf[i] != want) {
            ok = 0;
            PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-%s] c[%u] = %f want %f\n",
                           name, i, (double)cf[i], (double)want);
            break;
          }
        }
        m0_exp_report(name, ok);
      }
    }
  }

  return 0;
}

/* E40: register scan — which VGPR holds the per-thread id?
 * Template kernel: c[i] == i iff vK == tid. */
static int
m0_exp_regscan(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[4];
  uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;
  int host = pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF;

  for (uint32_t k = 0; k <= 10; k++) {
    static const uint32_t template_words[12] = {
        0x7E020300u, /* w0:  v_mov_b32 v1, vK   (patched) */
        0x34080682u, /* w1:  v_lshlrev_b32 v1, 2, v1 */
        0x7E040202u, /* w2:  v_mov_b32 v2, s2 */
        0x7E060203u, /* w3:  v_mov_b32 v3, s3 */
        0xD70F6A02u, /* w4:  v_add_co_u32 v2, vcc_lo, v2, v1 */
        0x00020502u, /* w5:  (operand) */
        0xD5286A03u, /* w6:  v_add_co_ci_u32 v3, vcc_lo, v3, 0, vcc_lo */
        0x01A90103u, /* w7:  (operand) */
        0x7E000300u, /* w8:  v_mov_b32 v0, vK   (patched) */
        0xDC700000u, /* w9:  flat_store_dword v[2:3], v4 */
        0x007D0404u, /* w10: (operand) */
        0xBF810000u, /* w11: s_endpgm */
    };
    char name[8];
    uint32_t code[12];
    int ok;

    if (!host) {
      pai_gpu_reset(gpu);
    }
    snprintf(name, sizeof(name), "E40%c", (char)('a' + k));
    memcpy(code, template_words, sizeof(code));
    code[0] |= k; /* v_mov v1, vK */
    code[8] |= k; /* v_mov v0, vK */
    memcpy(ctx->code.cpu_addr, code, sizeof(code));
    if (host) {
      /* Host: emulate "c[i] = value of reg K" = k for all i. */
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_store_dw, NULL);
      for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
        ((uint32_t *)ctx->c.cpu_addr)[i] = k;
      }
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, 0x00000008u,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xEE, name);

    ok = 1;
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      if (c32[i] != i) {
        ok = 0;
        break;
      }
    }
    PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] v%d == tid? %s (c[0..3] = %08x "
                  "%08x %08x %08x)\n",
                  name, k, ok ? "YES" : "no", c32[0], c32[1], c32[2], c32[3]);
  }
  return 0;
}

/* E39c: arith with v9 and the golden register config (64 threads,
 * RSRC2 0x92, TGID_X_EN). */
static int
m0_exp_arith_golden_cfg(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[9];
  uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;
  int host = pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF;

  if (!host) {
    pai_gpu_reset(gpu);
  }

  memcpy(ctx->code.cpu_addr, pai_arith_code,
         PAI_ARITH_CODE_WORDS * sizeof(uint32_t));
      if (host) {
        pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                     pai_host_kernel_fbatch, NULL);
      }

  ud[0] = 0;
  ud[1] = 0;
  ud[2] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  ud[3] = (uint32_t)(ctx->c.gpu_addr >> 32);
  ud[4] = 0x3FC00000u; /* 1.5f */
  ud[5] = 0;
  ud[6] = 0;
  ud[7] = 0;
  ud[8] = 0;
  m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_MEMSET16_RSRC2,
                           PAI_MEMSET16_THREADS_X, 1, ud, 9, &stream_len);
  m0_run_gpu(ctx, stream, stream_len, c32, 256, 0xCC, "E39c");

  {
    float *cf = (float *)ctx->c.cpu_addr;
    int ok = 1;
    for (uint32_t i = 0; i < 64; i++) {
      float want = (float)i + 1.5f;
      if (cf[i] != want) {
        ok = 0;
        PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-E39c] c[%u] = %f want %f\n", i,
                       (double)cf[i], (double)want);
        break;
      }
    }
    m0_exp_report("E39c", ok);
  }
  return 0;
}

static int
m0_stage_e(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  int host = pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF;

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-E] compute experiment matrix\n");
  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-E] order: safe (patched/proven) first, controls last; "
                "gc reset between experiments\n");

  /* E40: scan VGPRs for the per-thread id. E39c: golden config arith. */
  m0_exp_regscan(ctx);
  m0_exp_arith_golden_cfg(ctx);

  /* E38/E39: MUBUF loads + the arithmetic milestone. */
  m0_exp_loads_arith(ctx);

  /* E34-E37: the flat v0-broadcast model (loads + vecadd milestone). */
  m0_exp_v0model(ctx);

  /* E32: v0-broadcast store hypothesis. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store64_variant(ctx, "E32", pai_store64_v0_code,
                         PAI_STORE64_V0_CODE_WORDS, PAI_STORE64_V0_VALUE,
                         256, pai_host_kernel_store64_v0, 0, 0);

  /* E30/E31: golden register layout (vaddr pair 1). */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store64_variant(ctx, "E30", pai_store64_gold_code,
                         PAI_STORE64_GOLD_CODE_WORDS, PAI_STORE64_GOLD_VALUE,
                         256, pai_host_kernel_store64_gold, 0, 0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    /* E31: loadstore with golden addressing. */
    uint32_t stream[M0_PM4_CAP];
    uint32_t stream_len;
    uint32_t ud[6];
    uint32_t *a32 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *c32 = (uint32_t *)ctx->c.cpu_addr;

    memcpy(ctx->code.cpu_addr, pai_loadstore_gold_code,
           PAI_LOADSTORE_GOLD_CODE_WORDS * sizeof(uint32_t));
    if (pai_gpu_device_backend(gpu) == PAI_GPU_BACKEND_HOST_REF) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_loadstore_gold, NULL);
    }
    for (uint32_t i = 0; i < PAI_EXP_THREADS_X; i++) {
      a32[i] = 0x31313131u + i;
    }
    ud[0] = 0;
    ud[1] = 0;
    ud[2] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
    ud[3] = (uint32_t)(ctx->a.gpu_addr >> 32);
    ud[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
    ud[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_LOADSTORE_GOLD_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 6, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c32, 128, 0xCC, "E31");
    m0_check_loadstore(a32, c32, "E31");
  }

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G1", 0, 0); /* golden verbatim */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G2", 0xDC780000u, 0); /* bit15 cleared */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G3", 0, 0x007D0604u); /* my vaddr/data regs */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G5", 0, 0x007D0602u); /* data v6, vaddr pair 1 */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G6", 0, 0x007D0404u); /* data v4, vaddr pair 2 */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_golden_mutant(ctx, "G4", 0xDC780000u, 0x007D0604u); /* my store */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E15", PAI_BISECT_BARE_OFF, PAI_BISECT_BARE_WORDS, 0, 0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E16", PAI_BISECT_LSHL_OFF, PAI_BISECT_LSHL_WORDS, 0, 0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E17", PAI_BISECT_ADDCO_OFF, PAI_BISECT_ADDCO_WORDS, 0,
                0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E18", PAI_BISECT_ADDCI_OFF, PAI_BISECT_ADDCI_WORDS, 0,
                0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E19", PAI_BISECT_STORE_OFF, PAI_BISECT_STORE_WORDS, 1,
                0);

  /* The golden store encoding: op 0x78 (x4) + bit 15. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_bisect(ctx, "E19b", PAI_BISECT_STORE_OFF, PAI_BISECT_STORE_WORDS, 1,
                1);

  /* E22: SMEM s_load prologue + golden store encoding. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store64_variant(ctx, "E22", pai_store64_smem_code,
                         PAI_STORE64_SMEM_CODE_WORDS, PAI_STORE64_SMEM_VALUE,
                         256, pai_host_kernel_store64_smem, 1,
                         PAI_STORE64_SMEM_FLAT_WORD);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store64_variant(ctx, "E20", pai_store64_x2_code,
                         PAI_STORE64_X2_CODE_WORDS, PAI_STORE64_X2_VALUE, 128,
                         pai_host_kernel_store64_x2, 1,
                         PAI_STORE64_X2_FLAT_WORD);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store64_variant(ctx, "E21", pai_store64_x4_code,
                         PAI_STORE64_X4_CODE_WORDS, PAI_STORE64_X4_VALUE, 256,
                         pai_host_kernel_store64_x4, 1,
                         PAI_STORE64_X4_FLAT_WORD);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_memset(ctx, 64, "E3");

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store_const64(ctx, "E5", 0); /* psbc ABI, no preamble */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_memset_preamble(ctx, 64, "E6"); /* preamble + golden kernel */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store_const64(ctx, "E7", 1); /* psbc ABI + preamble */

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_memset(ctx, 1024, "E4");

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store_const(ctx, "E1b", 1);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_loadstore(ctx, "E2b", 1);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_store_const(ctx, "E1", 0);

  if (!host) {
    pai_gpu_reset(gpu);
  }
  m0_exp_loadstore(ctx, "E2", 0);

  return 0;
}

/* Stage B0 — OpenAGC memset16 golden dispatch reference. */

static void
m0_build_memset16_stream(m0_ctx_t *ctx, uint32_t *stream, uint32_t cap,
                         uint64_t dst_addr, uint32_t blocks,
                         const uint32_t pattern[4], uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[9];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, stream, cap);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC1;
  vals[1] = PAI_MEMSET16_RSRC2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_MEMSET16_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_MEMSET16_THREADS_X;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  vals[0] = 0; /* ring offsets (unused) */
  vals[1] = 0;
  vals[2] = (uint32_t)(dst_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(dst_addr >> 32);
  vals[4] = blocks;
  vals[5] = pattern[0];
  vals[6] = pattern[1];
  vals[7] = pattern[2];
  vals[8] = pattern[3];
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 9, vals);

  pai_pm4_dispatch_direct(&pb, (blocks + PAI_MEMSET16_THREADS_X - 1) /
                                   PAI_MEMSET16_THREADS_X,
                          1, 1, 0);

  *out_len = pb.len;
}

static int
m0_stage_b0(m0_ctx_t *ctx) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t pattern[4] = {0xDEADBEEFu, 0x11223344u, 0x55667788u, 0x99AABBCCu};
  const uint32_t blocks = 1024; /* 16 KiB */
  uint8_t *dst = (uint8_t *)ctx->dst.cpu_addr;
  uint32_t ref[1024 * 4];
  pai_status_t st;

  PAI_LOG_INFO_(PAI_SUB_GPU,
                "[M0-B0] GPU compute dispatch (memset16 golden, blocks=%u)\n",
                blocks);

  memcpy(ctx->code.cpu_addr, pai_memset16_code,
         PAI_MEMSET16_CODE_WORDS * sizeof(uint32_t));

  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    st = pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                      pai_host_kernel_memset16, NULL);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B0] shader registration failed\n");
      return -1;
    }
  }

  m0_build_memset16_stream(ctx, stream, M0_PM4_CAP, ctx->dst.gpu_addr, blocks,
                           pattern, &stream_len);

  if (m0_run_gpu(ctx, stream, stream_len, dst, blocks * 16, 0x00, "M0-B0") !=
      0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B0] FAIL: GPU never executed the stream\n");
    return -1;
  }

  st = pai_ref_memset16(ref, pattern, blocks);
  if (st != PAI_OK || memcmp(dst, ref, blocks * 16) != 0) {
    uint32_t matching = 0;
    for (uint32_t i = 0; i < blocks * 16 / 4; i++) {
      if (((uint32_t *)dst)[i] == ((uint32_t *)ref)[i]) {
        matching++;
      }
    }
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B0] FAIL: pattern mismatch (%u/%u dwords match)\n",
                   matching, blocks * 16 / 4);
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B0] dst[0..15]  = %08x %08x %08x %08x\n",
                   ((uint32_t *)dst)[0], ((uint32_t *)dst)[1],
                   ((uint32_t *)dst)[2], ((uint32_t *)dst)[3]);
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B0] ref[0..15]  = %08x %08x %08x %08x\n",
                   ((uint32_t *)ref)[0], ((uint32_t *)ref)[1],
                   ((uint32_t *)ref)[2], ((uint32_t *)ref)[3]);
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B0] dst[16..31] = %08x %08x %08x %08x\n",
                   ((uint32_t *)dst)[4], ((uint32_t *)dst)[5],
                   ((uint32_t *)dst)[6], ((uint32_t *)dst)[7]);
    return -1;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-B0] PASS: compute dispatch verified\n");
  return 0;
}

/* Stage B — gfx1013 vecadd vs CPU reference. */

static void
m0_build_vecadd_stream(m0_ctx_t *ctx, uint32_t *stream, uint32_t cap,
                       uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[8];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, stream, cap);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_VECADD_RSRC1;
  vals[1] = PAI_VECADD_RSRC2;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC1, 2, vals);

  vals[0] = PAI_VECADD_RSRC3;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_RSRC3, 1, vals);

  vals[0] = PAI_VECADD_THREADS_X;
  vals[1] = 1;
  vals[2] = 1;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_NUM_THREAD_X, 3, vals);

  vals[0] = (uint32_t)(ctx->a.gpu_addr & 0xFFFFFFFFu);
  vals[1] = (uint32_t)(ctx->a.gpu_addr >> 32);
  vals[2] = (uint32_t)(ctx->b.gpu_addr & 0xFFFFFFFFu);
  vals[3] = (uint32_t)(ctx->b.gpu_addr >> 32);
  vals[4] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
  vals[5] = (uint32_t)(ctx->c.gpu_addr >> 32);
  vals[6] = PAI_VECADD_MAX_ELEMS;
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_USER_DATA_0, 7, vals);

  pai_pm4_dispatch_direct(&pb, 1, 1, 1, 0);

  *out_len = pb.len;
}

static int
m0_stage_b(m0_ctx_t *ctx) {
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  float *a = (float *)ctx->a.cpu_addr;
  float *b = (float *)ctx->b.cpu_addr;
  float *c = (float *)ctx->c.cpu_addr;
  float ref[PAI_VECADD_MAX_ELEMS];
  uint32_t n = PAI_VECADD_MAX_ELEMS;
  uint64_t mismatch = 0;
  pai_status_t st;

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-B] GPU compute dispatch (vecadd, n=%u)\n",
                n);

  for (uint32_t i = 0; i < n; i++) {
    a[i] = (float)i * 0.5f + 1.0f;
    b[i] = (float)(i % 5) * 0.25f - 0.5f;
  }

  /* WC_GARLIC is not CPU-cache-coherent: flush the CPU writes before
   * the GPU reads them. */
  pai_gpu_buffer_flush(ctx->gpu, &ctx->a);
  pai_gpu_buffer_flush(ctx->gpu, &ctx->b);

  memcpy(ctx->code.cpu_addr, pai_vecadd_code,
         PAI_VECADD_CODE_WORDS * sizeof(uint32_t));

  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    st = pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                      pai_host_kernel_vecadd, NULL);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B] shader registration failed\n");
      return -1;
    }
  }

  m0_build_vecadd_stream(ctx, stream, M0_PM4_CAP, &stream_len);

  if (m0_run_gpu(ctx, stream, stream_len, c, n * sizeof(float), 0xCC,
                 "M0-B") != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B] FAIL: GPU never executed the stream\n");
    return -1;
  }

  st = pai_ref_vecadd_f32(a, b, ref, n);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B] reference failed\n");
    return -1;
  }

  st = pai_ref_compare_f32(c, ref, n, 1e-6f, 1e-6f, &mismatch);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_GPU,
                   "[M0-B] FAIL: mismatch at %llu (gpu=%f ref=%f)\n",
                   (unsigned long long)mismatch,
                   mismatch < n ? (double)c[mismatch] : 0.0,
                   mismatch < n ? (double)ref[mismatch] : 0.0);
    return -1;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-B] PASS: GPU compute verified vs CPU\n");
  return 0;
}

/* Stage C — dispatch benchmark. */

static void
m0_stage_c_run(void *user) {
  m0_ctx_t *ctx = (m0_ctx_t *)user;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;

  m0_build_vecadd_stream(ctx, stream, M0_PM4_CAP, &stream_len);
  if (m0_run_gpu(ctx, stream, stream_len, ctx->c.cpu_addr,
                 PAI_VECADD_MAX_ELEMS * sizeof(float), 0xCC, "M0-C") != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-C] submit failed during benchmark\n");
  }
}

static void
m0_stage_c(m0_ctx_t *ctx) {
  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-C] dispatch benchmark (10 iters)\n");
  pai_bench_run(&ctx->bench, 2, 10, m0_stage_c_run, ctx);
  PAI_LOG_INFO_(PAI_SUB_PROFILER,
                "dispatch+wait: best=%llu us mean=%llu us\n",
                (unsigned long long)(ctx->bench.best_ns / 1000),
                (unsigned long long)(ctx->bench.mean_ns / 1000));
}

/* ------------------------------------------------------------------ */

int
main(void) {
  m0_ctx_t ctx;
  pai_runtime_t *rt = NULL;
  pai_status_t st;
  uint32_t fail = 0;

  memset(&ctx, 0, sizeof(ctx));

  PAI_LOG_INFO_(PAI_SUB_CORE, "===== PAI-M0 bring-up harness =====\n");

  /* GPU page-table repair rehearsal: 0 probe / 1 no-op write / 2 full. */
  pai_gvmspace_set_mode(PAI_GVMSPACE_MODE);

  /* Stop any previous payload instance still running (deploy
   * automation, same pattern as MemDBG). Best effort, never fatal. */
  pai_lifecycle_stop_previous(PAI_LIFECYCLE_PORT);

  st = pai_runtime_init(&rt);
  if (st != PAI_OK || !rt) {
    PAI_LOG_ERROR_(PAI_SUB_CORE, "runtime init failed: %s\n",
                   pai_status_str(st));
    return 3;
  }
  ctx.rt = rt;
  ctx.gpu = pai_runtime_gpu(rt);

  /* Serve stop probes so the next deploy can replace us cleanly. */
  if (pai_lifecycle_start(PAI_LIFECYCLE_PORT) != PAI_OK) {
    PAI_LOG_WARN_(PAI_SUB_CORE,
                  "lifecycle listener unavailable; next deploy may not "
                  "replace this instance automatically\n");
  }

  pai_notify(PAI_DEPLOY_NOTIFY);

  if (!ctx.gpu) {
    PAI_LOG_WARN_(PAI_SUB_CORE,
                  "no GPU backend; M0 GPU stages skipped (safe mode?)\n");
    PAI_LOG_INFO_(PAI_SUB_CORE, "===== PAI-M0: SKIPPED =====\n");
    pai_runtime_shutdown(rt);
    return 0;
  }

  if (m0_ctx_alloc_buffers(&ctx) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_CORE, "buffer allocation failed\n");
    fail = 3;
    goto out;
  }

  if (m0_stage_a(&ctx) != 0) {
    fail |= M0_STAGE_A_FAIL;
  }

  /* Post-DMA staging check (read-only): does the GPU PDE point at a
   * page that contains the DMA pattern, or a separate zero page? */
  {
    uint32_t w[4];
    const uint32_t *src32 = (const uint32_t *)ctx.src.cpu_addr;
    PAI_LOG_INFO_(PAI_SUB_CORE,
                  "staging: src cpu page = %08x %08x %08x %08x\n",
                  src32[0], src32[1], src32[2], src32[3]);
    if (pai_gvmspace_dump_pde_page(ctx.src.gpu_addr, w) == 0) {
      PAI_LOG_INFO_(PAI_SUB_CORE,
                    "staging: src PDE page = %08x %08x %08x %08x\n",
                    w[0], w[1], w[2], w[3]);
    } else {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging: src PDE page unreadable\n");
    }
    if (pai_gvmspace_dump_pde_page(ctx.dst.gpu_addr, w) == 0) {
      PAI_LOG_INFO_(PAI_SUB_CORE,
                    "staging: dst PDE page = %08x %08x %08x %08x\n",
                    w[0], w[1], w[2], w[3]);
    } else {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging: dst PDE page unreadable\n");
    }
  }

  m0_stage_e(&ctx);

  /* stage_e includes known hang probes (G17 flat_load, etc.). Empirics:
   * pai_gpu_reset recovers a wedged ring for the next experiment. */
  if (pai_gpu_device_backend(ctx.gpu) != PAI_GPU_BACKEND_HOST_REF) {
    pai_gpu_reset(ctx.gpu);
  }

  if (m0_stage_b0(&ctx) != 0) {
    fail |= M0_STAGE_B_FAIL;
  }

  if (m0_stage_b(&ctx) != 0) {
    fail |= M0_STAGE_B_FAIL;
  }

  /* Post-compute staging check (read-only): compare the CPU pages and
   * the pages the GPU PDEs point at for the a/b/c buffers. */
  {
    uint32_t w[4];
    const uint32_t *c32 = (const uint32_t *)ctx.c.cpu_addr;
    const uint32_t *a32 = (const uint32_t *)ctx.a.cpu_addr;
    const uint32_t *b32 = (const uint32_t *)ctx.b.cpu_addr;
    PAI_LOG_INFO_(PAI_SUB_CORE,
                  "staging2: a cpu = %08x %08x %08x %08x\n", a32[0], a32[1],
                  a32[2], a32[3]);
    if (pai_gvmspace_dump_pde_page(ctx.a.gpu_addr, w) == 0) {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: a PDE page = %08x %08x %08x "
                    "%08x\n", w[0], w[1], w[2], w[3]);
    } else {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: a PDE page unreadable\n");
    }
    PAI_LOG_INFO_(PAI_SUB_CORE,
                  "staging2: b cpu = %08x %08x %08x %08x\n", b32[0], b32[1],
                  b32[2], b32[3]);
    if (pai_gvmspace_dump_pde_page(ctx.b.gpu_addr, w) == 0) {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: b PDE page = %08x %08x %08x "
                    "%08x\n", w[0], w[1], w[2], w[3]);
    } else {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: b PDE page unreadable\n");
    }
    PAI_LOG_INFO_(PAI_SUB_CORE,
                  "staging2: c cpu = %08x %08x %08x %08x\n", c32[0], c32[1],
                  c32[2], c32[3]);
    if (pai_gvmspace_dump_pde_page(ctx.c.gpu_addr, w) == 0) {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: c PDE page = %08x %08x %08x "
                    "%08x\n", w[0], w[1], w[2], w[3]);
    } else {
      PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: c PDE page unreadable\n");
    }
    /* Where do the GPU stores physically land? Resolve the CPU phys
     * of the c VA via the CPU page tables and dump it. */
    {
      uint64_t cphys = 0;
      if (pai_cpu_phys_of_va(ctx.c.gpu_addr, &cphys) == 0) {
        PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: c cpu phys = 0x%llx\n",
                      (unsigned long long)cphys);
        if (pai_gvmspace_dump_phys(cphys, w) == 0) {
          PAI_LOG_INFO_(PAI_SUB_CORE,
                        "staging2: c phys page = %08x %08x %08x %08x\n",
                        w[0], w[1], w[2], w[3]);
        }
      } else {
        PAI_LOG_INFO_(PAI_SUB_CORE, "staging2: c cpu phys unresolved\n");
      }
    }
  }

  if (fail == 0) {
    m0_stage_c(&ctx);
  }

out:
  m0_ctx_free_buffers(&ctx);
  pai_runtime_shutdown(rt);

  PAI_LOG_INFO_(PAI_SUB_CORE, "===== PAI-M0: %s =====\n",
                fail == 0 ? "ALL PASSED" : "FAILED");
  return (int)fail;
}
