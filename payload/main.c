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
 * `watch` is pre-filled with `watch_fill`; if the label times out but
 * `watch` changed, the stream ran and only the fence is broken.
 * Returns 0 on success, -1 when the stream never executed. */
static int
m0_run_gpu(m0_ctx_t *ctx, uint32_t *stream, uint32_t stream_len,
           void *watch, uint32_t watch_bytes, uint32_t watch_fill,
           const char *stage) {
  static const char *const fence_names[] = {
      "eop-action", "write-data", "eop-legacy"};

  for (uint32_t fence = 0; fence < 3; fence++) {
    pai_pm4_builder_t pb;
    pai_status_t st;

    memset(watch, (int)watch_fill, watch_bytes);
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
                   "[%s] attempt: fence=%s label=0x%llx value=0x%x\n", stage,
                   fence_names[fence], (unsigned long long)ctx->label.gpu_addr,
                   ctx->label_value);

    st = ctx->acb_mode
             ? pai_gpu_submit_acb(ctx->gpu, stream, pb.len)
             : pai_gpu_submit_q(ctx->gpu, stream, pb.len, 3);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "[%s] submit failed (fence=%s): %s\n",
                     stage, fence_names[fence], pai_status_str(st));
      continue;
    }

    st = pai_gpu_wait_label(ctx->gpu, ctx->label.gpu_addr, ctx->label_value,
                            M0_TIMEOUT_NS);
    if (st == PAI_OK) {
      PAI_LOG_INFO_(PAI_SUB_GPU, "[%s] label fired (fence=%s)\n", stage,
                    fence_names[fence]);

  return 0;
}

    /* Label timed out: did the GPU run the stream anyway? */
    {
      const uint8_t *w = (const uint8_t *)watch;
      uint32_t i;
      int changed = 0;
      for (i = 0; i < watch_bytes; i++) {
        if (w[i] != (uint8_t)watch_fill) {
          changed = 1;
          break;
        }
      }
      if (changed) {
        PAI_LOG_WARN_(PAI_SUB_GPU,
                      "[%s] data changed but label did not fire (fence=%s): "
                      "stream executed\n",
                      stage, fence_names[fence]);

  return 0;
}
    }

    PAI_LOG_WARN_(PAI_SUB_GPU,
                  "[%s] no execution observed (fence=%s, label timeout)\n",
                  stage, fence_names[fence]);
  }

  return -1;
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
m0_build_dispatch_stream(m0_ctx_t *ctx, uint32_t *stream, uint32_t cap,
                         uint32_t rsrc2, uint32_t threads_x, uint32_t groups_x,
                         const uint32_t *user_data, uint32_t ud_count,
                         uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[9];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, stream, cap);

  vals[0] = (uint32_t)(code >> 8);
  vals[1] = (uint32_t)(code >> 40);
  pai_pm4_set_sh_reg_compute(&pb, PAI_REG_COMPUTE_PGM_LO, 2, vals);

  vals[0] = PAI_EXP_RSRC1;
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
  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-%s] %s\n", name, ok ? "PASS" : "FAIL");
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

/* E34-E37: the flat v0-broadcast model — loads and the real vecadd. */
static int
m0_exp_v0model(m0_ctx_t *ctx) {
  pai_gpu_device_t *gpu = ctx->gpu;
  uint32_t stream[M0_PM4_CAP];
  uint32_t stream_len;
  uint32_t ud[8];
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

  /* G23: SMEM->LDS staged vecadd - 128 elements in 4 dispatches. */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *a23 = (uint32_t *)ctx->a.cpu_addr;
    uint32_t *b23 = (uint32_t *)ctx->b.cpu_addr;
    uint32_t *c23 = (uint32_t *)ctx->c.cpu_addr;
    memset(c23, 0xCC, 128 * sizeof(uint32_t));
    for (uint32_t i = 0; i < PAI_VECADD_MAX_ELEMS; i++) {
      a23[i] = 0x11110000u + i;
      b23[i] = 0x00001111u + i;
    }
    memcpy(ctx->code.cpu_addr, pai_smemvecadd_code,
           PAI_G23_CODE_WORDS * sizeof(uint32_t));
    if (host) {
      pai_gpu_host_register_shader(gpu, ctx->code.gpu_addr,
                                   pai_host_kernel_g8, NULL);
    }
    for (uint32_t chunk = 0; chunk < 4; chunk++) {
      uint64_t a_base = ctx->a.gpu_addr + chunk * 128u;
      uint64_t b_base = ctx->b.gpu_addr + chunk * 128u;
      ud[0] = 0;
      ud[1] = 0;
      ud[2] = (uint32_t)(a_base & 0xFFFFFFFFu);
      ud[3] = (uint32_t)(a_base >> 32);
      ud[4] = (uint32_t)(b_base & 0xFFFFFFFFu);
      ud[5] = (uint32_t)(b_base >> 32);
      ud[6] = (uint32_t)(ctx->c.gpu_addr & 0xFFFFFFFFu);
      ud[7] = (uint32_t)(ctx->c.gpu_addr >> 32);
      m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, 0x00000048u,
                               PAI_EXP_THREADS_X, 1, ud, 8, &stream_len);
      m0_run_gpu(ctx, stream, stream_len, c23 + chunk * 32, 32 * 4, 0xCC,
                 "G23");
    }
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G23] c[0..15] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
                  c23[0], c23[1], c23[2], c23[3], c23[4], c23[5], c23[6],
                  c23[7], c23[8], c23[9], c23[10], c23[11], c23[12], c23[13],
                  c23[14], c23[15]);
    {
      int ok = 1;
      for (uint32_t i = 0; i < 16; i++) {
        if (c23[i] != (a23[i] + b23[i])) {
          ok = 0;
          break;
        }
      }
      m0_exp_report("G23", ok);
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

  /* G25: minimal LDS roundtrip - are the ds ops themselves broken? */
  if (!host) {
    pai_gpu_reset(gpu);
  }
  {
    uint32_t *c25 = (uint32_t *)ctx->c.cpu_addr;
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
    m0_build_dispatch_stream(ctx, stream, M0_PM4_CAP, PAI_G25_RSRC2,
                             PAI_EXP_THREADS_X, 1, ud, 4, &stream_len);
    m0_run_gpu(ctx, stream, stream_len, c25, 128, 0xCC, "G25");
    PAI_LOG_INFO_(PAI_SUB_GPU,
                  "[M0-G25] c[0..7] = %08x %08x %08x %08x %08x %08x "
                  "%08x %08x\n",
                  c25[0], c25[1], c25[2], c25[3], c25[4], c25[5], c25[6],
                  c25[7]);
    {
      int ok = (c25[0] == PAI_G25_VALUE) && (c25[1] == PAI_G25_VALUE);
      m0_exp_report("G25", ok);
    }
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
