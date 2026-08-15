/*
 * prosperoai.elf — PAI-M0 bring-up harness
 *
 * First vertical slice (whitepaper §43):
 *
 *   bootstrap -> capability detection -> GPU memory allocation ->
 *   command submission -> simple tensor kernel -> readback ->
 *   CPU reference comparison -> benchmark
 *
 * Stage A proves the /dev/gc submission path with a raw IT_DMA_DATA
 * copy (hardware-proven packet layout).
 * Stage B dispatches the gfx1013 vecadd compute kernel and validates
 * the result against the CPU reference backend.
 * Stage C measures sustained dispatch+wait latency.
 *
 * Exit code: bit 0 = stage A failed, bit 1 = stage B failed.
 */

#include <pai/api.h>
#include <pai/log.h>

#include <bench.h>
#include <hal/hal.h>
#include <host_kernels.h>
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
      pai_gpu_buffer_alloc(g, &ctx->a, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->b, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->c, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->code, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK ||
      pai_gpu_buffer_alloc(g, &ctx->label, sz, PAI_GPU_BUF_CPU_VISIBLE) !=
          PAI_OK) {
    return -1;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Stage A: raw IT_DMA_DATA copy through /dev/gc                       */
/* ------------------------------------------------------------------ */

static int
m0_stage_a(m0_ctx_t *ctx) {
  uint32_t pm4[M0_PM4_CAP];
  pai_pm4_builder_t pb;
  uint32_t *pattern = (uint32_t *)ctx->src.cpu_addr;
  pai_status_t st;

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-A] GPU DMA copy (%u KiB)\n",
                M0_DMA_BYTES >> 10);

  for (uint32_t i = 0; i < M0_DMA_BYTES / 4; i++) {
    pattern[i] = 0x9E370001u ^ (i * 0x85EBCA6Bu);
  }
  memset(ctx->dst.cpu_addr, 0, M0_DMA_BYTES);

  pai_pm4_builder_init(&pb, pm4, M0_PM4_CAP);
  if (!pai_pm4_dma_data(&pb, ctx->src.gpu_addr, ctx->dst.gpu_addr,
                        M0_DMA_BYTES)) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-A] pm4 build failed\n");
    return -1;
  }

  ctx->label_value++;
  *(volatile uint32_t *)ctx->label.cpu_addr = 0;
  st = pai_gpu_submit_wait(ctx->gpu, pm4, pb.len, ctx->label.gpu_addr,
                           ctx->label_value, M0_TIMEOUT_NS);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-A] submit failed: %s\n",
                   pai_status_str(st));
    return -1;
  }

  if (memcmp(ctx->src.cpu_addr, ctx->dst.cpu_addr, M0_DMA_BYTES) != 0) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-A] FAIL: destination mismatch\n");
    return -1;
  }

  PAI_LOG_INFO_(PAI_SUB_GPU, "[M0-A] PASS: GPU DMA verified\n");
  return 0;
}

/* ------------------------------------------------------------------ */
/* Stage B: gfx1013 vecadd compute kernel vs CPU reference             */
/* ------------------------------------------------------------------ */

static void
m0_build_vecadd_stream(m0_ctx_t *ctx, uint32_t *pm4, uint32_t cap,
                       uint32_t *out_len) {
  pai_pm4_builder_t pb;
  uint32_t vals[8];
  uint64_t code = ctx->code.gpu_addr;

  pai_pm4_builder_init(&pb, pm4, cap);

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
  uint32_t pm4[M0_PM4_CAP];
  uint32_t pm4_len;
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
  memset(c, 0xCC, n * sizeof(float));

  /* Upload the gfx1013 kernel into GPU-visible memory. */
  memcpy(ctx->code.cpu_addr, pai_vecadd_code,
         PAI_VECADD_CODE_WORDS * sizeof(uint32_t));

  /* Host reference backend: bind the emulated kernel to the code addr. */
  if (pai_gpu_device_backend(ctx->gpu) == PAI_GPU_BACKEND_HOST_REF) {
    st = pai_gpu_host_register_shader(ctx->gpu, ctx->code.gpu_addr,
                                      pai_host_kernel_vecadd, NULL);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B] shader registration failed\n");
      return -1;
    }
  }

  m0_build_vecadd_stream(ctx, pm4, M0_PM4_CAP, &pm4_len);

  ctx->label_value++;
  *(volatile uint32_t *)ctx->label.cpu_addr = 0;
  st = pai_gpu_submit_wait(ctx->gpu, pm4, pm4_len, ctx->label.gpu_addr,
                           ctx->label_value, M0_TIMEOUT_NS);
  if (st != PAI_OK) {
    PAI_LOG_ERROR_(PAI_SUB_GPU, "[M0-B] submit failed: %s\n",
                   pai_status_str(st));
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

/* ------------------------------------------------------------------ */
/* Stage C: dispatch benchmark                                         */
/* ------------------------------------------------------------------ */

static void
m0_stage_c_run(void *user) {
  m0_ctx_t *ctx = (m0_ctx_t *)user;
  uint32_t pm4[M0_PM4_CAP];
  uint32_t pm4_len;

  m0_build_vecadd_stream(ctx, pm4, M0_PM4_CAP, &pm4_len);
  ctx->label_value++;
  *(volatile uint32_t *)ctx->label.cpu_addr = 0;
  if (pai_gpu_submit_wait(ctx->gpu, pm4, pm4_len, ctx->label.gpu_addr,
                          ctx->label_value, M0_TIMEOUT_NS) != PAI_OK) {
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

  /* Replace any previous payload instance still running (deploy
   * automation, same pattern as MemDBG). Best effort: never fatal. */
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

  if (m0_stage_b(&ctx) != 0) {
    fail |= M0_STAGE_B_FAIL;
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
