/*
 * ProsperoAI — GEMM benchmark (§31).
 *   bench_gemm [M,N,K] [--iters N] [--warmup N]
 * Validates correctness against the reference backend before accepting
 * performance numbers, then times the optimized path. Defaults:
 * 512,512,512, 10 iterations, 2 warmup runs.
 */

#include <opt_ops.h>

#include <bench.h>
#include <ref_ops.h>

#include <pai/version.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
parse_u64(const char *s) {
  return strtoull(s, NULL, 10);
}

int
main(int argc, char **argv) {
  uint64_t m = 512;
  uint64_t n = 512;
  uint64_t k = 512;
  uint32_t iters = 10;
  uint32_t warmup = 2;
  float *a;
  float *b;
  float *c_opt;
  float *c_ref;
  int8_t *w_q;
  float *scales;
  uint64_t num_groups;
  uint64_t first_mismatch;
  uint64_t flops;
  uint32_t i;

  /* Parse args: first non-flag arg is "M,N,K". */
  for (int arg = 1; arg < argc; arg++) {
    if (strcmp(argv[arg], "--iters") == 0 && arg + 1 < argc) {
      iters = (uint32_t)parse_u64(argv[++arg]);
    } else if (strcmp(argv[arg], "--warmup") == 0 && arg + 1 < argc) {
      warmup = (uint32_t)parse_u64(argv[++arg]);
    } else if (argv[arg][0] != '-') {
      if (sscanf(argv[arg], "%llu,%llu,%llu", (unsigned long long *)&m,
                 (unsigned long long *)&n, (unsigned long long *)&k) != 3) {
        fprintf(stderr, "usage: bench_gemm [M,N,K] [--iters N] [--warmup N]\n");
        return 2;
      }
    }
  }
  if (m == 0 || n == 0 || k == 0) {
    fprintf(stderr, "bad shape\n");
    return 2;
  }

  pai_opt_init();
  printf("ProsperoAI %u.%u.%u | backend %s | gemm %llux%llux%llu\n",
         PAI_VERSION_MAJOR, PAI_VERSION_MINOR, PAI_VERSION_PATCH,
         pai_opt_backend_name(), (unsigned long long)m, (unsigned long long)n,
         (unsigned long long)k);

  a = (float *)malloc((size_t)(m * k) * sizeof(float));
  b = (float *)malloc((size_t)(k * n) * sizeof(float));
  c_opt = (float *)malloc((size_t)(m * n) * sizeof(float));
  c_ref = (float *)malloc((size_t)(m * n) * sizeof(float));
  if (a == NULL || b == NULL || c_opt == NULL || c_ref == NULL) {
    fprintf(stderr, "out of memory\n");
    return 2;
  }
  srand(42);
  for (uint64_t t = 0; t < m * k; t++) {
    a[t] = (float)(rand() % 2000) / 500.0f - 2.0f;
  }
  for (uint64_t t = 0; t < k * n; t++) {
    b[t] = (float)(rand() % 2000) / 500.0f - 2.0f;
  }

  /* Correctness gate: optimized must match the reference oracle (§37). */
  if (pai_ref_gemm_f32(m, n, k, a, b, c_ref) != PAI_OK ||
      pai_opt_gemm_f32(m, n, k, a, b, c_opt) != PAI_OK ||
      pai_ref_compare_f32(c_opt, c_ref, m * n, 1e-4f, 1e-4f,
                          &first_mismatch) != PAI_OK) {
    fprintf(stderr, "FAIL: optimized GEMM diverges from reference\n");
    return 1;
  }

  /* Warmup + measurement. */
  for (i = 0; i < warmup; i++) {
    pai_opt_gemm_f32(m, n, k, a, b, c_opt);
  }
  flops = 2 * m * n * k;
  {
    uint64_t *samples = (uint64_t *)malloc(iters * sizeof(uint64_t));
    if (samples == NULL) {
      return 2;
    }
    for (i = 0; i < iters; i++) {
      uint64_t t0 = pai_bench_now_ns();
      pai_opt_gemm_f32(m, n, k, a, b, c_opt);
      samples[i] = pai_bench_now_ns() - t0;
    }
    {
      uint64_t med = pai_bench_median(samples, iters);
      uint64_t mn = pai_bench_min(samples, iters);
      printf("f32  gemm: min %5.2f ms  median %5.2f ms  ->  %8.1f GFLOPS "
             "(min) / %8.1f GFLOPS (median)\n",
             (double)mn / 1e6, (double)med / 1e6,
             (double)flops / (double)mn, (double)flops / (double)med);
    }
    free(samples);
  }

  /* w8 quantized GEMM. */
  {
    uint64_t gs = k;
    num_groups = n; /* per-tensor scale per column */
    w_q = (int8_t *)malloc((size_t)(k * n));
    scales = (float *)malloc((size_t)num_groups * sizeof(float));
    if (w_q == NULL || scales == NULL) {
      return 2;
    }
    /* Symmetric per-column scale (group = full column). */
    for (uint64_t j = 0; j < n; j++) {
      float max_abs = 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        float av = b[t * n + j] < 0 ? -b[t * n + j] : b[t * n + j];
        if (av > max_abs) {
          max_abs = av;
        }
      }
      scales[j] = max_abs > 0.0f ? max_abs / 127.0f : 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        int32_t q = scales[j] > 0.0f
                        ? (int32_t)(b[t * n + j] / scales[j] + 0.5f)
                        : 0;
        if (q > 127) {
          q = 127;
        }
        if (q < -127) {
          q = -127;
        }
        w_q[t * n + j] = (int8_t)q;
      }
    }
    if (pai_ref_gemm_w8_f32(m, n, k, a, w_q, scales, gs, c_ref) != PAI_OK ||
        pai_opt_gemm_w8_f32(m, n, k, a, w_q, scales, gs, c_opt) != PAI_OK ||
        pai_ref_compare_f32(c_opt, c_ref, m * n, 1e-3f, 1e-3f,
                            &first_mismatch) != PAI_OK) {
      fprintf(stderr, "FAIL: w8 GEMM diverges from reference\n");
      return 1;
    }
    for (i = 0; i < warmup; i++) {
      pai_opt_gemm_w8_f32(m, n, k, a, w_q, scales, gs, c_opt);
    }
    {
      uint64_t *samples = (uint64_t *)malloc(iters * sizeof(uint64_t));
      if (samples == NULL) {
        return 2;
      }
      for (i = 0; i < iters; i++) {
        uint64_t t0 = pai_bench_now_ns();
        pai_opt_gemm_w8_f32(m, n, k, a, w_q, scales, gs, c_opt);
        samples[i] = pai_bench_now_ns() - t0;
      }
      {
        uint64_t med = pai_bench_median(samples, iters);
        uint64_t mn = pai_bench_min(samples, iters);
        printf("w8   gemm: min %5.2f ms  median %5.2f ms  ->  %8.1f GFLOPS "
               "(min) / %8.1f GFLOPS (median)\n",
               (double)mn / 1e6, (double)med / 1e6, (double)flops / (double)mn,
               (double)flops / (double)med);
      }
      free(samples);
    }
    free(w_q);
    free(scales);
  }

  free(a);
  free(b);
  free(c_opt);
  free(c_ref);
  return 0;
}
