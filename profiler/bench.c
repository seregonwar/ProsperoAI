#include "bench.h"

#include <time.h>

#ifdef _WIN32
#include <windows.h>

uint64_t
pai_clock_ns(void) {
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER now;

  if (freq.QuadPart == 0) {
    QueryPerformanceFrequency(&freq);
  }
  QueryPerformanceCounter(&now);
  return (uint64_t)((double)now.QuadPart * 1000000000.0 /
                    (double)freq.QuadPart);
}
#else
uint64_t
pai_clock_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}
#endif

void
pai_bench_run(pai_bench_result_t *out, uint32_t warmup, uint32_t iters,
              void (*fn)(void *ctx), void *ctx) {
  uint64_t start, end;
  uint64_t total = 0;
  uint64_t best = UINT64_MAX;

  out->iters = iters;
  out->total_ns = 0;
  out->best_ns = 0;
  out->mean_ns = 0;

  for (uint32_t i = 0; i < warmup; i++) {
    fn(ctx);
  }

  for (uint32_t i = 0; i < iters; i++) {
    start = pai_clock_ns();
    fn(ctx);
    end = pai_clock_ns();

    uint64_t dt = end - start;
    total += dt;
    if (dt < best) {
      best = dt;
    }
  }

  out->total_ns = total;
  out->best_ns = best;
  out->mean_ns = total / iters;
}

double
pai_bench_gib_s(const pai_bench_result_t *result, uint64_t bytes) {
  if (result->mean_ns == 0) {
    return 0.0;
  }
  return (double)bytes / (double)result->mean_ns;
}
