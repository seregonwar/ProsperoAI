/*
 * ProsperoAI — profiler
 *
 * Minimal timing/benchmark primitives feeding the PAI-M0 harness
 * (whitepaper §30/§31). Grows into the full profiling subsystem in
 * later phases.
 */

#ifndef PAI_PROFILER_BENCH_H
#define PAI_PROFILER_BENCH_H

#include <stdint.h>

/* Monotonic timestamp in nanoseconds. */
uint64_t pai_clock_ns(void);

typedef struct pai_bench_result {
  uint32_t iters;
  uint64_t total_ns;
  uint64_t best_ns;
  uint64_t mean_ns;
} pai_bench_result_t;

/*
 * Run `fn` (warmup + iters) times and summarize. fn receives ctx.
 */
void pai_bench_run(pai_bench_result_t *out, uint32_t warmup, uint32_t iters,
                   void (*fn)(void *ctx), void *ctx);

/* Convenience: benchmark over a buffer size. */
double pai_bench_gib_s(const pai_bench_result_t *result, uint64_t bytes);

#endif /* PAI_PROFILER_BENCH_H */
