/*
 * ProsperoAI — benchmark harness (whitepaper §31)
 *
 * Reproducible benchmarking helpers: a monotonic nanosecond clock and
 * statistical summaries. Benchmarks validate correctness before
 * accepting performance numbers, run warmup iterations, and report
 * min/median over the measured samples (median is robust against
 * scheduler noise; min approximates the uncongested cost).
 */

#ifndef PAI_BENCH_H
#define PAI_BENCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Monotonic clock, nanoseconds. */
uint64_t pai_bench_now_ns(void);

/* Sorted-array median; n must be > 0. */
uint64_t pai_bench_median(uint64_t *samples, uint32_t n);

/* Minimum of the samples. */
uint64_t pai_bench_min(const uint64_t *samples, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* PAI_BENCH_H */
