#include "bench.h"

#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint64_t
pai_bench_now_ns(void) {
  LARGE_INTEGER freq;
  LARGE_INTEGER counter;

  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&counter);
  if (freq.QuadPart <= 0) {
    return 0;
  }
  return (uint64_t)((counter.QuadPart * 1000000000LL) / freq.QuadPart);
}

#else

#include <time.h>

uint64_t
pai_bench_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

#endif

uint64_t
pai_bench_median(uint64_t *samples, uint32_t n) {
  /* Simple insertion sort over a small sample set. */
  for (uint32_t i = 1; i < n; i++) {
    uint64_t key = samples[i];
    uint32_t j = i;
    while (j > 0 && samples[j - 1] > key) {
      samples[j] = samples[j - 1];
      j--;
    }
    samples[j] = key;
  }
  return samples[n / 2];
}

uint64_t
pai_bench_min(const uint64_t *samples, uint32_t n) {
  uint64_t m = samples[0];
  for (uint32_t i = 1; i < n; i++) {
    if (samples[i] < m) {
      m = samples[i];
    }
  }
  return m;
}
