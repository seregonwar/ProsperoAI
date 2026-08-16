/*
 * ProsperoAI — Prospero Protocol
 *
 * Shared monotonic clock (ns) for protocol-side client tooling: the
 * ping health check and the gateway remote bridge both need deadlines
 * without depending on the profiler. Host-side only.
 */

#include <protocol/protocol.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

uint64_t
pai_proto_now_ns(void) {
  static LARGE_INTEGER freq;
  static int have_freq = 0;
  LARGE_INTEGER c;
  if (!have_freq) {
    QueryPerformanceFrequency(&freq);
    have_freq = 1;
  }
  QueryPerformanceCounter(&c);
  return (uint64_t)((double)c.QuadPart * 1e9 / (double)freq.QuadPart);
}
#else
#include <time.h>

uint64_t
pai_proto_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif
