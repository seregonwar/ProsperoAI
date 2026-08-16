/*
 * ProsperoAI — memory subsystem
 *
 * Small shared helpers used by the static planner and the suballocator.
 */

#ifndef PAI_MEMORY_MEM_H
#define PAI_MEMORY_MEM_H

#include <stdint.h>

static inline int
pai_mem_is_pow2(uint64_t v) {
  return v != 0 && (v & (v - 1)) == 0;
}

/* Round v up to a power-of-two alignment. v <= UINT64_MAX - (align - 1). */
static inline uint64_t
pai_mem_align_up(uint64_t v, uint64_t align) {
  return (v + align - 1) & ~(align - 1);
}

/* Round v down to a power-of-two alignment. */
static inline uint64_t
pai_mem_align_down(uint64_t v, uint64_t align) {
  return v & ~(align - 1);
}

/*
 * Smallest o >= 0 such that (1 << o) >= v. Returns 0 for v == 0,
 * 63 for any value needing more than 63 bits.
 */
static inline uint32_t
pai_mem_ceil_log2(uint64_t v) {
  uint64_t p = 1;
  uint32_t o = 0;

  if (v == 0) {
    return 0;
  }
  while (p < v && o < 63) {
    p <<= 1;
    o++;
  }
  return o;
}

/* Largest o >= 0 such that (1 << o) <= v. Returns 0 for v == 0. */
static inline uint32_t
pai_mem_floor_log2(uint64_t v) {
  uint32_t o = 0;

  while (v > 1) {
    v >>= 1;
    o++;
  }
  return o;
}

#endif /* PAI_MEMORY_MEM_H */
