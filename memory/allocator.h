/*
 * ProsperoAI — buddy suballocator (whitepaper §16, GPU allocator
 * foundation). Power-of-two blocks, O(log n) split/merge, strict
 * alignment, low fragmentation over one large region (device memory or
 * a host test region). Region must be a power of two >= 128 bytes and
 * aligned to its size.
 */

#ifndef PAI_MEMORY_ALLOCATOR_H
#define PAI_MEMORY_ALLOCATOR_H

#include <pai/error.h>

#include <stdint.h>

#define PAI_BUDDY_MIN_BLOCK 64u   /* 2^6 */
#define PAI_BUDDY_MIN_SIZE 128u
#define PAI_BUDDY_MAX_ORDER 30u   /* region <= 2^30 bytes */

typedef struct pai_buddy_block {
  struct pai_buddy_block *next;
} pai_buddy_block_t;

typedef struct pai_buddy {
  uint8_t *base;                 /* region base, aligned to size   */
  uint64_t size;                 /* region bytes, power of two     */
  uint32_t max_order;            /* log2(size)                     */
  pai_buddy_block_t *free_lists[PAI_BUDDY_MAX_ORDER + 1];
  void *owned;                   /* malloc base if we own the region */
  uint64_t used_bytes;           /* sum of allocated block sizes    */
  uint64_t allocs;
  uint64_t frees;
  uint64_t splits;
  uint64_t merges;
} pai_buddy_t;

/*
 * Init over caller-owned memory. size must be a power of two >= 128 and
 * `memory` must be aligned to size. Returns PAI_ERR_INVALID_ARG otherwise.
 */
pai_status_t pai_buddy_init(pai_buddy_t *buddy, void *memory, uint64_t size);

/*
 * Init over an internally allocated region. `size` must be a power of
 * two >= 128 and small enough to allocate. Destroy with pai_buddy_destroy.
 */
pai_status_t pai_buddy_init_owned(pai_buddy_t *buddy, uint64_t size);

/* Release an owned region. No-op for caller-owned regions. */
void pai_buddy_destroy(pai_buddy_t *buddy);

/*
 * Allocate `size` bytes aligned to `align` (power of two, >= 1).
 * Returns NULL when the region cannot satisfy the request.
 */
void *pai_buddy_alloc(pai_buddy_t *buddy, uint64_t size, uint64_t align);

/* Return a pointer from pai_buddy_alloc. Invalid pointers are UB. */
void pai_buddy_free(pai_buddy_t *buddy, void *ptr);

uint64_t pai_buddy_used(const pai_buddy_t *buddy);
uint64_t pai_buddy_avail(const pai_buddy_t *buddy);

#endif /* PAI_MEMORY_ALLOCATOR_H */
