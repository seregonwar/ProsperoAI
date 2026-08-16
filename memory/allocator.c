#include "allocator.h"
#include "mem.h"

#include <stdlib.h>
#include <string.h>

/*
 * Block layout (block of order o, base aligned to 2^o):
 *
 *   [0..8)   magic   — PAI_BUDDY_MAGIC_FREE while free, MAGIC_ALLOC while used
 *   [8..12)  order   — block order, valid in both states
 *   [12..16) padding
 *   [16..24) next    — free-list link (free blocks only)
 *   [24..32) padding
 *   [32..)   payload — starts at align_up(base + 32, align); a backpointer
 *                      to the block base is stored at payload - 8.
 *
 * The header is never touched by user data, which makes buddy-merge
 * detection exact: a buddy is free iff its magic is MAGIC_FREE. The
 * free-list link lives at +16 so it can never clobber the magic/order.
 */

#define PAI_BUDDY_MAGIC_FREE  UINT64_C(0x4652454546414942) /* "FREEFAIB" */
#define PAI_BUDDY_MAGIC_ALLOC UINT64_C(0x414C4C4F46414942) /* "ALLOFAIB" */
#define PAI_BUDDY_HEADER      32u

static uint64_t
block_off(const pai_buddy_t *buddy, const uint8_t *block) {
  return (uint64_t)(block - buddy->base);
}

static int
block_is_free(const uint8_t *block, uint32_t order) {
  return block != NULL && *(const uint64_t *)block == PAI_BUDDY_MAGIC_FREE &&
         *(const uint32_t *)(block + 8) == order;
}

static void
block_set_header(uint8_t *block, uint64_t magic, uint32_t order) {
  *(uint64_t *)block = magic;
  *(uint32_t *)(block + 8) = order;
}

static void
free_list_push(pai_buddy_t *buddy, uint32_t order, uint8_t *block) {
  block_set_header(block, PAI_BUDDY_MAGIC_FREE, order);
  *(void **)(block + 16) = buddy->free_lists[order];
  buddy->free_lists[order] = (pai_buddy_block_t *)block;
}

static uint8_t *
free_list_pop(pai_buddy_t *buddy, uint32_t order) {
  pai_buddy_block_t *node = buddy->free_lists[order];

  if (node == NULL) {
    return NULL;
  }
  buddy->free_lists[order] = *(pai_buddy_block_t **)((uint8_t *)node + 16);
  return (uint8_t *)node;
}

static void
free_list_remove(pai_buddy_t *buddy, uint32_t order, uint8_t *block) {
  pai_buddy_block_t **link = &buddy->free_lists[order];
  pai_buddy_block_t *node;

  while ((node = *link) != NULL) {
    if ((uint8_t *)node == block) {
      *link = *(pai_buddy_block_t **)((uint8_t *)node + 16);
      return;
    }
    link = (pai_buddy_block_t **)((uint8_t *)node + 16);
  }
}

pai_status_t
pai_buddy_init(pai_buddy_t *buddy, void *memory, uint64_t size) {
  if (!buddy || !memory || !pai_mem_is_pow2(size) || size < PAI_BUDDY_MIN_SIZE ||
      (uintptr_t)memory % size != 0) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(buddy, 0, sizeof(*buddy));
  buddy->base = (uint8_t *)memory;
  buddy->size = size;
  buddy->max_order = pai_mem_floor_log2(size);
  free_list_push(buddy, buddy->max_order, buddy->base);
  return PAI_OK;
}

pai_status_t
pai_buddy_init_owned(pai_buddy_t *buddy, uint64_t size) {
  uint8_t *raw;
  uint8_t *base;
  pai_status_t st;

  if (!buddy || !pai_mem_is_pow2(size) || size < PAI_BUDDY_MIN_SIZE ||
      size > (UINT64_C(1) << PAI_BUDDY_MAX_ORDER)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (size > SIZE_MAX / 2) {
    return PAI_ERR_NOMEM;
  }

  raw = (uint8_t *)malloc((size_t)(size * 2));
  if (raw == NULL) {
    return PAI_ERR_NOMEM;
  }
  base = (uint8_t *)((uintptr_t)raw + (size - ((uintptr_t)raw % size)) % size);

  st = pai_buddy_init(buddy, base, size);
  if (st != PAI_OK) {
    free(raw);
    return st;
  }
  buddy->owned = raw;
  return PAI_OK;
}

void
pai_buddy_destroy(pai_buddy_t *buddy) {
  if (buddy != NULL) {
    free(buddy->owned);
    buddy->owned = NULL;
  }
}

void *
pai_buddy_alloc(pai_buddy_t *buddy, uint64_t size, uint64_t align) {
  uint32_t order;
  uint32_t found;
  uint8_t *block;
  uint8_t *ptr;

  if (!buddy || size == 0 || align == 0 || !pai_mem_is_pow2(align) ||
      size > UINT64_MAX - align || size + align > UINT64_MAX - PAI_BUDDY_HEADER) {
    return NULL;
  }

  order = pai_mem_ceil_log2(size + align + PAI_BUDDY_HEADER);
  if (order < 6) {
    order = 6; /* PAI_BUDDY_MIN_BLOCK */
  }
  if (order > buddy->max_order) {
    return NULL;
  }

  /* Smallest order >= requested with a free block. */
  found = order;
  while (found <= buddy->max_order && buddy->free_lists[found] == NULL) {
    found++;
  }
  if (found > buddy->max_order) {
    return NULL;
  }

  block = free_list_pop(buddy, found);

  /* Split down to the target order, freeing the right halves. */
  while (found > order) {
    uint8_t *half;
    found--;
    half = block + (UINT64_C(1) << found);
    block_set_header(half, PAI_BUDDY_MAGIC_FREE, found);
    free_list_push(buddy, found, half);
    buddy->splits++;
  }

  block_set_header(block, PAI_BUDDY_MAGIC_ALLOC, order);
  buddy->used_bytes += UINT64_C(1) << order;
  buddy->allocs++;

  ptr = (uint8_t *)pai_mem_align_up((uint64_t)(uintptr_t)(block + PAI_BUDDY_HEADER),
                                    align);
  *(uint64_t *)(ptr - 8) = (uint64_t)(uintptr_t)block;
  return ptr;
}

void
pai_buddy_free(pai_buddy_t *buddy, void *ptr) {
  uint8_t *block;
  uint32_t order;

  if (buddy == NULL || ptr == NULL) {
    return;
  }

  block = (uint8_t *)(uintptr_t)(*(uint64_t *)((uint8_t *)ptr - 8));
  if (block < buddy->base || block >= buddy->base + buddy->size) {
    return; /* foreign pointer: ignore */
  }
  if (*(uint64_t *)block != PAI_BUDDY_MAGIC_ALLOC) {
    return; /* double free / foreign pointer: ignore */
  }

  order = *(uint32_t *)(block + 8);
  buddy->used_bytes -= UINT64_C(1) << order;
  buddy->frees++;

  block_set_header(block, PAI_BUDDY_MAGIC_FREE, order);

  /* Merge with the buddy while it is free. */
  while (order < buddy->max_order) {
    uint64_t sz = UINT64_C(1) << order;
    uint64_t buddy_off = (uint64_t)block_off(buddy, block) ^ sz;
    uint8_t *buddy_block = buddy->base + buddy_off;

    if (buddy_off >= buddy->size || !block_is_free(buddy_block, order)) {
      break;
    }
    free_list_remove(buddy, order, buddy_block);
    block = buddy_block < block ? buddy_block : block;
    order++;
    buddy->merges++;
  }

  free_list_push(buddy, order, block);
}

uint64_t
pai_buddy_used(const pai_buddy_t *buddy) {
  return buddy ? buddy->used_bytes : 0;
}

uint64_t
pai_buddy_avail(const pai_buddy_t *buddy) {
  return buddy ? buddy->size - buddy->used_bytes : 0;
}
