#include "arena.h"

#include <stddef.h>

void
pai_arena_init(pai_arena_t *arena, void *memory, uint64_t size) {
  arena->base = (uint8_t *)memory;
  arena->size = size;
  arena->used = 0;
}

void *
pai_arena_alloc(pai_arena_t *arena, uint64_t size, uint32_t align) {
  uintptr_t base = (uintptr_t)arena->base;
  uintptr_t cur = base + arena->used;
  uintptr_t aligned;
  uint64_t offset;

  if (align == 0 || (align & (align - 1)) != 0) {
    return NULL;
  }

  aligned = (cur + align - 1) & ~(uintptr_t)(align - 1);
  offset = aligned - base;

  if (offset > arena->size || size > arena->size - offset) {
    return NULL;
  }

  arena->used = offset + size;
  return (void *)aligned;
}

void
pai_arena_reset(pai_arena_t *arena) {
  arena->used = 0;
}

uint64_t
pai_arena_used(const pai_arena_t *arena) {
  return arena->used;
}

uint64_t
pai_arena_free(const pai_arena_t *arena) {
  return arena->size - arena->used;
}
