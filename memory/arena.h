/*
 * ProsperoAI — bump arena allocator (whitepaper §16).
 * Dedicated arenas per lifetime class; generic heap stays off the hot
 * paths.
 */

#ifndef PAI_MEMORY_ARENA_H
#define PAI_MEMORY_ARENA_H

#include <pai/error.h>

#include <stdint.h>

typedef struct pai_arena {
  uint8_t *base;
  uint64_t size;
  uint64_t used;
} pai_arena_t;

void pai_arena_init(pai_arena_t *arena, void *memory, uint64_t size);

/* Allocate from the arena (NULL when out of space). align = pow2. */
void *pai_arena_alloc(pai_arena_t *arena, uint64_t size, uint32_t align);

/* Free everything allocated so far. */
void pai_arena_reset(pai_arena_t *arena);

uint64_t pai_arena_used(const pai_arena_t *arena);
uint64_t pai_arena_free(const pai_arena_t *arena);

#endif /* PAI_MEMORY_ARENA_H */
