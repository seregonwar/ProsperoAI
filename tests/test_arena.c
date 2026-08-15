#include "test.h"

#include <arena.h>

#include <stdint.h>

TEST_MAIN_BEGIN()

{
  static uint8_t mem[4096];
  pai_arena_t arena;
  void *p1, *p2, *p3;

  pai_arena_init(&arena, mem, sizeof(mem));

  p1 = pai_arena_alloc(&arena, 64, 16);
  CHECK(p1 != NULL);
  CHECK_EQ_UINT((uintptr_t)p1 & 15, 0);

  p2 = pai_arena_alloc(&arena, 64, 64);
  CHECK(p2 != NULL);
  CHECK_EQ_UINT((uintptr_t)p2 & 63, 0);

  /* alignment padding must not overlap previous allocation */
  CHECK((uintptr_t)p2 >= (uintptr_t)p1 + 64);

  CHECK_EQ_UINT(pai_arena_used(&arena), (uintptr_t)p2 - (uintptr_t)mem + 64);

  pai_arena_reset(&arena);
  CHECK_EQ_UINT(pai_arena_used(&arena), 0);

  p3 = pai_arena_alloc(&arena, 64, 16);
  CHECK_EQ_UINT((uintptr_t)p3, (uintptr_t)p1);
}

{
  static uint8_t mem[256];
  pai_arena_t arena;

  pai_arena_init(&arena, mem, sizeof(mem));
  CHECK(pai_arena_alloc(&arena, 300, 16) == NULL);
  CHECK(pai_arena_alloc(&arena, 64, 3) == NULL); /* non-power-of-two align */
  CHECK_EQ_UINT(pai_arena_used(&arena), 0);
}

TEST_MAIN_END()
