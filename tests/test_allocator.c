#include "test.h"

#include <allocator.h>

#include <stdint.h>
#include <string.h>

#define REGION_SIZE (UINT64_C(1) << 20) /* 1 MiB */

typedef struct live_alloc {
  void *ptr;
  uint64_t size;
} live_alloc_t;

/* Shadow check: no two simultaneously-live allocations may overlap. */
static void
check_live_no_overlap(const live_alloc_t *live, uint32_t n, void *ptr,
                      uint64_t size) {
  for (uint32_t i = 0; i < n; i++) {
    uintptr_t a = (uintptr_t)live[i].ptr;
    uintptr_t b = (uintptr_t)ptr;
    CHECK(!(a < b + size && b < a + live[i].size));
  }
}

TEST_MAIN_BEGIN()

{
  /* Caller-owned region, power-of-two size, aligned base. */
  static uint8_t raw[512 * 2];
  pai_buddy_t buddy;
  uint8_t *base;
  void *p1, *p2;

  base = (uint8_t *)((uintptr_t)raw + (512 - ((uintptr_t)raw % 512)) % 512);
  CHECK_EQ_INT(pai_buddy_init(&buddy, base, 512), PAI_OK);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), 512);

  p1 = pai_buddy_alloc(&buddy, 64, 64);
  CHECK(p1 != NULL);
  CHECK_EQ_UINT((uintptr_t)p1 % 64, 0);

  p2 = pai_buddy_alloc(&buddy, 64, 64);
  CHECK(p2 != NULL);

  pai_buddy_free(&buddy, p1);
  pai_buddy_free(&buddy, p2);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), 512); /* fully merged */

  p1 = pai_buddy_alloc(&buddy, 64, 64);
  CHECK(p1 != NULL);
  pai_buddy_free(&buddy, p1);
  pai_buddy_destroy(&buddy);
}

{
  /* Init validation. */
  pai_buddy_t buddy;
  static uint8_t raw[512 * 2];
  uint8_t *base = (uint8_t *)((uintptr_t)raw + (512 - ((uintptr_t)raw % 512)) % 512);

  CHECK_EQ_INT(pai_buddy_init(&buddy, NULL, 512), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_buddy_init(&buddy, base, 300), PAI_ERR_INVALID_ARG);  /* not pow2 */
  CHECK_EQ_INT(pai_buddy_init(&buddy, base, 64), PAI_ERR_INVALID_ARG);   /* too small */
  CHECK_EQ_INT(pai_buddy_init(&buddy, base + 1, 512), PAI_ERR_INVALID_ARG); /* misaligned */
}

{
  /* Owned region: split/merge counters and availability accounting. */
  pai_buddy_t buddy;
  void *p;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);

  p = pai_buddy_alloc(&buddy, 1024, 16);
  CHECK(p != NULL);
  CHECK_EQ_UINT((uintptr_t)p % 16, 0);
  CHECK(buddy.splits >= 9); /* 2^20 -> 2^11 */
  CHECK_EQ_UINT(pai_buddy_used(&buddy) + pai_buddy_avail(&buddy), REGION_SIZE);

  pai_buddy_free(&buddy, p);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE); /* re-merged */
  CHECK(buddy.merges >= 9);

  /* Exhaustion: 512 KiB + 512 KiB = full region. */
  p = pai_buddy_alloc(&buddy, REGION_SIZE / 2, 16);
  CHECK(p != NULL);
  CHECK(pai_buddy_alloc(&buddy, REGION_SIZE / 2, 16) == NULL);
  CHECK(pai_buddy_alloc(&buddy, 1, 1) == NULL);
  pai_buddy_free(&buddy, p);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);

  pai_buddy_destroy(&buddy);
}

{
  /* Alignment larger than the payload. */
  pai_buddy_t buddy;
  void *p;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);

  p = pai_buddy_alloc(&buddy, 8, 128);
  CHECK(p != NULL);
  CHECK_EQ_UINT((uintptr_t)p % 128, 0);
  pai_buddy_free(&buddy, p);

  p = pai_buddy_alloc(&buddy, 1, 4096);
  CHECK(p != NULL);
  CHECK_EQ_UINT((uintptr_t)p % 4096, 0);
  pai_buddy_free(&buddy, p);

  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);
  pai_buddy_destroy(&buddy);
}

{
  /* Deterministic multi-level merge: a deep split re-merges in order. */
  pai_buddy_t buddy;
  void *small, *large;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);

  small = pai_buddy_alloc(&buddy, 64, 64);  /* order 8: 12 splits */
  large = pai_buddy_alloc(&buddy, REGION_SIZE / 4, 16); /* order 19 */
  CHECK(small != NULL && large != NULL);

  /* Freeing the small block merges up to its allocated sibling. The
   * 256 KiB request rounds up to a 512 KiB (order 19) buddy block. */
  pai_buddy_free(&buddy, small);
  CHECK(buddy.merges >= 10);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE / 2);

  /* Freeing the large sibling merges everything back. */
  pai_buddy_free(&buddy, large);
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);
  CHECK(buddy.merges >= 11);

  pai_buddy_destroy(&buddy);
}

{
  /* Invalid allocation requests. */
  pai_buddy_t buddy;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);
  CHECK(pai_buddy_alloc(&buddy, 0, 16) == NULL);
  CHECK(pai_buddy_alloc(&buddy, 16, 0) == NULL);
  CHECK(pai_buddy_alloc(&buddy, 16, 3) == NULL); /* non-power-of-two align */
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);
  pai_buddy_destroy(&buddy);
}

{
  /* Double free is ignored, accounting stays intact. */
  pai_buddy_t buddy;
  void *p;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);
  p = pai_buddy_alloc(&buddy, 256, 16);
  CHECK(p != NULL);
  pai_buddy_free(&buddy, p);
  pai_buddy_free(&buddy, p); /* ignored */
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);
  pai_buddy_destroy(&buddy);
}

{
  /* Randomized churn: deterministic seed, shadow overlap checking. */
  pai_buddy_t buddy;
  live_alloc_t live[128];
  uint32_t nlive = 0;
  uint32_t seed = 0xA110C09u;

  CHECK_EQ_INT(pai_buddy_init_owned(&buddy, REGION_SIZE), PAI_OK);

  for (uint32_t it = 0; it < 400; it++) {
    uint32_t r = seed = seed * 1103515245u + 12345u;

    if (r % 3 == 0 && nlive > 0) {
      /* Free a random live allocation. */
      uint32_t idx = (r >> 8) % nlive;
      pai_buddy_free(&buddy, live[idx].ptr);
      live[idx] = live[--nlive];
    } else if (nlive < 128) {
      uint64_t size = 1 + ((r >> 4) % 65536);
      uint64_t align = UINT64_C(1) << ((r >> 20) % 8); /* 1..128 */
      void *p = pai_buddy_alloc(&buddy, size, align);

      if (p != NULL) {
        CHECK_EQ_UINT((uintptr_t)p % align, 0);
        check_live_no_overlap(live, nlive, p, size);
        live[nlive].ptr = p;
        live[nlive].size = size;
        nlive++;
      }
    }
    CHECK_EQ_UINT(pai_buddy_used(&buddy) + pai_buddy_avail(&buddy),
                  REGION_SIZE);
  }

  for (uint32_t i = 0; i < nlive; i++) {
    pai_buddy_free(&buddy, live[i].ptr);
  }
  CHECK_EQ_UINT(pai_buddy_avail(&buddy), REGION_SIZE);
  CHECK(buddy.merges > 0);
  pai_buddy_destroy(&buddy);
}

TEST_MAIN_END()
