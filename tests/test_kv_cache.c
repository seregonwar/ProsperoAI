#include "test.h"

#include <kv_cache.h>

#include <string.h>

TEST_MAIN_BEGIN()

{
  /* Owned init + reserve + touch semantics. */
  pai_kv_cache_t kv;
  uint32_t pos;
  uint32_t i;
  uint64_t h = 0;

  CHECK_EQ_INT(pai_kv_cache_init_owned(&kv, 4096, 2, 8), PAI_OK);
  /* Per-position cost 2*8=16 B rounds up to a 64 B buddy block, so the
   * 4096 B region holds 64 positions (not 128). */
  CHECK_EQ_UINT(pai_kv_cache_capacity_bytes(&kv), 64 * 2 * 8);

  /* Reserve positions. */
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 1, &pos), PAI_OK);
  CHECK_EQ_UINT(pos, 0);
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 5, &pos), PAI_OK);
  CHECK_EQ_UINT(pos, 1);
  CHECK_EQ_UINT(kv.positions, 6);

  /* First touch on a fresh position: miss, then matching hash: hit. */
  h = pai_kv_prefix_hash(0, 7);
  CHECK_EQ_INT(pai_kv_cache_touch(&kv, 0, h), 0);
  CHECK_EQ_INT(pai_kv_cache_touch(&kv, 0, h), 1);
  CHECK_EQ_UINT(kv.stats_prefix_miss, 1);
  CHECK_EQ_UINT(kv.stats_prefix_hits, 1);

  /* Different hash at the same position: miss again. */
  CHECK_EQ_INT(pai_kv_cache_touch(&kv, 0, 0x1234), 0);
  CHECK_EQ_UINT(kv.stats_prefix_miss, 2);

  /* Rolling prefix hash is incremental. */
  {
    uint64_t a = pai_kv_prefix_hash(pai_kv_prefix_hash(0, 1), 2);
    uint64_t b = pai_kv_prefix_hash(pai_kv_prefix_hash(0, 1), 2);
    CHECK_EQ_UINT(a, b);
    CHECK(a != pai_kv_prefix_hash(0, 1));
  }

  /* Reset returns to empty. */
  pai_kv_cache_reset(&kv);
  CHECK_EQ_UINT(kv.positions, 0);
  CHECK_EQ_UINT(kv.stats_prefix_miss, 0);
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 3, &pos), PAI_OK);
  CHECK_EQ_UINT(pos, 0);

  /* Exhaustion: the region cannot hold more than its capacity. */
  for (i = 0; i < kv.capacity_tokens + 2; i++) {
    pai_status_t st = pai_kv_cache_reserve(&kv, 1, &pos);
    if (st == PAI_ERR_NOMEM) {
      break;
    }
    CHECK_EQ_INT(st, PAI_OK);
  }
  CHECK(kv.positions <= kv.capacity_tokens);
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 1, &pos), PAI_ERR_NOMEM);

  pai_kv_cache_destroy(&kv);
}

{
  /* Caller-owned region init. */
  static uint8_t raw[8192 * 2];
  pai_kv_cache_t kv;
  uint8_t *base;
  uint32_t pos;

  base = (uint8_t *)((uintptr_t)raw + (8192 - ((uintptr_t)raw % 8192)) % 8192);
  CHECK_EQ_INT(pai_kv_cache_init(&kv, base, 8192, 1, 16), PAI_OK);
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 2, &pos), PAI_OK);
  CHECK_EQ_UINT(pos, 0);
  pai_kv_cache_destroy(&kv);
}

{
  /* Validation. */
  pai_kv_cache_t kv;
  static uint8_t raw[256 * 2];
  uint8_t *base =
      (uint8_t *)((uintptr_t)raw + (256 - ((uintptr_t)raw % 256)) % 256);

  CHECK_EQ_INT(pai_kv_cache_init(&kv, NULL, 256, 1, 8), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_kv_cache_init(&kv, base, 300, 1, 8), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_kv_cache_init(&kv, base, 256, 0, 8), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_kv_cache_init(&kv, base, 256, 1, 0), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_kv_cache_reserve(&kv, 0, NULL), PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
