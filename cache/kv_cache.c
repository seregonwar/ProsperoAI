#include "kv_cache.h"

#include <stdlib.h>
#include <string.h>

pai_status_t
pai_kv_cache_init(pai_kv_cache_t *cache, void *memory, uint64_t region_size,
                  uint32_t num_layers, uint32_t kv_bytes_per_token) {
  pai_status_t st;
  uint64_t block_tokens_cap;

  if (cache == NULL || memory == NULL || num_layers == 0 ||
      num_layers > PAI_KV_MAX_LAYERS || kv_bytes_per_token == 0 ||
      region_size < PAI_BUDDY_MIN_SIZE || (region_size & (region_size - 1))) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(cache, 0, sizeof(*cache));
  st = pai_buddy_init(&cache->buddy, memory, region_size);
  if (st != PAI_OK) {
    return st;
  }
  cache->region = (uint8_t *)memory;
  cache->owned_region = 0;
  cache->region_size = region_size;
  cache->num_layers = num_layers;
  cache->kv_bytes_per_token = kv_bytes_per_token;

  /* The buddy rounds every block up to a power of two (min 64 B), so
   * capacity must be computed from the rounded per-position cost, not
   * the raw bytes, or pai_kv_cache_reserve would fail before the
   * claimed capacity is reached. */
  {
    uint64_t block_bytes = (uint64_t)num_layers * kv_bytes_per_token;
    uint64_t rounded = 64u;
    while (rounded < block_bytes && rounded < (UINT64_C(1) << 30)) {
      rounded <<= 1;
    }
    block_tokens_cap = region_size / rounded;
    if (block_tokens_cap > PAI_KV_BLOCK_TOKENS) {
      block_tokens_cap = PAI_KV_BLOCK_TOKENS;
    }
    if (block_tokens_cap == 0) {
      pai_buddy_destroy(&cache->buddy);
      memset(cache, 0, sizeof(*cache));
      return PAI_ERR_INVALID_ARG;
    }
    cache->capacity_tokens = (uint32_t)block_tokens_cap;
  }

  cache->block_ptrs = (uint64_t *)calloc(cache->capacity_tokens,
                                         sizeof(uint64_t));
  cache->prefix_hashes = (uint64_t *)calloc(cache->capacity_tokens,
                                            sizeof(uint64_t));
  cache->block_bytes_used = (uint32_t *)calloc(cache->capacity_tokens,
                                               sizeof(uint32_t));
  if (cache->block_ptrs == NULL || cache->prefix_hashes == NULL ||
      cache->block_bytes_used == NULL) {
    pai_kv_cache_destroy(cache);
    return PAI_ERR_NOMEM;
  }
  return PAI_OK;
}

pai_status_t
pai_kv_cache_init_owned(pai_kv_cache_t *cache, uint64_t region_size,
                        uint32_t num_layers, uint32_t kv_bytes_per_token) {
  uint8_t *raw;
  uint8_t *base;
  pai_status_t st;

  if (cache == NULL || region_size < PAI_BUDDY_MIN_SIZE ||
      (region_size & (region_size - 1)) ||
      region_size > SIZE_MAX / 2) {
    return PAI_ERR_INVALID_ARG;
  }

  raw = (uint8_t *)malloc((size_t)(region_size * 2));
  if (raw == NULL) {
    return PAI_ERR_NOMEM;
  }
  base = (uint8_t *)((uintptr_t)raw +
                     (region_size - ((uintptr_t)raw % region_size)) %
                         region_size);

  st = pai_kv_cache_init(cache, base, region_size, num_layers,
                         kv_bytes_per_token);
  if (st != PAI_OK) {
    free(raw);
    return st;
  }
  cache->owned_region = 1; /* init() memsets the struct; re-assert after */
  /* Hand the raw allocation to the buddy so its destroy frees it. */
  cache->buddy.owned = raw;
  return PAI_OK;
}

void
pai_kv_cache_destroy(pai_kv_cache_t *cache) {
  if (cache == NULL) {
    return;
  }
  free(cache->block_ptrs);
  free(cache->prefix_hashes);
  free(cache->block_bytes_used);
  /* Frees the raw owned allocation when pai_kv_cache_init_owned was
   * used (raw base is stored in buddy.owned; buddy_destroy is a no-op
   * for caller-owned regions). */
  if (cache->owned_region) {
    pai_buddy_destroy(&cache->buddy);
  }
  memset(cache, 0, sizeof(*cache));
}

uint64_t
pai_kv_cache_capacity_bytes(const pai_kv_cache_t *cache) {
  if (cache == NULL) {
    return 0;
  }
  return (uint64_t)cache->capacity_tokens * cache->num_layers *
         cache->kv_bytes_per_token;
}

pai_status_t
pai_kv_cache_reserve(pai_kv_cache_t *cache, uint32_t tokens,
                     uint32_t *out_pos) {
  uint64_t block_bytes;
  uint64_t need;
  void *block;

  if (cache == NULL || tokens == 0 || out_pos == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (cache->positions + tokens > cache->capacity_tokens) {
    return PAI_ERR_NOMEM;
  }

  block_bytes = (uint64_t)cache->num_layers * cache->kv_bytes_per_token;
  need = block_bytes * tokens;
  block = pai_buddy_alloc(&cache->buddy, need, 64u);
  if (block == NULL) {
    return PAI_ERR_NOMEM;
  }

  *out_pos = cache->positions;
  cache->block_ptrs[cache->positions] =
      (uint64_t)((uint8_t *)block - cache->region);
  cache->block_bytes_used[cache->positions] = (uint32_t)need;
  cache->positions += tokens;
  cache->stats_blocks++;
  return PAI_OK;
}

int
pai_kv_cache_touch(pai_kv_cache_t *cache, uint32_t pos, uint64_t prefix_hash) {
  if (cache == NULL || pos >= cache->positions) {
    return 0;
  }
  cache->tick++;
  cache->last_touched = cache->tick;
  if (cache->prefix_hashes[pos] == prefix_hash) {
    cache->stats_prefix_hits++;
    return 1;
  }
  cache->prefix_hashes[pos] = prefix_hash;
  cache->stats_prefix_miss++;
  return 0;
}

uint64_t
pai_kv_prefix_hash(uint64_t hash, uint32_t token_id) {
  /* FNV-1a over the token id bytes. */
  hash ^= (uint64_t)(token_id & 0xFF);
  hash *= 0x100000001B3ull;
  hash ^= (uint64_t)((token_id >> 8) & 0xFF);
  hash *= 0x100000001B3ull;
  hash ^= (uint64_t)((token_id >> 16) & 0xFF);
  hash *= 0x100000001B3ull;
  hash ^= (uint64_t)((token_id >> 24) & 0xFF);
  hash *= 0x100000001B3ull;
  return hash;
}

void
pai_kv_cache_reset(pai_kv_cache_t *cache) {
  void *owned = NULL;

  if (cache == NULL) {
    return;
  }
  cache->positions = 0;
  cache->tick = 0;
  cache->last_touched = 0;
  cache->stats_blocks = 0;
  cache->stats_prefix_hits = 0;
  cache->stats_prefix_miss = 0;
  memset(cache->block_ptrs, 0,
         (size_t)cache->capacity_tokens * sizeof(uint64_t));
  memset(cache->prefix_hashes, 0,
         (size_t)cache->capacity_tokens * sizeof(uint64_t));
  memset(cache->block_bytes_used, 0,
         (size_t)cache->capacity_tokens * sizeof(uint32_t));

  /* Reset the buddy free lists without releasing the region. */
  owned = cache->buddy.owned;
  cache->buddy.owned = NULL;
  pai_buddy_destroy(&cache->buddy);
  if (cache->region != NULL) {
    pai_buddy_init(&cache->buddy, cache->region, cache->region_size);
  }
  /* pai_buddy_init memsets the struct: re-assert the owned pointer. */
  cache->buddy.owned = owned;
}
