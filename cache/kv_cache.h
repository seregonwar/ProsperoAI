/*
 * ProsperoAI — KV cache manager (whitepaper §19).
 * Per-session cache sized from model metadata (layers x kv bytes/token
 * x context); blocks are suballocated from a single buddy-managed
 * region (§16) so growth is cheap and fragmentation stays low.
 * v0: buddy-backed blocks + prefix caching (rolling FNV-1a hash lets a
 * request reusing a seen prefix skip recomputing it, §19). Cache
 * quantization and spill hooks are reserved for Phase 3/7.
 */

#ifndef PAI_CACHE_KV_CACHE_H
#define PAI_CACHE_KV_CACHE_H

#include <pai/error.h>

#include <allocator.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_KV_MAX_LAYERS 64u

/* Fixed block size in tokens (positions per cache block). */
#define PAI_KV_BLOCK_TOKENS 128u

typedef struct pai_kv_cache {
  pai_buddy_t buddy;          /* buddy suballocator over the region      */
  uint8_t *region;            /* region memory                          */
  uint64_t region_size;       /* power of two, >= PAI_BUDDY_MIN_SIZE     */
  uint32_t owned_region;      /* 1 when pai_kv_cache_init_owned allocated */
  uint32_t num_layers;        /* KV layers (e.g. 2 per transformer block) */
  uint32_t kv_bytes_per_token;/* bytes per layer per token position      */
  uint32_t capacity_tokens;   /* total cached token positions            */

  /* Per-position metadata (allocated blocks). */
  uint64_t *block_ptrs;       /* owned: region offset of each block      */
  uint64_t *prefix_hashes;    /* owned: rolling hash after each position */
  uint32_t *block_bytes_used; /* owned: bytes written per block          */
  uint32_t positions;         /* number of cached token positions        */

  /* Scheduler integration (§19: KV residency is part of global
   * scheduling). Set when the owning session is placed. */
  uint64_t last_touched;      /* monotonic tick of the latest access     */
  uint64_t tick;              /* scheduler tick counter                  */

  /* Reserved hooks (Phase 3+). */
  uint32_t quant_enabled;     /* cache quantization flag (0 = off)       */
  uint32_t spill_enabled;     /* RAM/storage spill flag (0 = off)        */

  uint64_t stats_blocks;      /* blocks allocated                        */
  uint64_t stats_prefix_hits; /* prefix cache hits                       */
  uint64_t stats_prefix_miss; /* prefix cache misses                     */
} pai_kv_cache_t;

/*
 * Initialize over caller-owned memory: `memory` must be aligned to
 * `region_size` (a power of two >= PAI_BUDDY_MIN_SIZE). The cache owns
 * no allocation in this mode.
 */
pai_status_t pai_kv_cache_init(pai_kv_cache_t *cache, void *memory,
                               uint64_t region_size, uint32_t num_layers,
                               uint32_t kv_bytes_per_token);

/*
 * Initialize with an internally allocated region (destroy with
 * pai_kv_cache_destroy).
 */
pai_status_t pai_kv_cache_init_owned(pai_kv_cache_t *cache,
                                     uint64_t region_size, uint32_t num_layers,
                                     uint32_t kv_bytes_per_token);

void pai_kv_cache_destroy(pai_kv_cache_t *cache);

/* Total byte capacity of the cache region. */
uint64_t pai_kv_cache_capacity_bytes(const pai_kv_cache_t *cache);

/*
 * Reserve `tokens` new positions, returning the starting position in
 * *out_pos (positions are appended monotonically). PAI_ERR_NOMEM when
 * the region is exhausted.
 */
pai_status_t pai_kv_cache_reserve(pai_kv_cache_t *cache, uint32_t tokens,
                                  uint32_t *out_pos);

/*
 * Access a cached position: bumps the touch tick and, when `prefix_hash`
 * matches the rolling hash recorded at `pos`, counts a prefix hit
 * (returns 1); otherwise counts a miss and records the hash (0).
 */
int pai_kv_cache_touch(pai_kv_cache_t *cache, uint32_t pos,
                       uint64_t prefix_hash);

/* Rolling FNV-1a hash over a token id stream (prefix caching seed). */
uint64_t pai_kv_prefix_hash(uint64_t hash, uint32_t token_id);

/* Reset the cache to empty (keeps the region). */
void pai_kv_cache_reset(pai_kv_cache_t *cache);

#ifdef __cplusplus
}
#endif

#endif /* PAI_CACHE_KV_CACHE_H */
