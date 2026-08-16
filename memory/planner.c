#include "planner.h"
#include "mem.h"

#include <stdlib.h>
#include <string.h>

/* Free range in the region, kept in a sorted linked list. */
typedef struct free_node {
  uint64_t off;
  uint64_t size;
  struct free_node *next;
} free_node_t;

/* Sort entry: an allocation plus its original add-order index. */
typedef struct sort_entry {
  pai_mem_alloc_t alloc;
  uint32_t add_idx;
} sort_entry_t;

/* Live allocation during the sweep. */
typedef struct live_alloc {
  uint64_t end;
  uint64_t off;
  uint64_t size;
} live_alloc_t;

static int
alloc_cmp(const void *a, const void *b) {
  const sort_entry_t *pa = (const sort_entry_t *)a;
  const sort_entry_t *pb = (const sort_entry_t *)b;
  const pai_mem_alloc_t *ea = &pa->alloc;
  const pai_mem_alloc_t *eb = &pb->alloc;

  /* Pinned allocs sort last: they extend the stable region tail. */
  if (ea->pinned != eb->pinned) {
    return ea->pinned < eb->pinned ? -1 : 1;
  }
  if (ea->start != eb->start) {
    return ea->start < eb->start ? -1 : 1;
  }
  if (ea->end != eb->end) {
    /* Longer-lived first among equal starts: better reuse. */
    return ea->end > eb->end ? -1 : 1;
  }
  return 0;
}

void
pai_mem_planner_init(pai_mem_planner_t *planner) {
  memset(planner, 0, sizeof(*planner));
}

pai_status_t
pai_mem_planner_set_arena_align(pai_mem_planner_t *planner, uint64_t align) {
  if (!planner || align == 0 || !pai_mem_is_pow2(align)) {
    return PAI_ERR_INVALID_ARG;
  }
  planner->arena_align = align;
  return PAI_OK;
}

static pai_status_t
planner_add_common(pai_mem_planner_t *planner, uint64_t size, uint64_t align,
                   uint64_t start, uint64_t end, uint32_t pinned) {
  pai_mem_alloc_t *slot;

  if (!planner || size == 0 || align == 0 || !pai_mem_is_pow2(align) ||
      end <= start || size > UINT64_MAX - (align - 1)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (planner->count >= PAI_MEM_PLANNER_MAX_ALLOCS) {
    return PAI_ERR_NOMEM;
  }

  slot = &planner->allocs[planner->count++];
  slot->size = size;
  slot->align = align;
  slot->start = start;
  slot->end = end;
  slot->pinned = pinned;
  return PAI_OK;
}

pai_status_t
pai_mem_planner_add(pai_mem_planner_t *planner, uint64_t size, uint64_t align,
                    uint64_t start, uint64_t end) {
  return planner_add_common(planner, size, align, start, end, 0);
}

pai_status_t
pai_mem_planner_add_pinned(pai_mem_planner_t *planner, uint64_t size,
                           uint64_t align, uint64_t start, uint64_t end) {
  return planner_add_common(planner, size, align, start, end, 1);
}

/* Insert [off, off+size) into the sorted free list, merging neighbours. */
static void
free_list_insert(free_node_t **head, free_node_t *pool, uint32_t *pool_used,
                 uint64_t off, uint64_t size) {
  free_node_t *prev = NULL;
  free_node_t *cur = *head;

  while (cur != NULL && cur->off < off) {
    prev = cur;
    cur = cur->next;
  }

  /* Merge with predecessor. */
  if (prev != NULL && prev->off + prev->size == off) {
    prev->size += size;
    /* Merge with successor if the combined range touches it. */
    if (cur != NULL && off + size == cur->off) {
      prev->size += cur->size;
      prev->next = cur->next;
    }
    return;
  }

  /* Merge with successor only. */
  if (cur != NULL && off + size == cur->off) {
    cur->off = off;
    cur->size += size;
    return;
  }

  /* Fresh node. */
  if (*pool_used >= PAI_MEM_PLANNER_MAX_ALLOCS * 2 + 4) {
    /* Pool exhaustion only loses a reuse opportunity, never correctness. */
    return;
  }
  {
    free_node_t *node = &pool[(*pool_used)++];
    node->off = off;
    node->size = size;
    node->next = cur;
    if (prev != NULL) {
      prev->next = node;
    } else {
      *head = node;
    }
  }
}

pai_status_t
pai_mem_plan_build(const pai_mem_planner_t *planner, pai_mem_plan_t *plan) {
  sort_entry_t *sorted;
  free_node_t *pool;
  free_node_t *free_head = NULL;
  live_alloc_t *live;
  uint32_t pool_used = 0;
  uint32_t live_count = 0;
  uint64_t cursor = 0;
  uint64_t max_end = 0;
  uint64_t peak_live = 0;
  uint64_t naive = 0;
  uint32_t reuses = 0;

  pai_status_t st = PAI_OK;

  if (!planner || !plan) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(plan, 0, sizeof(*plan));
  plan->count = planner->count;
  if (planner->count == 0) {
    return PAI_OK;
  }

  sorted = (sort_entry_t *)malloc(sizeof(sort_entry_t) * planner->count);
  pool = (free_node_t *)malloc(sizeof(free_node_t) * (PAI_MEM_PLANNER_MAX_ALLOCS * 2 + 4));
  live = (live_alloc_t *)malloc(sizeof(live_alloc_t) * planner->count);
  plan->offsets = (uint64_t *)malloc(sizeof(uint64_t) * planner->count);

  if (sorted == NULL || pool == NULL || live == NULL || plan->offsets == NULL) {
    free(sorted);
    free(pool);
    free(live);
    free(plan->offsets);
    plan->offsets = NULL;
    plan->count = 0;
    return PAI_ERR_NOMEM;
  }

  for (uint32_t i = 0; i < planner->count; i++) {
    sorted[i].alloc = planner->allocs[i];
    sorted[i].add_idx = i;
  }
  qsort(sorted, planner->count, sizeof(sort_entry_t), alloc_cmp);

  for (uint32_t i = 0; i < planner->count; i++) {
    const pai_mem_alloc_t *a = &sorted[i].alloc;
    uint32_t add_idx = sorted[i].add_idx;
    uint64_t aligned_size = pai_mem_align_up(a->size, a->align);
    uint64_t off;
    free_node_t *best = NULL;
    free_node_t *best_prev = NULL;
    free_node_t *cur;
    free_node_t *prev = NULL;

    naive += aligned_size;

    if (a->pinned) {
      /* Pinned allocs never reuse space: extend the region tail. */
      plan->pinned_allocs++;
      plan->pinned_bytes += a->size;
      off = pai_mem_align_up(cursor, a->align);
      if (off > UINT64_MAX - a->size) {
        st = PAI_ERR_NOMEM; /* region overflow */
        goto cleanup;
      }
      cursor = off + a->size;
      plan->offsets[add_idx] = off;
      max_end = max_end > off + a->size ? max_end : off + a->size;
      live[live_count].end = a->end;
      live[live_count].off = off;
      live[live_count].size = a->size;
      live_count++;
      continue;
    }

    /* Release allocations whose lifetime ended before this step. */
    for (uint32_t j = 0; j < live_count;) {
      if (live[j].end <= a->start) {
        free_list_insert(&free_head, pool, &pool_used, live[j].off,
                         live[j].size);
        live[j] = live[--live_count];
      } else {
        j++;
      }
    }

    /* Best-fit over the free list, honouring alignment. */
    cur = free_head;
    prev = NULL;
    while (cur != NULL) {
      uint64_t aligned = pai_mem_align_up(cur->off, a->align);
      if (cur->size >= (aligned - cur->off) + a->size &&
          (best == NULL || cur->size < best->size)) {
        best = cur;
        best_prev = prev;
      }
      prev = cur;
      cur = cur->next;
    }

    if (best != NULL) {
      uint64_t aligned = pai_mem_align_up(best->off, a->align);
      uint64_t left_pad = aligned - best->off;
      uint64_t right = best->size - left_pad - a->size;

      /* Unlink best. */
      if (best_prev != NULL) {
        best_prev->next = best->next;
      } else {
        free_head = best->next;
      }

      if (left_pad > 0) {
        free_list_insert(&free_head, pool, &pool_used, best->off, left_pad);
      }
      if (right > 0) {
        free_list_insert(&free_head, pool, &pool_used, aligned + a->size, right);
      }

      off = aligned;
      reuses++;
    } else {
      /* No reusable range: extend the region. */
      off = pai_mem_align_up(cursor, a->align);
      if (off > UINT64_MAX - a->size) {
        st = PAI_ERR_NOMEM; /* region overflow */
        goto cleanup;
      }
      cursor = off + a->size;
    }

    plan->offsets[add_idx] = off;
    max_end = max_end > off + a->size ? max_end : off + a->size;

    live[live_count].end = a->end;
    live[live_count].off = off;
    live[live_count].size = a->size;
    live_count++;

    /* Recompute live bytes for peak tracking. */
    {
      uint64_t live_bytes = 0;
      for (uint32_t j = 0; j < live_count; j++) {
        live_bytes += live[j].size;
      }
      if (live_bytes > peak_live) {
        peak_live = live_bytes;
      }
    }
  }

  {
    uint64_t raw = cursor > max_end ? cursor : max_end;
    uint64_t arena = planner->arena_align ? planner->arena_align : 1;
    plan->region_bytes = pai_mem_align_up(raw, arena);
  }
  plan->naive_bytes = naive;
  plan->reuse_bytes = naive >= plan->region_bytes ? naive - plan->region_bytes : 0;
  plan->peak_live = peak_live;
  plan->reuses = reuses;

cleanup:
  free(sorted);
  free(pool);
  free(live);
  return st;
}

pai_status_t
pai_mem_plan_check_budget(const pai_mem_plan_t *plan, uint64_t budget) {
  if (!plan) {
    return PAI_ERR_INVALID_ARG;
  }
  return plan->region_bytes <= budget ? PAI_OK : PAI_ERR_NOMEM;
}

void
pai_mem_plan_free(pai_mem_plan_t *plan) {
  if (plan != NULL) {
    free(plan->offsets);
    plan->offsets = NULL;
    plan->count = 0;
  }
}
