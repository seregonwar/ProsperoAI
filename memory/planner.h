/*
 * ProsperoAI — static memory planner (whitepaper §16). Compiled graphs
 * make tensor lifetimes known ahead of execution, so buffers are reused
 * when lifetimes do not overlap: step-based allocation requests become
 * a single contiguous region layout with reuse, mirroring what the GPU
 * allocator later carves out of device memory.
 */

#ifndef PAI_MEMORY_PLANNER_H
#define PAI_MEMORY_PLANNER_H

#include <pai/error.h>

#include <stdint.h>

/* Capacity is generous relative to real graphs (PAI_GRAPH_MAX_VALUES =
 * 256) while keeping the planner struct small enough for payload stacks. */
#define PAI_MEM_PLANNER_MAX_ALLOCS 1024u
#define PAI_MEM_PLANNER_DEFAULT_ALIGN 16u

typedef struct pai_mem_alloc {
  uint64_t size;  /* payload bytes, > 0                    */
  uint64_t align; /* power of two, > 0                     */
  uint64_t start; /* first live step, inclusive            */
  uint64_t end;   /* first dead step, exclusive; end > start */
  uint32_t pinned; /* weights/constants: never reused       */
} pai_mem_alloc_t;

typedef struct pai_mem_planner {
  pai_mem_alloc_t allocs[PAI_MEM_PLANNER_MAX_ALLOCS];
  uint32_t count;
  uint64_t arena_align; /* region_bytes rounding; 0 = default (16) */
} pai_mem_planner_t;

typedef struct pai_mem_plan {
  uint64_t *offsets;    /* final offset per alloc, in add order */
  uint32_t count;       /* allocs covered by the plan            */
  uint64_t region_bytes; /* total region size, alignment rounded  */
  uint64_t naive_bytes;  /* aligned sizes summed, no reuse        */
  uint64_t reuse_bytes;  /* naive_bytes - region_bytes            */
  uint64_t peak_live;    /* max concurrently-live bytes at a step */
  uint32_t reuses;       /* allocs placed entirely in reused space */
  uint32_t pinned_allocs; /* pinned allocs in the plan            */
  uint64_t pinned_bytes;  /* summed pinned payload bytes          */
} pai_mem_plan_t;

void pai_mem_planner_init(pai_mem_planner_t *planner);

/* Set the region rounding: every plan region_bytes is a multiple of
 * `align` (power of two, > 0). Default 1 (no rounding); GPU plans
 * set e.g. 4096 for the device arena. */
pai_status_t pai_mem_planner_set_arena_align(pai_mem_planner_t *planner,
                                             uint64_t align);

/*
 * Register an allocation: live for start <= step < end. Reusable.
 */
pai_status_t pai_mem_planner_add(pai_mem_planner_t *planner, uint64_t size,
                                 uint64_t align, uint64_t start, uint64_t end);

/*
 * Register a pinned allocation (weights/constants): never shares
 * space, always placed in the stable tail of the region.
 */
pai_status_t pai_mem_planner_add_pinned(pai_mem_planner_t *planner,
                                        uint64_t size, uint64_t align,
                                        uint64_t start, uint64_t end);

/*
 * Build the layout. Greedy sweep by start step; expired allocations are
 * freed into a best-fit free list so later allocations reuse their space.
 * The caller owns `offsets` until pai_mem_plan_free. Allocations with
 * identical (start, end) lifetimes are ordered arbitrarily; callers must
 * not depend on which of a tie gets which offset.
 */
pai_status_t pai_mem_plan_build(const pai_mem_planner_t *planner,
                                pai_mem_plan_t *plan);

/* PAI_OK when the plan fits the budget (region_bytes <= budget),
 * PAI_ERR_NOMEM otherwise. */
pai_status_t pai_mem_plan_check_budget(const pai_mem_plan_t *plan,
                                       uint64_t budget);

void pai_mem_plan_free(pai_mem_plan_t *plan);

#endif /* PAI_MEMORY_PLANNER_H */
