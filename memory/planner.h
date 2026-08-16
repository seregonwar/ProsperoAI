/*
 * ProsperoAI — memory subsystem
 *
 * Static memory planner (whitepaper §16). Compiled graphs let tensor
 * lifetimes be known ahead of execution, so buffers can be reused when
 * lifetimes do not overlap. The planner consumes a set of allocation
 * requests with step-based lifetimes and produces a single contiguous
 * region layout with reuse, mirroring what the GPU allocator later
 * carves out of device memory.
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
} pai_mem_alloc_t;

typedef struct pai_mem_planner {
  pai_mem_alloc_t allocs[PAI_MEM_PLANNER_MAX_ALLOCS];
  uint32_t count;
} pai_mem_planner_t;

typedef struct pai_mem_plan {
  uint64_t *offsets;    /* final offset per alloc, in add order */
  uint32_t count;       /* allocs covered by the plan            */
  uint64_t region_bytes; /* total region size, alignment rounded  */
  uint64_t naive_bytes;  /* aligned sizes summed, no reuse        */
  uint64_t reuse_bytes;  /* naive_bytes - region_bytes            */
  uint64_t peak_live;    /* max concurrently-live bytes at a step */
  uint32_t reuses;       /* allocs placed entirely in reused space */
} pai_mem_plan_t;

void pai_mem_planner_init(pai_mem_planner_t *planner);

/*
 * Register an allocation. start/end are execution steps; the region is
 * live for start <= step < end. Returns PAI_ERR_INVALID_ARG for bad
 * sizes/align/lifetimes and PAI_ERR_NOMEM when the table is full.
 */
pai_status_t pai_mem_planner_add(pai_mem_planner_t *planner, uint64_t size,
                                 uint64_t align, uint64_t start, uint64_t end);

/*
 * Build the layout. Greedy sweep by start step; expired allocations are
 * freed into a best-fit free list so later allocations reuse their space.
 * The caller owns `offsets` until pai_mem_plan_free. Allocations with
 * identical (start, end) lifetimes are ordered arbitrarily; callers must
 * not depend on which of a tie gets which offset.
 */
pai_status_t pai_mem_plan_build(const pai_mem_planner_t *planner,
                                pai_mem_plan_t *plan);

void pai_mem_plan_free(pai_mem_plan_t *plan);

#endif /* PAI_MEMORY_PLANNER_H */
