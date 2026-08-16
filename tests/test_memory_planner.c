#include "test.h"

#include <planner.h>

#include <stdint.h>
#include <string.h>

/* Every pair with overlapping lifetimes must use disjoint storage. */
static void
check_no_overlap(const pai_mem_planner_t *p, const pai_mem_plan_t *plan) {
  for (uint32_t i = 0; i < p->count; i++) {
    CHECK_EQ_UINT(plan->offsets[i] % p->allocs[i].align, 0);
    for (uint32_t j = i + 1; j < p->count; j++) {
      int lifetime_overlap =
          p->allocs[i].start < p->allocs[j].end &&
          p->allocs[j].start < p->allocs[i].end;
      int mem_overlap =
          plan->offsets[i] < plan->offsets[j] + p->allocs[j].size &&
          plan->offsets[j] < plan->offsets[i] + p->allocs[i].size;
      CHECK(!(lifetime_overlap && mem_overlap));
    }
  }
}

TEST_MAIN_BEGIN()

{
  /* Empty plan. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.region_bytes, 0);
  CHECK_EQ_UINT(plan.count, 0);
  pai_mem_plan_free(&plan);
}

{
  /* Sequential non-overlapping lifetimes fully reuse one buffer. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 100, 16, 0, 2), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 100, 16, 2, 4), PAI_OK);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);

  CHECK_EQ_UINT(plan.region_bytes, 100);
  CHECK_EQ_UINT(plan.naive_bytes, 224);
  CHECK_EQ_UINT(plan.reuse_bytes, 124);
  CHECK_EQ_UINT(plan.reuses, 1);
  CHECK_EQ_UINT(plan.offsets[0], plan.offsets[1]);
  pai_mem_plan_free(&plan);
}

{
  /* Overlapping lifetimes must get distinct storage. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 100, 16, 0, 3), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 100, 16, 1, 4), PAI_OK);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);

  CHECK_EQ_UINT(plan.region_bytes, 212); /* [0,100) and [112,212) */
  CHECK_EQ_UINT(plan.reuse_bytes, 12);
  CHECK(plan.offsets[0] != plan.offsets[1]);
  pai_mem_plan_free(&plan);
}

{
  /* Alignment is honoured even when reusing a freed range. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 8, 64, 0, 1), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 8, 64, 1, 2), PAI_OK);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);

  CHECK_EQ_UINT(plan.offsets[0] % 64, 0);
  CHECK_EQ_UINT(plan.offsets[1], plan.offsets[0]); /* reused, still aligned */
  pai_mem_plan_free(&plan);
}

{
  /* Identical lifetimes fully overlap: no reuse is possible. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 64, 16, 0, 5), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 64, 16, 0, 5), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 64, 16, 0, 5), PAI_OK);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);

  CHECK_EQ_UINT(plan.region_bytes, 192);
  CHECK_EQ_UINT(plan.reuses, 0);
  CHECK(plan.offsets[0] != plan.offsets[1]);
  CHECK(plan.offsets[1] != plan.offsets[2]);
  pai_mem_plan_free(&plan);
}

{
  /* Best-fit picks the smallest free range that fits. */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 200, 16, 0, 2), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 200, 16, 1, 3), PAI_OK);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 64, 16, 2, 4), PAI_OK);
  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);

  CHECK_EQ_UINT(plan.offsets[2], 0); /* reuses the freed 200B range */
  CHECK_EQ_UINT(plan.region_bytes, 408);
  CHECK_EQ_UINT(plan.naive_bytes, 480);
  CHECK_EQ_UINT(plan.reuse_bytes, 72);
  pai_mem_plan_free(&plan);
}

{
  /* Randomized lifetimes: invariants hold (fixed seed). */
  static pai_mem_planner_t p;
  pai_mem_plan_t plan;
  const uint32_t aligns[] = {1, 8, 16, 32, 64, 128};
  uint64_t naive = 0;
  uint32_t seed = 0x5EED1234u;

  pai_mem_planner_init(&p);
  for (uint32_t i = 0; i < 300; i++) {
    uint64_t size = 1 + (seed = seed * 1103515245u + 12345u) % 4096;
    uint64_t align = aligns[(seed = seed * 1103515245u + 12345u) % 6];
    uint64_t start = (seed = seed * 1103515245u + 12345u) % 64;
    uint64_t len = 1 + (seed = seed * 1103515245u + 12345u) % 64;
    naive += (size + align - 1) & ~(align - 1);
    CHECK_EQ_INT(pai_mem_planner_add(&p, size, align, start, start + len),
                 PAI_OK);
  }

  CHECK_EQ_INT(pai_mem_plan_build(&p, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.naive_bytes, naive);
  CHECK(plan.region_bytes <= plan.naive_bytes);
  CHECK(plan.region_bytes >= plan.peak_live);
  CHECK_EQ_UINT(plan.reuse_bytes, plan.naive_bytes - plan.region_bytes);
  check_no_overlap(&p, &plan);
  pai_mem_plan_free(&plan);
}

{
  /* Table capacity. */
  static pai_mem_planner_t p;

  pai_mem_planner_init(&p);
  for (uint32_t i = 0; i < PAI_MEM_PLANNER_MAX_ALLOCS; i++) {
    CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 16, 0, 1), PAI_OK);
  }
  CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 16, 0, 1), PAI_ERR_NOMEM);
}

{
  /* Invalid arguments. */
  static pai_mem_planner_t p;

  pai_mem_planner_init(&p);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 0, 16, 0, 1), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 0, 0, 1), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 3, 0, 1), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 16, 1, 1), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_mem_planner_add(&p, 16, 16, 2, 1), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_mem_planner_add(&p, UINT64_MAX, 16, 0, 1),
               PAI_ERR_INVALID_ARG); /* overflow */
  CHECK_EQ_INT(pai_mem_plan_build(NULL, NULL), PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
