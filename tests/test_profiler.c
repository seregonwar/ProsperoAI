#include "test.h"

#include <profiler/profile.h>
#include <scheduler/scheduler.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* T6 plan profiler: per-step counters, profile log, replay hash and the
 * hooked plan runner. */

/* Tiny graph: x[4] -> GEMV(w[2,4]) -> h[2] -> RELU -> y[2] */
static void
build_graph(pai_graph_t *graph, pai_graph_value_id *out_vx,
            pai_graph_value_id *out_vw, pai_graph_value_id *out_vy) {
  const uint64_t v4[1] = {4};
  const uint64_t v2[1] = {2};
  const uint64_t m24[2] = {2, 4};
  pai_graph_value_id x, w, h, y;

  pai_graph_init(graph);
  x = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  w = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m24, 0);
  h = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v2, 0);
  y = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v2, 0);
  {
    pai_graph_value_id in[] = {w, x};
    pai_graph_value_id out[] = {h};
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMV, 2, 1, in, out) != 0);
  }
  {
    pai_graph_value_id in[] = {h};
    pai_graph_value_id out[] = {y};
    CHECK(pai_graph_add_op(graph, PAI_OP_RELU, 1, 1, in, out) != 0);
  }
  pai_graph_set_input(graph, x);
  pai_graph_set_output(graph, y);
  *out_vx = x;
  *out_vw = w;
  *out_vy = y;
}

static void
fill_graph(pai_graph_mem_plan_t *mp, uint8_t *region, pai_graph_value_id vx,
           pai_graph_value_id vw, pai_graph_value_id vy) {
  /* w = [[1,0,0,0],[0,1,0,0]], x = {-1, 3, 0, 0} -> h = {-1, 3} ->
   * relu -> y = {0, 3} */
  float w[8];
  const float x[4] = {-1.0f, 3.0f, 0.0f, 0.0f};
  memset(w, 0, sizeof(w));
  w[0] = 1.0f;
  w[5] = 1.0f;
  memcpy(region + pai_graph_mem_plan_offset(mp, vx), x, sizeof(x));
  memcpy(region + pai_graph_mem_plan_offset(mp, vw), w, sizeof(w));
}

TEST_MAIN_BEGIN()

{
  /* manual step accounting: totals, log, deterministic replay hash */
  pai_profile_t p1, p2;
  pai_profile_init(&p1);
  pai_profile_init(&p2);
  CHECK(pai_profile_add_step(&p1, 1, PAI_SCHED_DEV_GPU, 1000, 5000, 64) ==
        PAI_OK);
  CHECK(pai_profile_add_step(&p1, 2, PAI_SCHED_DEV_CPU, 2000, 0, 0) ==
        PAI_OK);
  CHECK(pai_profile_finish(&p1) > 0);
  CHECK_EQ_UINT(p1.num_steps, 2);
  CHECK_EQ_UINT(p1.total_cpu_ns, 3000);
  CHECK_EQ_UINT(p1.total_gpu_ns, 5000);
  CHECK_EQ_UINT(p1.total_transfer_bytes, 64);
  CHECK_EQ_UINT(p1.total_ns, 8000);
  CHECK(p1.hash != 0);
  CHECK(strstr(p1.log, "op=1 dev=gpu") != NULL);
  CHECK(strstr(p1.log, "op=2 dev=cpu") != NULL);

  /* identical structural profile -> same hash */
  CHECK(pai_profile_add_step(&p2, 1, PAI_SCHED_DEV_GPU, 777, 888, 64) ==
        PAI_OK);
  CHECK(pai_profile_add_step(&p2, 2, PAI_SCHED_DEV_CPU, 999, 0, 0) ==
        PAI_OK);
  CHECK(pai_profile_finish(&p2) > 0);
  CHECK_EQ_UINT(p1.hash, p2.hash); /* timings excluded from the hash */

  /* different transfer payload -> different hash */
  {
    pai_profile_t p3;
    CHECK(pai_profile_add_step(&p3, 1, PAI_SCHED_DEV_GPU, 1000, 5000, 65) ==
          PAI_OK);
    CHECK(pai_profile_add_step(&p3, 2, PAI_SCHED_DEV_CPU, 2000, 0, 0) ==
          PAI_OK);
    CHECK(pai_profile_finish(&p3) > 0);
    CHECK(p3.hash != p1.hash);
  }

  /* capacity and arg validation */
  {
    pai_profile_t p;
    pai_profile_init(&p);
    for (int i = 0; i < PAI_PROFILE_MAX_STEPS; i++) {
      CHECK(pai_profile_add_step(&p, (uint32_t)(i + 1), PAI_SCHED_DEV_CPU, 1,
                                 0, 0) == PAI_OK);
    }
    CHECK(pai_profile_add_step(&p, 999, PAI_SCHED_DEV_CPU, 1, 0, 0) ==
          PAI_ERR_NOMEM);
  }
  CHECK(pai_profile_add_step(NULL, 1, PAI_SCHED_DEV_CPU, 1, 0, 0) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_profile_add_step(&p1, 0, PAI_SCHED_DEV_CPU, 1, 0, 0) ==
        PAI_ERR_INVALID_ARG);
}

{
  /* full plan run through the hooked executor */
  pai_graph_t graph;
  pai_graph_value_id vx, vw, vy;
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  pai_profile_t prof;
  pai_sched_run_stats_t stats;
  uint8_t *region;
  float got;

  build_graph(&graph, &vx, &vw, &vy);
  CHECK(pai_graph_memory_plan(&graph, &mp) == PAI_OK);
  CHECK(pai_sched_build(&graph, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_OK);
  region = (uint8_t *)calloc(1, (size_t)mp.region_bytes);
  CHECK(region != NULL);
  if (region != NULL) {
    fill_graph(&mp, region, vx, vw, vy);
    CHECK(pai_profile_run_plan(&graph, &mp, &plan, region, 128, &prof,
                               &stats) == PAI_OK);
    CHECK_EQ_UINT(prof.num_steps, 2);
    CHECK_EQ_UINT(stats.steps_executed, 2);
    /* gemv is placed on the GPU by default: transfer credited, no cpu
     * wall; relu is CPU: no transfer */
    CHECK_EQ_UINT(prof.total_transfer_bytes, 128);
    CHECK(prof.total_cpu_ns > 0);
    CHECK(prof.total_ns > 0);
    CHECK(prof.hash != 0);
    CHECK(strstr(prof.log, "dev=gpu") != NULL);
    CHECK(strstr(prof.log, "dev=cpu") != NULL);
    /* execution actually happened: y = relu(w*x) = {0, 3} */
    memcpy(&got, region + pai_graph_mem_plan_offset(&mp, vy),
           sizeof(got));
    CHECK(fabsf(got - 0.0f) < 1e-6f);
    memcpy(&got, region + pai_graph_mem_plan_offset(&mp, vy) + 4,
           sizeof(got));
    CHECK(fabsf(got - 3.0f) < 1e-6f);
    free(region);
  }

  /* arg validation */
  CHECK(pai_profile_run_plan(&graph, &mp, &plan, NULL, 0, &prof, NULL) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_profile_run_plan(NULL, &mp, &plan, region, 0, &prof, NULL) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_profile_run_plan(&graph, NULL, &plan, region, 0, &prof, NULL) ==
        PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
