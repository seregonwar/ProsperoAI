#include "test.h"

#include <scheduler/scheduler.h>

#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * Synthetic network (Phase 1 target graph), executed end-to-end:
 *
 *   x[4] -> GEMM(w1[4,4]) -> h1 -> ADD(b[4]) -> h2 -> RELU -> h3
 *        -> GEMM(w2[4,4]=I) -> h4 -> SOFTMAX -> y            (output)
 *
 * plus executor-coverage ops on small buffers:
 *   GEMV(w3[4,4], x) -> g1          (diagonal weights)
 *   LAYERNORM(h3) -> ln
 *   RMSNORM(h4)  -> rn
 *   MUL(h3, h3)  -> m1
 *   CONCAT(y, y) -> y2 -> COPY -> y3 -> RESHAPE[2,4] -> y4
 *   CONVERT(h3)  -> cv
 *
 * Value ids (insertion order): x=1 w1=2 h1=3 b=4 h2=5 h3=6 w2=7 h4=8
 * y=9 w3=10 g1=11 ln=12 rn=13 m1=14 y2=15 y3=16 y4=17 cv=18.
 */

enum {
  V_X = 1, V_W1, V_H1, V_B, V_H2, V_H3, V_W2, V_H4, V_Y,
  V_W3, V_G1, V_LN, V_RN, V_M1, V_Y2, V_Y3, V_Y4, V_CV, V_COUNT = V_CV
};

static pai_graph_t g_net; /* shared by tests that need the full network */

static void
build_net(pai_graph_t *graph) {
  const uint64_t v4[1] = {4};
  const uint64_t m4[2] = {4, 4};
  const uint64_t v8[1] = {8};
  const uint64_t m28[2] = {2, 4};
  pai_graph_value_id x, w1, h1, b, h2, h3, w2, h4, y, w3, g1, ln, rn, m1,
      y2, y3, y4, cv;

  pai_graph_init(graph);
  x = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  w1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m4, 0);
  h1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  b = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  h2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  h3 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  w2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m4, 0);
  h4 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  y = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  w3 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m4, 0);
  g1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  ln = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  rn = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  m1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);
  y2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  y3 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  y4 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m28, 0);
  cv = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v4, 0);

  {
    pai_graph_value_id i1[] = {x, w1}, o1[] = {h1};
    pai_graph_value_id i2[] = {h1, b}, o2[] = {h2};
    pai_graph_value_id i3[] = {h2}, o3[] = {h3};
    pai_graph_value_id i4[] = {h3, w2}, o4[] = {h4};
    pai_graph_value_id i5[] = {h4}, o5[] = {y};
    pai_graph_value_id i6[] = {w3, x}, o6[] = {g1};
    pai_graph_value_id i7[] = {h3}, o7[] = {ln};
    pai_graph_value_id i8[] = {h4}, o8[] = {rn};
    pai_graph_value_id i9[] = {h3, h3}, o9[] = {m1};
    pai_graph_value_id i10[] = {y, y}, o10[] = {y2};
    pai_graph_value_id i11[] = {y2}, o11[] = {y3};
    pai_graph_value_id i12[] = {y3}, o12[] = {y4};
    pai_graph_value_id i13[] = {h3}, o13[] = {cv};
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, i1, o1) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_ADD, 2, 1, i2, o2) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_RELU, 1, 1, i3, o3) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, i4, o4) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_SOFTMAX, 1, 1, i5, o5) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMV, 2, 1, i6, o6) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_LAYERNORM, 1, 1, i7, o7) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_RMSNORM, 1, 1, i8, o8) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_MUL, 2, 1, i9, o9) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_CONCAT, 2, 1, i10, o10) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_COPY, 1, 1, i11, o11) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_RESHAPE, 1, 1, i12, o12) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_CONVERT, 1, 1, i13, o13) != 0);
  }

  pai_graph_set_input(graph, x);
  pai_graph_set_output(graph, y);
  pai_graph_set_output(graph, g1);
  pai_graph_set_output(graph, ln);
  pai_graph_set_output(graph, rn);
  pai_graph_set_output(graph, m1);
  pai_graph_set_output(graph, y2);
  pai_graph_set_output(graph, y3);
  pai_graph_set_output(graph, y4);
  pai_graph_set_output(graph, cv);
}

static void
fill_net(pai_graph_mem_plan_t *mp, uint8_t *region) {
  /* x = {1,2,3,4} */
  {
    const float x[4] = {1, 2, 3, 4};
    memcpy(region + pai_graph_mem_plan_offset(mp, V_X), x, sizeof(x));
  }
  /* w1: row i all (i+1) -> h1 = 10 * (i+1) */
  {
    float w1[16];
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++) w1[i * 4 + j] = (float)(i + 1);
    memcpy(region + pai_graph_mem_plan_offset(mp, V_W1), w1, sizeof(w1));
  }
  /* b = {0.5, -0.5, 1.5, -1.5} */
  {
    const float b[4] = {0.5f, -0.5f, 1.5f, -1.5f};
    memcpy(region + pai_graph_mem_plan_offset(mp, V_B), b, sizeof(b));
  }
  /* w2: identity */
  {
    float w2[16];
    memset(w2, 0, sizeof(w2));
    for (int i = 0; i < 4; i++) w2[i * 4 + i] = 1.0f;
    memcpy(region + pai_graph_mem_plan_offset(mp, V_W2), w2, sizeof(w2));
  }
  /* w3: diagonal {1,2,3,4} */
  {
    float w3[16];
    memset(w3, 0, sizeof(w3));
    for (int i = 0; i < 4; i++) w3[i * 4 + i] = (float)(i + 1);
    memcpy(region + pai_graph_mem_plan_offset(mp, V_W3), w3, sizeof(w3));
  }
}

static float
getv(const pai_graph_mem_plan_t *mp, const uint8_t *region, pai_graph_value_id v,
     uint32_t i) {
  float f;
  memcpy(&f, region + pai_graph_mem_plan_offset(mp, v) + 4u * i, sizeof(f));
  return f;
}

/* Expected outputs, recomputed from the fill data with doubles. */
static void
compute_expected(double *out_h2, double *out_y) {
  const double x[4] = {1, 2, 3, 4};
  const double b[4] = {0.5, -0.5, 1.5, -1.5};
  double h1[4];
  double h3[4];
  double mx = 0;
  double sum = 0;
  int i;
  int j;

  /* w1: row l is all (l+1) -> h1[j] = sum_l x[l]*(l+1) */
  for (j = 0; j < 4; j++) {
    h1[j] = 0;
    for (int l = 0; l < 4; l++) {
      h1[j] += x[l] * (double)(l + 1);
    }
    out_h2[j] = h1[j] + b[j];
    h3[j] = out_h2[j] > 0 ? out_h2[j] : 0.0; /* relu */
  }
  /* w2 = identity, so the softmax input is h3 */
  for (i = 0; i < 4; i++) {
    if (h3[i] > mx) {
      mx = h3[i];
    }
  }
  for (i = 0; i < 4; i++) {
    sum += exp(h3[i] - mx);
  }
  for (i = 0; i < 4; i++) {
    out_y[i] = exp(h3[i] - mx) / sum;
  }
}

TEST_MAIN_BEGIN()

build_net(&g_net);

{
  /* plan build: order, epochs, placement, counts */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  uint32_t seen[PAI_GRAPH_MAX_OPS + 1];
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_OK);
  CHECK_EQ_UINT(plan.num_steps, 13);
  CHECK_EQ_UINT(plan.gpu_steps, 4); /* gemm, gemm, gemv, softmax */
  CHECK_EQ_UINT(plan.cpu_steps, 9);
  memset(seen, 0, sizeof(seen));
  for (uint32_t i = 0; i < plan.num_steps; i++) {
    uint8_t expect_gpu;
    pai_graph_op_id op = plan.steps[i].op_id;
    CHECK_EQ_UINT(plan.steps[i].epoch, i);
    /* dense topo order: every op id exactly once */
    CHECK(op >= 1 && op <= g_net.num_ops);
    CHECK_EQ_UINT(seen[op], 0);
    seen[op] = 1;
    CHECK(plan.steps[i].live_bytes > 0);
    CHECK(plan.steps[i].live_bytes <= plan.peak_live);
    /* default placement: heavy compute -> GPU */
    switch (g_net.ops[op].kind) {
      case PAI_OP_GEMM:
      case PAI_OP_GEMV:
      case PAI_OP_MATMUL:
      case PAI_OP_ATTENTION:
      case PAI_OP_SOFTMAX:
        expect_gpu = 1;
        break;
      default:
        expect_gpu = 0;
    }
    CHECK_EQ_UINT(plan.steps[i].device,
                  expect_gpu ? PAI_SCHED_DEV_GPU : PAI_SCHED_DEV_CPU);
  }
  /* every op was placed exactly once */
  for (uint32_t o = 1; o <= g_net.num_ops; o++) {
    CHECK_EQ_UINT(seen[o], 1);
  }
  /* stats echo the memory plan */
  CHECK_EQ_UINT(plan.region_bytes, mp.region_bytes);
  CHECK_EQ_UINT(plan.naive_bytes, mp.naive_bytes);
  CHECK_EQ_UINT(plan.peak_live, mp.peak_live);

  /* every value is produced exactly once across the plan */
  {
    uint64_t produced_total = 0;
    uint64_t released_total = 0;
    uint64_t naive = 0;
    for (uint32_t i = 0; i < plan.num_steps; i++) {
      produced_total += plan.steps[i].produced_bytes;
      released_total += plan.steps[i].released_bytes;
    }
    for (pai_graph_value_id v = 1; v <= g_net.num_values; v++) {
      naive += g_net.values[v].size_bytes;
    }
    CHECK_EQ_UINT(produced_total, naive);
    CHECK(released_total > 0 && released_total < naive);
  }
}

{
  /* EXCLUSIVE pushes compute ops to the GPU */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_EXCLUSIVE, NULL, &plan) ==
        PAI_OK);
  CHECK_EQ_UINT(plan.gpu_steps, 9); /* compute kinds -> GPU */
  CHECK_EQ_UINT(plan.cpu_steps, 4); /* concat, copy, reshape, convert */
  for (uint32_t i = 0; i < plan.num_steps; i++) {
    pai_graph_op_id op = plan.steps[i].op_id;
    switch (g_net.ops[op].kind) {
      case PAI_OP_ADD:
      case PAI_OP_MUL:
      case PAI_OP_RELU:
      case PAI_OP_LAYERNORM:
      case PAI_OP_RMSNORM:
        CHECK_EQ_UINT(plan.steps[i].device, PAI_SCHED_DEV_GPU);
        break;
      case PAI_OP_CONCAT:
      case PAI_OP_COPY:
      case PAI_OP_RESHAPE:
      case PAI_OP_CONVERT:
        CHECK_EQ_UINT(plan.steps[i].device, PAI_SCHED_DEV_CPU);
        break;
      default:
        break;
    }
  }
}

{
  /* explicit hints override the policy */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  uint8_t hints[PAI_GRAPH_MAX_OPS + 1];
  memset(hints, 0, sizeof(hints));
  hints[2] = PAI_SCHED_DEV_GPU;  /* add -> GPU */
  hints[10] = PAI_SCHED_DEV_CPU; /* concat -> CPU (already) */
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_INTERACTIVE, hints, &plan) ==
        PAI_OK);
  for (uint32_t i = 0; i < plan.num_steps; i++) {
    if (plan.steps[i].op_id == 2) { /* add */
      CHECK_EQ_UINT(plan.steps[i].device, PAI_SCHED_DEV_GPU);
    }
  }
  CHECK_EQ_UINT(plan.gpu_steps, 5);
  /* invalid hint rejected */
  hints[2] = 99;
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_INTERACTIVE, hints, &plan) ==
        PAI_ERR_INVALID_ARG);
}

{
  /* memory modes (§17) and residency checks */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_BALANCED, NULL, &plan) ==
        PAI_OK);
  CHECK(pai_sched_plan_memory_mode(&plan, plan.region_bytes) ==
        PAI_SCHED_MEM_PERFORMANCE);
  /* Balanced requires the region to be strictly larger than the peak
   * working set (otherwise peak-live == full residency). */
  if (plan.region_bytes > plan.peak_live) {
    CHECK(pai_sched_plan_memory_mode(&plan, plan.peak_live) ==
          PAI_SCHED_MEM_BALANCED);
  }
  CHECK(pai_sched_plan_memory_mode(&plan, plan.peak_live - 1) ==
        PAI_SCHED_MEM_CAPACITY);
  CHECK(pai_sched_plan_memory_mode(&plan, plan.region_bytes + 1024) ==
        PAI_SCHED_MEM_PERFORMANCE);
  CHECK(pai_sched_plan_check(&plan, plan.peak_live) == PAI_OK);
  CHECK(pai_sched_plan_check(&plan, plan.peak_live - 1) == PAI_ERR_NOMEM);
  CHECK(pai_sched_plan_check(NULL, 1) == PAI_ERR_INVALID_ARG);
}

{
  /* END-TO-END: execute the synthetic network and verify outputs */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  pai_sched_run_stats_t stats;
  uint8_t *region;

  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_OK);

  region = (uint8_t *)calloc(1, (size_t)mp.region_bytes);
  CHECK(region != NULL);
  if (region != NULL) {
    fill_net(&mp, region);
    CHECK(pai_sched_execute(&g_net, &mp, &plan, region, &stats) == PAI_OK);
    CHECK_EQ_UINT(stats.steps_executed, 13);
    CHECK_EQ_UINT(stats.gpu_steps, 4);
    CHECK_EQ_UINT(stats.cpu_steps, 9);

    {
      double h2_exp[4];
      double y_exp[4];
      compute_expected(h2_exp, y_exp);
      /* h2 = h1 + b (h3/h1 die mid-run; only values live until the end
       * are readable after pai_sched_execute returns). */
      for (int i = 0; i < 4; i++) {
        CHECK(fabsf(getv(&mp, region, V_H2, (uint32_t)i) -
                    (float)h2_exp[i]) < 1e-4f);
      }
      /* y = softmax(h4) where h4 == h3 (w2 = identity). y being the
       * non-uniform softmax also proves relu actually executed (a
       * no-op relu would leave h3 zeroed and y uniform). */
      for (int i = 0; i < 4; i++) {
        CHECK(fabsf(getv(&mp, region, V_Y, (uint32_t)i) -
                    (float)y_exp[i]) < 1e-4f);
      }
    }
    /* g1 = diag(1..4) * x = {1,4,9,16} */
    {
      const float g1_exp[4] = {1, 4, 9, 16};
      for (int i = 0; i < 4; i++) {
        CHECK(fabsf(getv(&mp, region, V_G1, (uint32_t)i) - g1_exp[i]) < 1e-5f);
      }
    }
    /* m1 = MUL(h3,h3) = h2^2 and cv = CONVERT(h3) = h2 are outputs
     * (live to the end), so they are readable after execution. */
    {
      double h2_exp[4];
      double y_exp[4];
      compute_expected(h2_exp, y_exp);
      for (int i = 0; i < 4; i++) {
        float h2 = (float)h2_exp[i];
        CHECK(fabsf(getv(&mp, region, V_M1, (uint32_t)i) - h2 * h2) < 1e-4f);
        CHECK(fabsf(getv(&mp, region, V_CV, (uint32_t)i) - h2) < 1e-4f);
      }
    }
    /* y2 = concat(y,y); y3 = copy; y4 = reshape (same data) */
    for (int i = 0; i < 8; i++) {
      float yv = getv(&mp, region, V_Y, (uint32_t)(i % 4));
      CHECK(fabsf(getv(&mp, region, V_Y2, (uint32_t)i) - yv) < 1e-6f);
      CHECK(fabsf(getv(&mp, region, V_Y3, (uint32_t)i) - yv) < 1e-6f);
      CHECK(fabsf(getv(&mp, region, V_Y4, (uint32_t)i) - yv) < 1e-6f);
    }
    /* rmsnorm invariant: sum of squares of output == n (unit RMS) */
    {
      float acc = 0;
      for (int i = 0; i < 4; i++) {
        float r = getv(&mp, region, V_RN, (uint32_t)i);
        acc += r * r;
      }
      CHECK(fabsf(acc - 4.0f) < 1e-3f);
    }
    /* layernorm invariant: mean ~0, var ~1 */
    {
      double mean = 0;
      for (int i = 0; i < 4; i++) {
        mean += getv(&mp, region, V_LN, (uint32_t)i);
      }
      mean /= 4.0;
      CHECK(fabsf((float)mean) < 1e-4f);
    }

    free(region);
  }
}

{
  /* determinism: two runs on fresh regions match exactly */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  uint8_t *r1;
  uint8_t *r2;
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(&g_net, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_OK);
  r1 = (uint8_t *)calloc(1, (size_t)mp.region_bytes);
  r2 = (uint8_t *)calloc(1, (size_t)mp.region_bytes);
  CHECK(r1 != NULL && r2 != NULL);
  if (r1 != NULL && r2 != NULL) {
    fill_net(&mp, r1);
    fill_net(&mp, r2);
    CHECK(pai_sched_execute(&g_net, &mp, &plan, r1, NULL) == PAI_OK);
    CHECK(pai_sched_execute(&g_net, &mp, &plan, r2, NULL) == PAI_OK);
    CHECK(memcmp(r1, r2, (size_t)mp.region_bytes) == 0);
  }
  free(r1);
  free(r2);
}

{
  /* unsupported ops and shape mismatches are surfaced */
  pai_graph_t graph;
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  uint8_t region[256];
  const uint64_t v4[1] = {4};
  const uint64_t m43[2] = {4, 3};
  const uint64_t m22[2] = {2, 2};
  pai_graph_value_id a, b, c;

  pai_graph_init(&graph);
  /* gemm shape mismatch: b is 2x2 but k must be 3 */
  a = pai_graph_add_value(&graph, PAI_DTYPE_F32, 2, m43, 0);
  b = pai_graph_add_value(&graph, PAI_DTYPE_F32, 2, m22, 0);
  c = pai_graph_add_value(&graph, PAI_DTYPE_F32, 2, m22, 0);
  {
    pai_graph_value_id in[] = {a, b};
    pai_graph_value_id out[] = {c};
    CHECK(pai_graph_add_op(&graph, PAI_OP_GEMM, 2, 1, in, out) != 0);
  }
  CHECK(pai_graph_memory_plan(&graph, &mp) == PAI_OK);
  CHECK(pai_sched_build(&graph, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_OK);
  CHECK(mp.region_bytes <= sizeof(region));
  CHECK(pai_sched_execute(&graph, &mp, &plan, region, NULL) ==
        PAI_ERR_MISMATCH);

  /* attention executes: 1 head, hd=1, seq=2. q=0, k=0 -> uniform
   * scores; v = {10, 20}: out[0] = 10, out[1] = mean(10,20) = 15. */
  {
    const uint64_t sq[3] = {2, 1, 1}; /* [seq, H, hd] position-major */
    pai_graph_value_id aq, ak, av, ao;
    pai_graph_mem_plan_t amp;
    pai_sched_plan_t aplan;
    uint8_t *aregion;
    const float z[2] = {0.0f, 0.0f};
    const float vv[2] = {10.0f, 20.0f};

    pai_graph_init(&graph);
    aq = pai_graph_add_value(&graph, PAI_DTYPE_F32, 3, sq, 0);
    ak = pai_graph_add_value(&graph, PAI_DTYPE_F32, 3, sq, 0);
    av = pai_graph_add_value(&graph, PAI_DTYPE_F32, 3, sq, 0);
    ao = pai_graph_add_value(&graph, PAI_DTYPE_F32, 3, sq, 0);
    {
      pai_graph_value_id in[] = {aq, ak, av};
      pai_graph_value_id out[] = {ao};
      CHECK(pai_graph_add_op(&graph, PAI_OP_ATTENTION, 3, 1, in, out) != 0);
    }
    CHECK(pai_graph_memory_plan(&graph, &amp) == PAI_OK);
    CHECK(pai_sched_build(&graph, &amp, PAI_SCHED_INTERACTIVE, NULL, &aplan) ==
          PAI_OK);
    aregion = (uint8_t *)calloc(1, (size_t)amp.region_bytes);
    CHECK(aregion != NULL);
    if (aregion != NULL) {
      memcpy(aregion + pai_graph_mem_plan_offset(&amp, aq), z, sizeof(z));
      memcpy(aregion + pai_graph_mem_plan_offset(&amp, ak), z, sizeof(z));
      memcpy(aregion + pai_graph_mem_plan_offset(&amp, av), vv, sizeof(vv));
      CHECK(pai_sched_execute(&graph, &amp, &aplan, aregion, NULL) == PAI_OK);
      CHECK(fabsf(getv(&amp, aregion, ao, 0) - 10.0f) < 1e-5f);
      CHECK(fabsf(getv(&amp, aregion, ao, 1) - 15.0f) < 1e-5f);
      free(aregion);
    }
  }

  /* non-f32 tensors are rejected by the v0 f32-only executor */
  pai_graph_init(&graph);
  {
    const uint64_t v4[1] = {4};
    pai_graph_value_id ua, ub, uc;
    ua = pai_graph_add_value(&graph, PAI_DTYPE_U8, 1, v4, 0);
    ub = pai_graph_add_value(&graph, PAI_DTYPE_U8, 1, v4, 0);
    uc = pai_graph_add_value(&graph, PAI_DTYPE_U8, 1, v4, 0);
    {
      pai_graph_value_id in[] = {ua, ub};
      pai_graph_value_id out[] = {uc};
      CHECK(pai_graph_add_op(&graph, PAI_OP_COPY, 2, 1, in, out) != 0);
    }
    CHECK(pai_graph_memory_plan(&graph, &mp) == PAI_OK);
    CHECK(pai_sched_build(&graph, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
          PAI_OK);
    CHECK(mp.region_bytes <= sizeof(region));
    CHECK(pai_sched_execute(&graph, &mp, &plan, region, NULL) ==
          PAI_ERR_UNSUPPORTED);
  }
}

{
  /* sessions: priority pick, destroy, capacity */
  pai_sched_sessions_t ss;
  uint64_t s1, s2, s3;
  int idx;
  pai_sched_sessions_init(&ss);
  CHECK(pai_sched_session_create(&ss, 1, PAI_SCHED_INTERACTIVE, &s1) == PAI_OK);
  CHECK(pai_sched_session_create(&ss, 5, PAI_SCHED_THROUGHPUT, &s2) == PAI_OK);
  CHECK(pai_sched_session_create(&ss, 3, PAI_SCHED_BALANCED, &s3) == PAI_OK);
  idx = pai_sched_pick_session(&ss);
  CHECK(idx >= 0 && ss.slots[idx].id == s2); /* highest priority */
  CHECK(pai_sched_session_active(&ss, s1) == 1);
  CHECK(pai_sched_session_destroy(&ss, s2) == PAI_OK);
  CHECK(pai_sched_session_active(&ss, s2) == 0);
  idx = pai_sched_pick_session(&ss);
  CHECK(idx >= 0 && ss.slots[idx].id == s3);
  CHECK(pai_sched_session_destroy(&ss, s1) == PAI_OK);
  CHECK(pai_sched_session_destroy(&ss, s3) == PAI_OK);
  CHECK(pai_sched_pick_session(&ss) == -1);
  CHECK(pai_sched_session_destroy(&ss, 42) == PAI_ERR_MISMATCH);

  /* full registry */
  pai_sched_sessions_init(&ss);
  {
    uint64_t id;
    int ok = 1;
    for (uint32_t i = 0; i < PAI_SCHED_MAX_SESSIONS; i++) {
      if (pai_sched_session_create(&ss, 0, PAI_SCHED_INTERACTIVE, &id) !=
          PAI_OK) {
        ok = 0;
      }
    }
    CHECK(ok == 1);
    CHECK(pai_sched_session_create(&ss, 0, PAI_SCHED_INTERACTIVE, &id) ==
          PAI_ERR_NOMEM);
  }
  CHECK(pai_sched_session_create(&ss, 0, 99, &s1) == PAI_ERR_INVALID_ARG);
}

{
  /* invalid arguments */
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  CHECK(pai_graph_memory_plan(&g_net, &mp) == PAI_OK);
  CHECK(pai_sched_build(NULL, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_sched_build(&g_net, NULL, PAI_SCHED_INTERACTIVE, NULL, &plan) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_sched_build(&g_net, &mp, (pai_sched_policy_t)99, NULL, &plan) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_sched_execute(&g_net, &mp, NULL, NULL, NULL) ==
        PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
