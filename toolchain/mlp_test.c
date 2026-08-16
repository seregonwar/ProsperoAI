/*
 * ProsperoAI — T7 exit test: MLP 8->16->8 differential harness.
 *
 * Phase-1 exit test (mission): one MLP forward pass must agree across
 * every backend — CPU reference (scheduler executor + op registry
 * eager), GPU serial kernels and the host-reference mirrors. This file
 * is Seat B's leg: graph build + CPU-ref execution + profile
 * plausibility (T6). Seat A's leg (GPU serial G48/G22 + host-ref
 * dispatch) plugs into the marked section once T4C lands.
 *
 * Exit: prints per-section PASS/FAIL and returns 0 only when the
 * CPU-ref outputs match a double-precision manual computation and the
 * plan profile is plausible.
 *
 * Build: host-tests/host-reference preset; binary `mlp_test`.
 */

#include <pai/error.h>
#include <pai/op.h>
#include <pai/tensor.h>

#include <graph/graph.h>
#include <hal/host_kernels.h>
#include <profiler/profile.h>
#include <ref_ops.h>
#include <scheduler/scheduler.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MLP_IN  8
#define MLP_HID 16
#define MLP_OUT 8

static int g_failures;

#define REQUIRE(cond)                                                        \
  do {                                                                       \
    if (!(cond)) {                                                           \
      printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);               \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

#define REQUIRE_EQ_UINT(a, b)                                                \
  do {                                                                       \
    unsigned long long va_ = (unsigned long long)(a);                        \
    unsigned long long vb_ = (unsigned long long)(b);                        \
    if (va_ != vb_) {                                                        \
      printf("  FAIL %s:%d: %s == %s (%llu != %llu)\n", __FILE__, __LINE__,  \
             #a, #b, va_, vb_);                                              \
      g_failures++;                                                          \
    }                                                                        \
  } while (0)

/* Deterministic weights (kept in exact float for the fill and in double
 * for the oracle so both reference paths are cross-checked). */
static void
fill_weights(float *w1, float *b1, float *w2, float *b2) {
  for (int i = 0; i < MLP_IN; i++) {
    for (int j = 0; j < MLP_HID; j++) {
      w1[i * MLP_HID + j] = (float)(((i * 3 + j * 5) % 7) - 3) * 0.1f;
    }
  }
  for (int j = 0; j < MLP_HID; j++) {
    b1[j] = (float)(j % 5) * 0.05f;
  }
  for (int i = 0; i < MLP_HID; i++) {
    for (int j = 0; j < MLP_OUT; j++) {
      w2[i * MLP_OUT + j] = (float)(((i * 7 + j * 2) % 9) - 4) * 0.05f;
    }
  }
  for (int j = 0; j < MLP_OUT; j++) {
    b2[j] = (float)((j * 3) % 4) * 0.1f;
  }
}

/* Double-precision oracle: y = W2 * relu(W1 * x + b1) + b2. */
static void
mlp_oracle(const float *x, const float *w1, const float *b1, const float *w2,
           const float *b2, double *y) {
  double h[MLP_HID];
  for (int j = 0; j < MLP_HID; j++) {
    double acc = 0;
    for (int i = 0; i < MLP_IN; i++) {
      acc += (double)x[i] * w1[i * MLP_HID + j];
    }
    acc += b1[j];
    h[j] = acc > 0 ? acc : 0.0; /* relu */
  }
  for (int j = 0; j < MLP_OUT; j++) {
    double acc = 0;
    for (int i = 0; i < MLP_HID; i++) {
      acc += h[i] * w2[i * MLP_OUT + j];
    }
    y[j] = acc + b2[j];
  }
}

/* Value ids in insertion order (graph.h: first added = 1). */
enum {
  V_X = 1,
  V_W1,
  V_B1,
  V_G1,
  V_A1,
  V_H,
  V_W2,
  V_B2,
  V_G2,
  V_Y
};

static void
build_mlp(pai_graph_t *graph) {
  const uint64_t v8[1] = {MLP_IN};
  const uint64_t v16[1] = {MLP_HID};
  const uint64_t m816[2] = {MLP_IN, MLP_HID};
  const uint64_t m168[2] = {MLP_HID, MLP_OUT};
  pai_graph_value_id x, w1, b1, g1, a1, h, w2, b2, g2, y;

  pai_graph_init(graph);
  x = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  w1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m816, 0);
  b1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  g1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  a1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  h = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  w2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m168, 0);
  b2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  g2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  y = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v8, 0);
  {
    pai_graph_value_id in[] = {x, w1};
    pai_graph_value_id out[] = {g1};
    REQUIRE(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, in, out) != 0);
  }
  {
    pai_graph_value_id in[] = {g1, b1};
    pai_graph_value_id out[] = {a1};
    REQUIRE(pai_graph_add_op(graph, PAI_OP_ADD, 2, 1, in, out) != 0);
  }
  {
    pai_graph_value_id in[] = {a1};
    pai_graph_value_id out[] = {h};
    REQUIRE(pai_graph_add_op(graph, PAI_OP_RELU, 1, 1, in, out) != 0);
  }
  {
    pai_graph_value_id in[] = {h, w2};
    pai_graph_value_id out[] = {g2};
    REQUIRE(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, in, out) != 0);
  }
  {
    pai_graph_value_id in[] = {g2, b2};
    pai_graph_value_id out[] = {y};
    REQUIRE(pai_graph_add_op(graph, PAI_OP_ADD, 2, 1, in, out) != 0);
  }
  pai_graph_set_input(graph, x);
  pai_graph_set_output(graph, y);
}

int
main(void) {
  pai_graph_t graph;
  pai_graph_mem_plan_t mp;
  pai_sched_plan_t plan;
  pai_sched_run_stats_t stats;
  pai_profile_t prof;
  uint8_t *region;
  float w1[MLP_IN * MLP_HID];
  float b1[MLP_HID];
  float w2[MLP_HID * MLP_OUT];
  float b2[MLP_OUT];
  const float x[MLP_IN] = {0.25f, -0.5f, 1.0f, -2.0f, 0.75f, 0.0f, 3.0f,
                           -1.5f};
  double y_expected[MLP_OUT];
  pai_status_t st;

  printf("== T7 exit test: MLP %d->%d->%d (B leg: harness + CPU-ref + "
         "profile; A leg: host-ref mirrors) ==\n",
         MLP_IN, MLP_HID, MLP_OUT);

  fill_weights(w1, b1, w2, b2);
  mlp_oracle(x, w1, b1, w2, b2, y_expected);

  build_mlp(&graph);
  st = pai_graph_memory_plan(&graph, &mp);
  REQUIRE(st == PAI_OK);
  if (st != PAI_OK) {
    return 1;
  }
  st = pai_sched_build(&graph, &mp, PAI_SCHED_INTERACTIVE, NULL, &plan);
  REQUIRE(st == PAI_OK);
  REQUIRE_EQ_UINT(plan.num_steps, 5);

  region = (uint8_t *)calloc(1, (size_t)mp.region_bytes);
  REQUIRE(region != NULL);
  if (region == NULL) {
    return 1;
  }
  memcpy(region + pai_graph_mem_plan_offset(&mp, V_X), x, sizeof(x));
  memcpy(region + pai_graph_mem_plan_offset(&mp, V_W1), w1, sizeof(w1));
  memcpy(region + pai_graph_mem_plan_offset(&mp, V_B1), b1, sizeof(b1));
  memcpy(region + pai_graph_mem_plan_offset(&mp, V_W2), w2, sizeof(w2));
  memcpy(region + pai_graph_mem_plan_offset(&mp, V_B2), b2, sizeof(b2));

  /* --- leg A: MLP forward pass through the T4 host-ref mirrors
   * (pai_host_kernel_t4_* = the GPU serial shaders G45/G47/G48
   * mirrored on host; HW-validated on console 9021, T4C). The graph
   * is kept untouched; the mirrors use the same packed-ab / header
   * ABI as the shaders: matmul header [K, N, a, b], one group per
   * cell; biasadd header [cols, pad, a, bias], one group per cell;
   * relu packed (a,b) pairs. y_mir is compared against the double
   * oracle with pai_ref_compare_f32 (tolerant). --- */
  {
    uint32_t ud[16] = {0};
    uint32_t hdr[6] = {0};
    float g1[MLP_HID], h_mir[MLP_HID];
    float g2[MLP_OUT], y_mir[MLP_OUT];
    float ab[2 * MLP_HID];
    pai_status_t st_a;
    uint64_t mism = 0;
    int all_ok = 1;

    /* G48 matmul: x[1x8] * W1[8x16] -> g1[1x16], one group per cell. */
    hdr[0] = MLP_IN;
    hdr[1] = MLP_HID;
    hdr[2] = (uint32_t)(uintptr_t)x;
    hdr[3] = (uint32_t)((uintptr_t)x >> 32);
    hdr[4] = (uint32_t)(uintptr_t)w1;
    hdr[5] = (uint32_t)((uintptr_t)w1 >> 32);
    ud[2] = (uint32_t)(uintptr_t)hdr;
    ud[3] = (uint32_t)((uintptr_t)hdr >> 32);
    ud[4] = (uint32_t)(uintptr_t)g1;
    ud[5] = (uint32_t)((uintptr_t)g1 >> 32);
    st_a = pai_host_kernel_t4_matmul(NULL, ud, 1, MLP_HID);
    REQUIRE(st_a == PAI_OK);

    /* G47 biasadd: a1[j] = g1[j] + b1[j] (cols = MLP_HID). */
    hdr[0] = MLP_HID;
    hdr[1] = 0;
    hdr[2] = (uint32_t)(uintptr_t)g1;
    hdr[3] = (uint32_t)((uintptr_t)g1 >> 32);
    hdr[4] = (uint32_t)(uintptr_t)b1;
    hdr[5] = (uint32_t)((uintptr_t)b1 >> 32);
    st_a = pai_host_kernel_t4_biasadd(NULL, ud, 1, MLP_HID);
    REQUIRE(st_a == PAI_OK);

    /* G45 relu: packed (a, b) pairs; a = g1 (biasadd wrote in place), b
     * unused. */
    for (int j = 0; j < MLP_HID; j++) {
      ab[2 * j] = g1[j];
      ab[2 * j + 1] = 0.0f;
    }
    ud[2] = (uint32_t)(uintptr_t)ab;
    ud[3] = (uint32_t)((uintptr_t)ab >> 32);
    ud[4] = (uint32_t)(uintptr_t)h_mir;
    ud[5] = (uint32_t)((uintptr_t)h_mir >> 32);
    st_a = pai_host_kernel_t4_relu(NULL, ud, 1, MLP_HID);
    REQUIRE(st_a == PAI_OK);

    /* G48 matmul: h[1x16] * W2[16x8] -> g2[1x8]. */
    hdr[0] = MLP_HID;
    hdr[1] = MLP_OUT;
    hdr[2] = (uint32_t)(uintptr_t)h_mir;
    hdr[3] = (uint32_t)((uintptr_t)h_mir >> 32);
    hdr[4] = (uint32_t)(uintptr_t)w2;
    hdr[5] = (uint32_t)((uintptr_t)w2 >> 32);
    ud[2] = (uint32_t)(uintptr_t)hdr;
    ud[3] = (uint32_t)((uintptr_t)hdr >> 32);
    ud[4] = (uint32_t)(uintptr_t)g2;
    ud[5] = (uint32_t)((uintptr_t)g2 >> 32);
    st_a = pai_host_kernel_t4_matmul(NULL, ud, 1, MLP_OUT);
    REQUIRE(st_a == PAI_OK);

    /* G47 biasadd: y_mir[j] = g2[j] + b2[j] (cols = MLP_OUT). */
    hdr[0] = MLP_OUT;
    hdr[1] = 0;
    hdr[2] = (uint32_t)(uintptr_t)g2;
    hdr[3] = (uint32_t)((uintptr_t)g2 >> 32);
    hdr[4] = (uint32_t)(uintptr_t)b2;
    hdr[5] = (uint32_t)((uintptr_t)b2 >> 32);
    ud[2] = (uint32_t)(uintptr_t)hdr;
    ud[3] = (uint32_t)((uintptr_t)hdr >> 32);
    ud[4] = (uint32_t)(uintptr_t)y_mir;
    ud[5] = (uint32_t)((uintptr_t)y_mir >> 32);
    st_a = pai_host_kernel_t4_biasadd(NULL, ud, 1, MLP_OUT);
    REQUIRE(st_a == PAI_OK);

    {
      /* The oracle is double; snap it to float so both sides are the
       * same arithmetic type (the mirror kernels are f32 end-to-end). */
      float y_want[MLP_OUT];
      for (int j = 0; j < MLP_OUT; j++) {
        y_want[j] = (float)y_expected[j];
      }
      st_a = pai_ref_compare_f32(y_mir, y_want, MLP_OUT, 1e-4f, 1e-4f,
                                 &mism);
      if (st_a != PAI_OK) {
        printf("  FAIL leg A: mirror mismatch at %llu (mir=%f want=%f)\n",
               (unsigned long long)mism,
               mism < MLP_OUT ? (double)y_mir[mism] : 0.0,
               mism < MLP_OUT ? (double)y_want[mism] : 0.0);
        all_ok = 0;
      }
    }
    if (all_ok && st_a == PAI_OK) {
      printf("  PASS host-ref mirrors (G45/G47/G48): y matches oracle\n");
    } else {
      g_failures++;
    }
  }

  /* 1) CPU-ref through the plan executor + kernel vtable gate. */
  st = pai_sched_execute_vtable(&graph, &mp, &plan, region, &stats);
  REQUIRE(st == PAI_OK);
  REQUIRE_EQ_UINT(stats.steps_executed, 5);
  {
    float y_cpu[MLP_OUT];
    int all_ok = 1;
    memcpy(y_cpu, region + pai_graph_mem_plan_offset(&mp, V_Y),
           sizeof(y_cpu));
    for (int j = 0; j < MLP_OUT; j++) {
      double err = fabs((double)y_cpu[j] - y_expected[j]);
      double tol = 1e-4 * (1.0 + fabs(y_expected[j]));
      if (err > tol) {
        printf("  FAIL y[%d]: cpu=%f want=%f (err=%g)\n", j,
               (double)y_cpu[j], y_expected[j], err);
        all_ok = 0;
      }
    }
    if (all_ok) {
      printf("  PASS CPU-ref executor (vtable): y matches oracle\n");
    } else {
      g_failures++;
    }
  }

  /* 2) Registry eager cross-check (T5A) on the first GEMM, against the
   * math oracle (the planner may reuse g1's storage, so the executor
   * buffer is not readable after the run). */
  {
    pai_op_registry_t reg;
    pai_tensor_t tx, tw, th;
    float h_eager[MLP_HID];
    pai_tensor_t *in[2] = {&tx, &tw};
    pai_tensor_t *out[1] = {&th};
    /* Eager matmul is rank-2 only: x viewed as a [1, 8] row. The local
     * x/w1 copies are used (the plan region reuses freed storage). */
    uint64_t s18[2] = {1, MLP_IN};
    uint64_t s816[2] = {MLP_IN, MLP_HID};
    uint64_t s16[1] = {MLP_HID};
    int all_ok = 1;
    REQUIRE(pai_op_registry_builtin(&reg) == PAI_OK);
    REQUIRE(pai_tensor_init(&tx, PAI_DTYPE_F32, 2, s18, (void *)x) == PAI_OK);
    REQUIRE(pai_tensor_init(&tw, PAI_DTYPE_F32, 2, s816, (void *)w1) ==
            PAI_OK);
    REQUIRE(pai_tensor_init(&th, PAI_DTYPE_F32, 1, s16, h_eager) == PAI_OK);
    st = pai_op_eval_f32(&reg, "matmul_f32", in, 2, out, 1, NULL, 0);
    REQUIRE(st == PAI_OK);
    if (st == PAI_OK) {
      for (int j = 0; j < MLP_HID; j++) {
        double want = 0.0;
        for (int i = 0; i < MLP_IN; i++) {
          want += (double)x[i] * w1[i * MLP_HID + j];
        }
        if (fabs((double)h_eager[j] - want) >
            1e-4 * (1.0 + fabs(want))) {
          all_ok = 0;
        }
      }
    }
    if (all_ok && st == PAI_OK) {
      printf("  PASS eager registry matmul_f32 == oracle (W1*x)\n");
    } else {
      printf("  FAIL eager matmul_f32 != oracle (W1*x)\n");
      g_failures++;
    }
  }

  /* 3) Profile plausibility (T6): 5 steps, wall > 0, replay hash. */
  {
    pai_profile_t p;
    uint64_t hash_a, hash_b;
    pai_profile_init(&p);
    st = pai_profile_run_plan(&graph, &mp, &plan, region, 0, &p, &stats);
    REQUIRE(st == PAI_OK);
    REQUIRE_EQ_UINT(p.num_steps, 5);
    REQUIRE(p.total_cpu_ns > 0);
    REQUIRE(p.total_ns >= p.total_cpu_ns);
    REQUIRE(p.hash != 0);
    hash_a = p.hash;
    /* same plan replays to the same structural hash */
    pai_profile_init(&p);
    st = pai_profile_run_plan(&graph, &mp, &plan, region, 0, &p, NULL);
    REQUIRE(st == PAI_OK);
    hash_b = p.hash;
    REQUIRE_EQ_UINT(hash_a, hash_b);
    if (p.hash != 0 && p.num_steps == 5) {
      printf("  PASS profile plausibility: 5 steps, wall=%llu ns, "
             "replay hash=%llu\n",
             (unsigned long long)p.total_ns, (unsigned long long)p.hash);
    } else {
      g_failures++;
    }
    printf("  plan profile log:\n%s", p.log);
  }

  free(region);

  printf("== T7 (B leg) %s ==\n", g_failures ? "FAILED" : "PASSED");
  return g_failures ? 1 : 0;
}
