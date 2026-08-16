#include "test.h"

#include <graph.h>

#include <stdint.h>
#include <string.h>

/* Recompute per-value lifetimes with the same rules as graph.c. */
typedef struct lifetime {
  uint64_t start;
  uint64_t end;
} lifetime_t;

static void
compute_lifetimes(const pai_graph_t *g, lifetime_t *lt) {
  uint32_t last_use[PAI_GRAPH_MAX_VALUES + 1];
  uint32_t used[PAI_GRAPH_MAX_VALUES + 1];

  memset(last_use, 0, sizeof(last_use));
  memset(used, 0, sizeof(used));
  for (pai_graph_op_id o = 1; o <= g->num_ops; o++) {
    const pai_graph_op_t *op = &g->ops[o];
    for (uint32_t i = 0; i < op->num_inputs; i++) {
      pai_graph_value_id v = op->inputs[i];
      used[v] = 1;
      if (op->epoch > last_use[v]) {
        last_use[v] = op->epoch;
      }
    }
  }

  for (pai_graph_value_id v = 1; v <= g->num_values; v++) {
    lt[v].start = g->values[v].producer != 0
                      ? g->ops[g->values[v].producer].epoch
                      : 0;
    if (g->values[v].is_output) {
      lt[v].end = g->num_ops;
    } else if (used[v]) {
      lt[v].end = (uint64_t)last_use[v] + 1;
    } else {
      lt[v].end = lt[v].start + 1;
    }
  }
}

/* Live-lifetime pairs must never share storage. */
static void
check_graph_plan(const pai_graph_t *g, const pai_graph_mem_plan_t *plan) {
  lifetime_t lt[PAI_GRAPH_MAX_VALUES + 1];

  compute_lifetimes(g, lt);
  CHECK(plan->region_bytes <= plan->naive_bytes);
  CHECK(plan->region_bytes >= plan->peak_live);

  for (pai_graph_value_id i = 1; i <= g->num_values; i++) {
    uint64_t off_i = pai_graph_mem_plan_offset(plan, i);
    CHECK_EQ_UINT(off_i % g->values[i].align, 0);
    for (pai_graph_value_id j = i + 1; j <= g->num_values; j++) {
      int lifetime_overlap = lt[i].start < lt[j].end && lt[j].start < lt[i].end;
      int mem_overlap = off_i < pai_graph_mem_plan_offset(plan, j) +
                                     g->values[j].size_bytes &&
                        pai_graph_mem_plan_offset(plan, j) <
                            off_i + g->values[i].size_bytes;
      CHECK(!(lifetime_overlap && mem_overlap));
    }
  }
}

static pai_graph_value_id
f32_value(pai_graph_t *g, uint64_t d0, uint64_t d1) {
  uint64_t shape[2] = {d0, d1};
  return pai_graph_add_value(g, PAI_DTYPE_F32, 2, shape, 16);
}

TEST_MAIN_BEGIN()

{
  /* Small synthetic network: x -> MM(w) -> +b -> relu -> MM(w2) -> softmax. */
  pai_graph_t g;
  pai_graph_value_id x, w, h1, b, h2, h3, w2, h4, y;
  pai_graph_op_id mm1, add1, relu1, mm2, sm;
  pai_graph_mem_plan_t plan;
  uint64_t v3off, v4off;

  pai_graph_init(&g);

  x = f32_value(&g, 1, 16);
  w = f32_value(&g, 16, 16);
  b = f32_value(&g, 1, 16);
  w2 = f32_value(&g, 16, 8);

  CHECK(x != 0 && w != 0 && b != 0 && w2 != 0);
  CHECK_EQ_UINT(g.values[x].size_bytes, 64);
  CHECK_EQ_UINT(g.values[w].size_bytes, 1024);

  h1 = f32_value(&g, 1, 16);
  h2 = f32_value(&g, 1, 16);
  h3 = f32_value(&g, 1, 16);
  h4 = f32_value(&g, 1, 8);
  y = f32_value(&g, 1, 8);

  mm1 = pai_graph_add_op(&g, PAI_OP_MATMUL, 2, 1, (pai_graph_value_id[]){x, w},
                         (pai_graph_value_id[]){h1});
  add1 = pai_graph_add_op(&g, PAI_OP_ADD, 2, 1, (pai_graph_value_id[]){h1, b},
                          (pai_graph_value_id[]){h2});
  relu1 = pai_graph_add_op(&g, PAI_OP_RELU, 1, 1, (pai_graph_value_id[]){h2},
                           (pai_graph_value_id[]){h3});
  mm2 = pai_graph_add_op(&g, PAI_OP_MATMUL, 2, 1, (pai_graph_value_id[]){h3, w2},
                         (pai_graph_value_id[]){h4});
  sm = pai_graph_add_op(&g, PAI_OP_SOFTMAX, 1, 1, (pai_graph_value_id[]){h4},
                        (pai_graph_value_id[]){y});

  CHECK(mm1 != 0 && add1 != 0 && relu1 != 0 && mm2 != 0 && sm != 0);

  /* A value cannot be produced twice. */
  CHECK_EQ_UINT(pai_graph_add_op(&g, PAI_OP_ADD, 1, 1,
                                 (pai_graph_value_id[]){x},
                                 (pai_graph_value_id[]){h1}),
                0);

  pai_graph_set_input(&g, x);
  pai_graph_set_input(&g, w);
  pai_graph_set_input(&g, b);
  pai_graph_set_input(&g, w2);
  pai_graph_set_output(&g, y);

  CHECK_EQ_INT(pai_graph_topo_sort(&g), PAI_OK);
  CHECK_EQ_UINT(g.ops[mm1].epoch, 0);
  CHECK_EQ_UINT(g.ops[add1].epoch, 1);
  CHECK_EQ_UINT(g.ops[relu1].epoch, 2);
  CHECK_EQ_UINT(g.ops[mm2].epoch, 3);
  CHECK_EQ_UINT(g.ops[sm].epoch, 4);

  CHECK_EQ_INT(pai_graph_memory_plan(&g, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.region_bytes, 1728);
  CHECK_EQ_UINT(plan.naive_bytes, 1920);
  CHECK_EQ_UINT(plan.peak_live, 1728);
  CHECK_EQ_UINT(plan.base.reuses, 4); /* h2/h3/h4/y all reuse freed space */

  /* Deterministic placement: the longest-lived input lands first. */
  CHECK_EQ_UINT(pai_graph_mem_plan_offset(&plan, w2), 0);

  /* The 64-byte bias/activation pair shares the two adjacent slots. */
  v3off = pai_graph_mem_plan_offset(&plan, h1);
  v4off = pai_graph_mem_plan_offset(&plan, b);
  CHECK(v3off == 512 || v3off == 576);
  CHECK(v4off == 512 || v4off == 576);
  CHECK(v3off != v4off);

  /* Every planned activation lives inside the planned region. */
  for (pai_graph_value_id v = 1; v <= g.num_values; v++) {
    CHECK(pai_graph_mem_plan_offset(&plan, v) < plan.region_bytes);
  }

  check_graph_plan(&g, &plan);
  pai_graph_mem_plan_free(&plan);
}

{
  /* Cycle detection. */
  pai_graph_t g;
  pai_graph_value_id v1, v2, v3;
  pai_graph_op_id a, b, c;

  pai_graph_init(&g);
  v1 = f32_value(&g, 1, 1);
  v2 = f32_value(&g, 1, 1);
  v3 = f32_value(&g, 1, 1);

  a = pai_graph_add_op(&g, PAI_OP_CUSTOM, 1, 1, (pai_graph_value_id[]){v3},
                       (pai_graph_value_id[]){v1});
  b = pai_graph_add_op(&g, PAI_OP_CUSTOM, 1, 1, (pai_graph_value_id[]){v1},
                       (pai_graph_value_id[]){v2});
  c = pai_graph_add_op(&g, PAI_OP_CUSTOM, 1, 1, (pai_graph_value_id[]){v2},
                       (pai_graph_value_id[]){v3});
  CHECK(a != 0 && b != 0 && c != 0);
  CHECK_EQ_INT(pai_graph_topo_sort(&g), PAI_ERR_MISMATCH);
  CHECK_EQ_INT(pai_graph_memory_plan(&g, NULL), PAI_ERR_INVALID_ARG);
}

{
  /* Dangling value (produced, never consumed) still gets storage. */
  pai_graph_t g;
  pai_graph_value_id v;
  pai_graph_mem_plan_t plan;

  pai_graph_init(&g);
  v = f32_value(&g, 4, 4);
  CHECK(pai_graph_add_op(&g, PAI_OP_CUSTOM, 0, 1, NULL,
                         (pai_graph_value_id[]){v}) != 0);
  CHECK_EQ_INT(pai_graph_memory_plan(&g, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.region_bytes, 64); /* 64 bytes, no reuse available */
  CHECK_EQ_UINT(pai_graph_mem_plan_offset(&plan, v), 0);
  pai_graph_mem_plan_free(&plan);
}

{
  /* Empty graph plans to zero. */
  pai_graph_t g;
  pai_graph_mem_plan_t plan;

  pai_graph_init(&g);
  CHECK_EQ_INT(pai_graph_memory_plan(&g, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.region_bytes, 0);
  CHECK_EQ_UINT(plan.naive_bytes, 0);
  pai_graph_mem_plan_free(&plan);
}

{
  /* Validation of values and ops. */
  pai_graph_t g;
  pai_graph_value_id v1, v2;
  uint64_t bad_rank0[1] = {4};
  uint64_t zero_dim[1] = {0};

  pai_graph_init(&g);
  CHECK_EQ_UINT(pai_graph_add_value(&g, PAI_DTYPE_F32, 0, bad_rank0, 16), 0);
  CHECK_EQ_UINT(pai_graph_add_value(&g, PAI_DTYPE_F32, 1, zero_dim, 16), 0);
  CHECK_EQ_UINT(pai_graph_add_value(&g, (pai_dtype_t)999, 1, zero_dim, 16), 0);

  v1 = f32_value(&g, 1, 1);
  v2 = f32_value(&g, 1, 1);
  CHECK_EQ_UINT(pai_graph_add_op(&g, PAI_OP_ADD, 2, 1,
                                 (pai_graph_value_id[]){v1, 999},
                                 (pai_graph_value_id[]){v2}),
                0); /* invalid input id */
  CHECK_EQ_UINT(pai_graph_add_op(&g, PAI_OP_ADD, 17, 0, NULL, NULL), 0);

  /* Self-consumption and duplicate outputs are rejected. */
  CHECK_EQ_UINT(pai_graph_add_op(&g, PAI_OP_RELU, 1, 1,
                                 (pai_graph_value_id[]){v1},
                                 (pai_graph_value_id[]){v1}),
                0);
  CHECK_EQ_UINT(pai_graph_add_op(&g, PAI_OP_ADD, 0, 2, NULL,
                                 (pai_graph_value_id[]){v1, v1}),
                0);

  /* Topo sort is idempotent. */
  {
    pai_graph_t g2;
    pai_graph_value_id a, b, c, d;
    pai_graph_op_id o1, o2;
    uint32_t e1, e2;

    pai_graph_init(&g2);
    a = f32_value(&g2, 1, 1);
    b = f32_value(&g2, 1, 1);
    c = f32_value(&g2, 1, 1);
    d = f32_value(&g2, 1, 1);
    o1 = pai_graph_add_op(&g2, PAI_OP_ADD, 2, 1, (pai_graph_value_id[]){a, b},
                          (pai_graph_value_id[]){c});
    o2 = pai_graph_add_op(&g2, PAI_OP_RELU, 1, 1, (pai_graph_value_id[]){c},
                          (pai_graph_value_id[]){d});
    CHECK(o1 != 0 && o2 != 0);
    CHECK_EQ_INT(pai_graph_topo_sort(&g2), PAI_OK);
    e1 = g2.ops[o1].epoch;
    e2 = g2.ops[o2].epoch;
    CHECK_EQ_INT(pai_graph_topo_sort(&g2), PAI_OK);
    CHECK_EQ_UINT(g2.ops[o1].epoch, e1);
    CHECK_EQ_UINT(g2.ops[o2].epoch, e2);
  }

  /* Kind string helper. */
  CHECK(strcmp("gemm", pai_graph_op_kind_str(PAI_OP_GEMM)) == 0);
  CHECK(strcmp("none", pai_graph_op_kind_str(PAI_OP_NONE)) == 0);
  CHECK(strcmp("?", pai_graph_op_kind_str((pai_graph_op_kind_t)999)) == 0);
}

TEST_MAIN_END()
