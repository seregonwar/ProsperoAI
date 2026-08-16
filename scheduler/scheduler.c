#include "scheduler.h"

#include <cpu/reference/ref_ops.h>

#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

/* Monotonic clock for the per-step hook (T6 profiler). */
static uint64_t
sched_clock_ns(void) {
#ifdef _WIN32
  static LARGE_INTEGER freq = {0};
  LARGE_INTEGER now;

  if (freq.QuadPart == 0) {
    QueryPerformanceFrequency(&freq);
  }
  QueryPerformanceCounter(&now);
  return (uint64_t)((double)now.QuadPart * 1000000000.0 /
                    (double)freq.QuadPart);
#else
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
#endif
}

static const char *const k_policy_names[] = {
    "interactive", "throughput", "exclusive", "balanced",
};
static const char *const k_device_names[] = {
    "any", "cpu", "gpu",
};
static const char *const k_mode_names[] = {
    "performance", "balanced", "capacity",
};

const char *
pai_sched_policy_name(pai_sched_policy_t policy) {
  if ((int)policy < 0 || (int)policy >=
                             (int)(sizeof(k_policy_names) /
                                   sizeof(k_policy_names[0]))) {
    return "?";
  }
  return k_policy_names[policy];
}

const char *
pai_sched_device_name(uint8_t device) {
  if (device >= PAI_SCHED_DEV_GPU + 1 ||
      device >= sizeof(k_device_names) / sizeof(k_device_names[0])) {
    return "?";
  }
  return k_device_names[device];
}

const char *
pai_sched_memory_mode_name(uint8_t mode) {
  if (mode >= PAI_SCHED_MEM_CAPACITY + 1 ||
      mode >= sizeof(k_mode_names) / sizeof(k_mode_names[0])) {
    return "?";
  }
  return k_mode_names[mode];
}

/* Default placement per op kind (whitepaper §3.2: heavy compute on the
 * GPU; scheduling/control-flow-adjacent work on the CPU). */
static uint8_t
default_device(pai_graph_op_kind_t kind) {
  switch (kind) {
    case PAI_OP_GEMM:
    case PAI_OP_GEMV:
    case PAI_OP_MATMUL:
    case PAI_OP_ATTENTION:
    case PAI_OP_SOFTMAX:
      return PAI_SCHED_DEV_GPU;
    default:
      return PAI_SCHED_DEV_CPU;
  }
}

/* Compute kinds that PAI_SCHED_EXCLUSIVE pushes to the GPU. */
static int
is_compute_kind(pai_graph_op_kind_t kind) {
  switch (kind) {
    case PAI_OP_ADD:
    case PAI_OP_MUL:
    case PAI_OP_GEMM:
    case PAI_OP_GEMV:
    case PAI_OP_MATMUL:
    case PAI_OP_RELU:
    case PAI_OP_SOFTMAX:
    case PAI_OP_LAYERNORM:
    case PAI_OP_RMSNORM:
    case PAI_OP_ROPE:
    case PAI_OP_ATTENTION:
      return 1;
    default:
      return 0;
  }
}

static pai_status_t
resolve_device(const pai_graph_t *graph, pai_graph_op_id op_id,
               pai_sched_policy_t policy, const uint8_t *hints,
               uint8_t *out_device) {
  uint8_t device;

  if (hints != NULL) {
    uint8_t h = hints[op_id];
    if (h != PAI_SCHED_DEV_ANY) {
      if (h != PAI_SCHED_DEV_CPU && h != PAI_SCHED_DEV_GPU) {
        return PAI_ERR_INVALID_ARG;
      }
      *out_device = h;
      return PAI_OK;
    }
  }

  device = default_device(graph->ops[op_id].kind);
  if (policy == PAI_SCHED_EXCLUSIVE && is_compute_kind(graph->ops[op_id].kind)) {
    device = PAI_SCHED_DEV_GPU;
  }
  *out_device = device;
  return PAI_OK;
}

static pai_status_t
sched_build_impl(pai_graph_t *graph, const pai_graph_mem_plan_t *mem_plan,
                pai_sched_policy_t policy, const uint8_t *device_hints,
                pai_sched_plan_t *out, int persistent) {
  pai_graph_op_id op_at_epoch[PAI_GRAPH_MAX_OPS];
  uint32_t start[PAI_GRAPH_MAX_VALUES + 1];
  uint32_t end[PAI_GRAPH_MAX_VALUES + 1];
  uint64_t live_before;
  uint64_t produced;
  uint64_t released;
  pai_status_t st;

  if (graph == NULL || mem_plan == NULL || out == NULL ||
      (int)policy < 0 || policy > PAI_SCHED_BALANCED) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));

  if (!graph->sorted) {
    st = pai_graph_topo_sort(graph);
    if (st != PAI_OK) {
      return st;
    }
  }
  st = persistent ? pai_graph_value_lifetimes_persistent(graph, start, end)
                  : pai_graph_value_lifetimes(graph, start, end);
  if (st != PAI_OK) {
    return st;
  }

  memset(op_at_epoch, 0, sizeof(op_at_epoch));
  for (pai_graph_op_id o = 1; o <= graph->num_ops; o++) {
    op_at_epoch[graph->ops[o].epoch] = o;
  }

  out->policy = policy;
  out->region_bytes = mem_plan->region_bytes;
  out->naive_bytes = mem_plan->naive_bytes;
  out->peak_live = mem_plan->peak_live;
  out->num_steps = graph->num_ops;

  for (uint32_t s = 0; s < graph->num_ops; s++) {
    pai_graph_op_id o = op_at_epoch[s];
    pai_sched_step_t *step = &out->steps[s];

    /* A dense epoch assignment is guaranteed by the topo sort; any
     * gap means an inconsistent graph. */
    if (o == 0) {
      return PAI_ERR_MISMATCH;
    }

    st = resolve_device(graph, o, policy, device_hints, &step->device);
    if (st != PAI_OK) {
      return st;
    }

    live_before = 0;
    produced = 0;
    released = 0;
    for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
      const pai_graph_value_t *value = &graph->values[v];
      if (start[v] < s && end[v] > s) {
        live_before += value->size_bytes;
      }
      if (end[v] == s) {
        released += value->size_bytes;
      }
      if (start[v] == s) {
        produced += value->size_bytes;
      }
    }

    step->op_id = o;
    step->epoch = s;
    step->produced_bytes = produced;
    step->released_bytes = released;
    step->live_bytes = live_before + produced;

    if (step->device == PAI_SCHED_DEV_GPU) {
      out->gpu_steps++;
    } else {
      out->cpu_steps++;
    }
  }

  return PAI_OK;
}

pai_status_t
pai_sched_build(pai_graph_t *graph, const pai_graph_mem_plan_t *mem_plan,
                pai_sched_policy_t policy, const uint8_t *device_hints,
                pai_sched_plan_t *out) {
  return sched_build_impl(graph, mem_plan, policy, device_hints, out, 0);
}

pai_status_t
pai_sched_build_persistent(pai_graph_t *graph,
                           const pai_graph_mem_plan_t *mem_plan,
                           pai_sched_policy_t policy,
                           const uint8_t *device_hints,
                           pai_sched_plan_t *out) {
  return sched_build_impl(graph, mem_plan, policy, device_hints, out, 1);
}

pai_sched_memory_mode_t
pai_sched_plan_memory_mode(const pai_sched_plan_t *plan, uint64_t budget) {
  if (plan == NULL) {
    return PAI_SCHED_MEM_CAPACITY;
  }
  if (budget >= plan->region_bytes) {
    return PAI_SCHED_MEM_PERFORMANCE;
  }
  if (budget >= plan->peak_live) {
    return PAI_SCHED_MEM_BALANCED;
  }
  return PAI_SCHED_MEM_CAPACITY;
}

pai_status_t
pai_sched_plan_check(const pai_sched_plan_t *plan, uint64_t budget) {
  if (plan == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (budget < plan->peak_live) {
    return PAI_ERR_NOMEM;
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Reference executor                                                  */
/* ------------------------------------------------------------------ */

static uint64_t
value_nelem(const pai_graph_t *graph, pai_graph_value_id v) {
  const pai_graph_value_t *value = &graph->values[v];
  uint64_t n = 1;
  for (uint32_t d = 0; d < value->rank; d++) {
    n *= value->shape[d];
  }
  return n;
}

static float *
value_ptr(const pai_graph_mem_plan_t *mem_plan, uint8_t *region,
          pai_graph_value_id v) {
  return (float *)(void *)(region + pai_graph_mem_plan_offset(mem_plan, v));
}

/*
 * GEMM/GEMV dispatch with shape validation. Rank-1 first operands are
 * row vectors: GEMM(x[k], W[k,n]) -> y[n] is accepted with m = 1.
 */
static pai_status_t
run_gemm(const pai_graph_t *graph, const pai_graph_mem_plan_t *mem_plan,
         uint8_t *region, pai_graph_value_id a_id, pai_graph_value_id b_id,
         pai_graph_value_id c_id, int is_gemv) {
  const pai_graph_value_t *a = &graph->values[a_id];
  const pai_graph_value_t *b = &graph->values[b_id];
  const pai_graph_value_t *c = &graph->values[c_id];
  uint64_t m;
  uint64_t n;
  uint64_t k;

  if (is_gemv) {
    /* y[m] = A[m,k] * x[k] */
    if (a->rank != 2 || b->rank != 1 || c->rank != 1) {
      return PAI_ERR_MISMATCH;
    }
    m = a->shape[0];
    k = a->shape[1];
    n = 1;
    if (b->shape[0] != k || c->shape[0] != m) {
      return PAI_ERR_MISMATCH;
    }
  } else if (a->rank == 1) {
    /* row vector: y[n] = x[k] * W[k,n] */
    m = 1;
    k = a->shape[0];
    if (b->rank != 2 || b->shape[0] != k) {
      return PAI_ERR_MISMATCH;
    }
    n = b->shape[1];
    if (c->rank == 1) {
      if (c->shape[0] != n) {
        return PAI_ERR_MISMATCH;
      }
    } else if (c->rank == 2) {
      if (c->shape[0] != 1 || c->shape[1] != n) {
        return PAI_ERR_MISMATCH;
      }
    } else {
      return PAI_ERR_MISMATCH;
    }
  } else {
    /* full matrix multiply: C[m,n] = A[m,k] * B[k,n] */
    if (a->rank != 2 || b->rank != 2) {
      return PAI_ERR_MISMATCH;
    }
    m = a->shape[0];
    k = a->shape[1];
    n = b->shape[1];
    if (b->shape[0] != k || c->rank != 2 || c->shape[0] != m ||
        c->shape[1] != n) {
      return PAI_ERR_MISMATCH;
    }
  }

  return pai_ref_gemm_f32(m, n, k, value_ptr(mem_plan, region, a_id),
                          value_ptr(mem_plan, region, b_id),
                          value_ptr(mem_plan, region, c_id));
}

pai_status_t
pai_sched_execute_hooked(const pai_graph_t *graph,
                         const pai_graph_mem_plan_t *mem_plan,
                         const pai_sched_plan_t *plan, uint8_t *region,
                         pai_sched_run_stats_t *out_stats,
                         pai_sched_step_fn step_fn, void *hook_ctx) {
  pai_sched_run_stats_t stats;
  pai_status_t st;

  if (graph == NULL || mem_plan == NULL || plan == NULL || region == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (plan->num_steps != graph->num_ops) {
    return PAI_ERR_MISMATCH;
  }
  memset(&stats, 0, sizeof(stats));

  for (uint32_t i = 0; i < plan->num_steps; i++) {
    const pai_sched_step_t *step = &plan->steps[i];
    const pai_graph_op_t *op = &graph->ops[step->op_id];

    /* The v0 reference executor is f32-only: element-count operations
     * would mis-size non-f32 buffers (memory-safety). */
    for (uint32_t k = 0; k < op->num_inputs; k++) {
      if (graph->values[op->inputs[k]].dtype != PAI_DTYPE_F32) {
        return PAI_ERR_UNSUPPORTED;
      }
    }
    for (uint32_t k = 0; k < op->num_outputs; k++) {
      if (graph->values[op->outputs[k]].dtype != PAI_DTYPE_F32) {
        return PAI_ERR_UNSUPPORTED;
      }
    }

    const pai_graph_value_t *a = op->num_inputs > 0
                                     ? &graph->values[op->inputs[0]]
                                     : NULL;
    const pai_graph_value_t *b = op->num_inputs > 1
                                     ? &graph->values[op->inputs[1]]
                                     : NULL;
    const pai_graph_value_t *c = op->num_outputs > 0
                                     ? &graph->values[op->outputs[0]]
                                     : NULL;
    uint64_t na = a != NULL ? value_nelem(graph, op->inputs[0]) : 0;
    uint64_t nb = b != NULL ? value_nelem(graph, op->inputs[1]) : 0;
    uint64_t nc = c != NULL ? value_nelem(graph, op->outputs[0]) : 0;
    float *pa = a != NULL ? value_ptr(mem_plan, region, op->inputs[0]) : NULL;
    float *pb = b != NULL ? value_ptr(mem_plan, region, op->inputs[1]) : NULL;
    float *pc = c != NULL ? value_ptr(mem_plan, region, op->outputs[0]) : NULL;
    uint64_t step_start = sched_clock_ns();

    switch (op->kind) {
      case PAI_OP_ADD:
        if (op->num_inputs != 2 || op->num_outputs != 1 || na != nb ||
            nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_vecadd_f32(pa, pb, pc, na);
        break;

      case PAI_OP_MUL:
        if (op->num_inputs != 2 || op->num_outputs != 1 || na != nb ||
            nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_vecmul_f32(pa, pb, pc, na);
        break;

      case PAI_OP_GEMM:
      case PAI_OP_MATMUL:
        if (op->num_inputs != 2 || op->num_outputs != 1) {
          return PAI_ERR_MISMATCH;
        }
        st = run_gemm(graph, mem_plan, region, op->inputs[0], op->inputs[1],
                      op->outputs[0], 0);
        break;

      case PAI_OP_GEMV:
        if (op->num_inputs != 2 || op->num_outputs != 1) {
          return PAI_ERR_MISMATCH;
        }
        st = run_gemm(graph, mem_plan, region, op->inputs[0], op->inputs[1],
                      op->outputs[0], 1);
        break;

      case PAI_OP_RELU:
        if (op->num_inputs != 1 || op->num_outputs != 1 || nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_relu_f32(pa, pc, na);
        break;

      case PAI_OP_SOFTMAX:
        if (op->num_inputs != 1 || op->num_outputs != 1 || nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_softmax_f32(pa, pc, na);
        break;

      case PAI_OP_RMSNORM: {
        /* Per-row normalization for rank-2 tensors (transformer
         * activations are [seq, dim]); rank-1 is a single row. The
         * optional second input is the per-element gain (gamma). */
        uint64_t rows;
        uint64_t cols;
        if (op->num_inputs < 1 || op->num_inputs > 2 || op->num_outputs != 1 ||
            a->rank < 1 || a->rank > 2) {
          return PAI_ERR_MISMATCH;
        }
        rows = a->rank == 2 ? a->shape[0] : 1;
        cols = a->shape[a->rank - 1];
        if (c->rank != a->rank || c->shape[a->rank - 1] != cols ||
            (a->rank == 2 && c->shape[0] != rows)) {
          return PAI_ERR_MISMATCH;
        }
        if (op->num_inputs == 2) {
          const pai_graph_value_t *g = &graph->values[op->inputs[1]];
          float *pg = value_ptr(mem_plan, region, op->inputs[1]);
          if (g->rank != 1 || g->shape[0] != cols) {
            return PAI_ERR_MISMATCH;
          }
          for (uint64_t r = 0; r < rows; r++) {
            st = pai_ref_rmsnorm_gamma_f32(pa + r * cols, pc + r * cols, cols,
                                           pg, 0.0f);
            if (st != PAI_OK) {
              break;
            }
          }
        } else {
          for (uint64_t r = 0; r < rows; r++) {
            st = pai_ref_rmsnorm_f32(pa + r * cols, pc + r * cols, cols, 0.0f);
            if (st != PAI_OK) {
              break;
            }
          }
        }
        break;
      }

      case PAI_OP_SILU:
        if (op->num_inputs != 1 || op->num_outputs != 1 || nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_silu_f32(pa, pc, na);
        break;

      case PAI_OP_LAYERNORM:
        /* v0: single-input layernorm (no gamma/beta tensors yet). */
        if (op->num_inputs != 1 || op->num_outputs != 1 || nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_layernorm_f32(pa, pc, na, NULL, NULL, 0.0f);
        break;

      case PAI_OP_CONCAT: {
        uint64_t expect;
        if (op->num_inputs != 2 || op->num_outputs != 1) {
          return PAI_ERR_MISMATCH;
        }
        expect = na + nb;
        if (nc != expect) {
          return PAI_ERR_MISMATCH;
        }
        if (a->rank == 2 && b->rank == 2 && c->rank == 2 &&
            (a->shape[1] != b->shape[1] || c->shape[1] != a->shape[1])) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_concat_f32(pa, pb, pc, na, nb);
        break;
      }

      case PAI_OP_COPY:
      case PAI_OP_RESHAPE:
        if (op->num_inputs != 1 || op->num_outputs != 1 || nc != na) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_copy_f32(pa, pc, na);
        break;

      case PAI_OP_CONVERT:
        if (op->num_inputs != 1 || op->num_outputs != 1 ||
            a->dtype != c->dtype) {
          return PAI_ERR_UNSUPPORTED; /* dtype-changing convert: Phase 2 */
        }
        st = pai_ref_copy_f32(pa, pc, na);
        break;

      case PAI_OP_ROPE: {
        /* Rotary embeddings: position-major x [seq, hd] (rank 2) or
         * [seq, H, hd] (rank 3), with position tables cos/sin
         * [ctx, r2] (r2 = rotary dim / 2). */
        const pai_graph_value_t *co = op->num_inputs > 1
                                          ? &graph->values[op->inputs[1]]
                                          : NULL;
        const pai_graph_value_t *si = op->num_inputs > 2
                                          ? &graph->values[op->inputs[2]]
                                          : NULL;
        uint64_t hd;
        uint64_t seq;
        uint64_t r2;
        uint64_t rows;
        uint64_t heads;
        if (op->num_inputs != 3 || op->num_outputs != 1 || a->rank < 2 ||
            a->rank > 3 || c->rank != a->rank) {
          return PAI_ERR_MISMATCH;
        }
        hd = a->shape[a->rank - 1];
        seq = a->shape[0];
        heads = a->rank == 3 ? a->shape[1] : 1;
        if (c->shape[a->rank - 1] != hd || c->shape[0] != seq ||
            (a->rank == 3 && c->shape[1] != heads)) {
          return PAI_ERR_MISMATCH;
        }
        if (co == NULL || si == NULL || co->rank != 2 || si->rank != 2 ||
            co->shape[1] != si->shape[1] || co->shape[0] != si->shape[0]) {
          return PAI_ERR_MISMATCH;
        }
        r2 = co->shape[1];
        if (2 * r2 > hd || seq == 0 || seq > co->shape[0] || hd == 0 ||
            heads == 0 || nc % hd != 0) {
          return PAI_ERR_MISMATCH;
        }
        rows = nc / hd;
        st = pai_ref_rope_f32(pa, rows, hd, seq, heads,
                              value_ptr(mem_plan, region, op->inputs[1]),
                              value_ptr(mem_plan, region, op->inputs[2]), r2,
                              pc);
        break;
      }

      case PAI_OP_ATTENTION: {
        /* Causal multi-head attention, position-major: q [seq, H, hd],
         * k/v [seq, HK, hd] -> out [seq, H, hd] (grouped-query: H %
         * HK == 0). */
        const pai_graph_value_t *v = op->num_inputs > 2
                                         ? &graph->values[op->inputs[2]]
                                         : NULL;
        uint64_t hh;
        uint64_t hk;
        uint64_t seq;
        uint64_t hd;
        if (op->num_inputs != 3 || op->num_outputs != 1 || a->rank != 3 ||
            b->rank != 3 || c->rank != 3 || v == NULL) {
          return PAI_ERR_MISMATCH;
        }
        seq = a->shape[0];
        hh = a->shape[1];
        hd = a->shape[2];
        hk = b->shape[1];
        if (b->shape[0] != seq || b->shape[2] != hd || v->rank != 3 ||
            v->shape[0] != seq || v->shape[1] != hk || v->shape[2] != hd ||
            c->rank != 3 || c->shape[0] != seq || c->shape[1] != hh ||
            c->shape[2] != hd || hh == 0 || hk == 0 || hh % hk != 0) {
          return PAI_ERR_MISMATCH;
        }
        st = pai_ref_attention_f32(hh, hk, seq, hd, pa, pb,
                                   value_ptr(mem_plan, region, op->inputs[2]),
                                   pc);
        break;
      }

      case PAI_OP_CUSTOM:
        return PAI_ERR_UNSUPPORTED;

      default:
        return PAI_ERR_UNSUPPORTED;
    }

    if (st != PAI_OK) {
      return st;
    }
    stats.steps_executed++;
    if (step->device == PAI_SCHED_DEV_GPU) {
      stats.gpu_steps++;
    } else {
      stats.cpu_steps++;
    }
    if (step_fn != NULL) {
      step_fn(hook_ctx, i, step->op_id, step->device,
              sched_clock_ns() - step_start);
    }
  }

  if (out_stats != NULL) {
    *out_stats = stats;
  }
  return PAI_OK;
}

pai_status_t
pai_sched_execute(const pai_graph_t *graph, const pai_graph_mem_plan_t *mem_plan,
                  const pai_sched_plan_t *plan, uint8_t *region,
                  pai_sched_run_stats_t *out_stats) {
  return pai_sched_execute_hooked(graph, mem_plan, plan, region, out_stats,
                                  NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* Kernel vtable (T5B)                                                  */
/* ------------------------------------------------------------------ */

/* Graph op kind -> registry op name (T5A vocabulary). */
static const char *
op_registry_name(pai_graph_op_kind_t kind) {
  switch (kind) {
    case PAI_OP_ADD:
      return "vecadd_f32";
    case PAI_OP_MUL:
      return "vecmul_f32";
    case PAI_OP_GEMM:
    case PAI_OP_MATMUL:
      return "matmul_f32";
    case PAI_OP_GEMV:
      return "gemv_f32";
    case PAI_OP_RELU:
      return "relu_f32";
    case PAI_OP_SOFTMAX:
      return "softmax_f32";
    case PAI_OP_SILU:
      return "silu_f32";
    case PAI_OP_LAYERNORM:
      return "layernorm_f32";
    case PAI_OP_RMSNORM:
      return "rmsnorm_f32";
    case PAI_OP_CONCAT:
      return "concat_f32";
    case PAI_OP_COPY:
    case PAI_OP_RESHAPE:
    case PAI_OP_CONVERT:
      return "copy_f32";
    case PAI_OP_ROPE:
      return "rope_f32";
    case PAI_OP_ATTENTION:
      return "attention_f32";
    default:
      return NULL; /* PAI_OP_NONE / PAI_OP_CUSTOM */
  }
}

pai_status_t
pai_kernel_select(const pai_graph_t *graph, pai_graph_op_id op_id,
                  const pai_op_registry_t *reg, pai_kernel_entry_t *out) {
  pai_op_registry_t local_reg;
  const pai_op_descriptor_t *desc;
  const char *name;

  if (graph == NULL || op_id == 0 || op_id > graph->num_ops || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (reg == NULL) {
    pai_status_t st = pai_op_registry_builtin(&local_reg);
    if (st != PAI_OK) {
      return st;
    }
    reg = &local_reg;
  }

  name = op_registry_name(graph->ops[op_id].kind);
  if (name == NULL) {
    return PAI_ERR_UNSUPPORTED; /* CUSTOM / unknown kinds */
  }
  desc = pai_op_registry_lookup(reg, name);
  if (desc == NULL) {
    return PAI_ERR_UNSUPPORTED; /* op not present in this registry */
  }

  memset(out, 0, sizeof(*out));
  out->kind = graph->ops[op_id].kind;
  out->op_name = desc->name;
  out->kernel_id = desc->kernel_id;
  out->device = default_device(graph->ops[op_id].kind);
  out->caps = desc->caps;
  return PAI_OK;
}

pai_status_t
pai_kernel_plan_check(const pai_graph_t *graph, const pai_sched_plan_t *plan,
                      uint32_t caps_required, const pai_op_registry_t *reg) {
  pai_op_registry_t local_reg;
  pai_status_t st;

  if (graph == NULL || plan == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (reg == NULL) {
    st = pai_op_registry_builtin(&local_reg);
    if (st != PAI_OK) {
      return st;
    }
    reg = &local_reg;
  }

  for (uint32_t i = 0; i < plan->num_steps; i++) {
    pai_kernel_entry_t entry;
    st = pai_kernel_select(graph, plan->steps[i].op_id, reg, &entry);
    if (st != PAI_OK) {
      return st;
    }
    if ((entry.caps & caps_required) != caps_required) {
      return PAI_ERR_UNSUPPORTED; /* no implementation for this backend */
    }
  }
  return PAI_OK;
}

pai_status_t
pai_sched_execute_vtable(const pai_graph_t *graph,
                         const pai_graph_mem_plan_t *mem_plan,
                         const pai_sched_plan_t *plan, uint8_t *region,
                         pai_sched_run_stats_t *out_stats) {
  pai_status_t st = pai_kernel_plan_check(graph, plan, PAI_OP_CAP_CPU_REF, NULL);
  if (st != PAI_OK) {
    return st;
  }
  return pai_sched_execute(graph, mem_plan, plan, region, out_stats);
}

/* ------------------------------------------------------------------ */
/* Session registry                                                    */
/* ------------------------------------------------------------------ */

void
pai_sched_sessions_init(pai_sched_sessions_t *sessions) {
  memset(sessions, 0, sizeof(*sessions));
  sessions->next_id = 1;
}

pai_status_t
pai_sched_session_create(pai_sched_sessions_t *sessions, int32_t priority,
                         uint8_t policy, uint64_t *out_id) {
  if (sessions == NULL || out_id == NULL || policy > PAI_SCHED_BALANCED) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0; i < PAI_SCHED_MAX_SESSIONS; i++) {
    if (!sessions->slots[i].active) {
      sessions->slots[i].id = sessions->next_id++;
      sessions->slots[i].priority = priority;
      sessions->slots[i].policy = policy;
      sessions->slots[i].active = 1;
      sessions->count++;
      *out_id = sessions->slots[i].id;
      return PAI_OK;
    }
  }
  return PAI_ERR_NOMEM;
}

pai_status_t
pai_sched_session_destroy(pai_sched_sessions_t *sessions, uint64_t id) {
  if (sessions == NULL || id == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint32_t i = 0; i < PAI_SCHED_MAX_SESSIONS; i++) {
    if (sessions->slots[i].active && sessions->slots[i].id == id) {
      sessions->slots[i].active = 0;
      sessions->count--;
      return PAI_OK;
    }
  }
  return PAI_ERR_MISMATCH;
}

int
pai_sched_session_active(const pai_sched_sessions_t *sessions, uint64_t id) {
  if (sessions == NULL || id == 0) {
    return 0;
  }
  for (uint32_t i = 0; i < PAI_SCHED_MAX_SESSIONS; i++) {
    if (sessions->slots[i].active && sessions->slots[i].id == id) {
      return 1;
    }
  }
  return 0;
}

int
pai_sched_pick_session(const pai_sched_sessions_t *sessions) {
  int best = -1;
  if (sessions == NULL) {
    return -1;
  }
  for (uint32_t i = 0; i < PAI_SCHED_MAX_SESSIONS; i++) {
    if (!sessions->slots[i].active) {
      continue;
    }
    if (best < 0 ||
        sessions->slots[i].priority > sessions->slots[best].priority) {
      best = (int)i;
    }
  }
  return best;
}
