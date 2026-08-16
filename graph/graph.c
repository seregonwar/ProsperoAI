#include "graph.h"

#include <string.h>

static const char *const k_op_names[] = {
    "none",   "add",     "mul",     "gemm",   "gemv",   "matmul", "relu",
    "softmax", "layernorm", "rmsnorm", "rope",  "attention", "reshape",
    "concat", "convert", "copy",    "silu",   "custom",
};

void
pai_graph_init(pai_graph_t *graph) {
  memset(graph, 0, sizeof(*graph));
}

pai_graph_value_id
pai_graph_add_value(pai_graph_t *graph, pai_dtype_t dtype, uint32_t rank,
                    const uint64_t *shape, uint64_t align) {
  pai_graph_value_t *value;
  pai_status_t st;
  uint64_t bytes;
  pai_graph_value_id id;

  if (!graph || !shape || rank == 0 || rank > PAI_TENSOR_MAX_RANK ||
      graph->num_values >= PAI_GRAPH_MAX_VALUES ||
      (align != 0 && (align & (align - 1)) != 0)) {
    return 0;
  }

  st = pai_tensor_contiguous_size(dtype, rank, shape, &bytes);
  if (st != PAI_OK) {
    return 0;
  }

  id = ++graph->num_values;
  value = &graph->values[id];
  memset(value, 0, sizeof(*value));
  value->dtype = dtype;
  value->rank = rank;
  value->size_bytes = bytes;
  value->align = align != 0 ? align : 16u;
  memcpy(value->shape, shape, sizeof(uint64_t) * rank);
  return id;
}

pai_graph_op_id
pai_graph_add_op(pai_graph_t *graph, pai_graph_op_kind_t kind,
                 uint32_t num_inputs, uint32_t num_outputs,
                 const pai_graph_value_id *inputs,
                 const pai_graph_value_id *outputs) {
  pai_graph_op_t *op;
  pai_graph_op_id id;

  if (!graph || num_inputs > PAI_GRAPH_MAX_ARITY ||
      num_outputs > PAI_GRAPH_MAX_ARITY ||
      (num_inputs > 0 && inputs == NULL) ||
      (num_outputs > 0 && outputs == NULL) ||
      graph->num_ops >= PAI_GRAPH_MAX_OPS) {
    return 0;
  }

  for (uint32_t i = 0; i < num_inputs; i++) {
    if (inputs[i] == 0 || inputs[i] > graph->num_values) {
      return 0;
    }
  }
  for (uint32_t i = 0; i < num_outputs; i++) {
    if (outputs[i] == 0 || outputs[i] > graph->num_values ||
        graph->values[outputs[i]].producer != 0) {
      return 0; /* value already produced: single-assignment */
    }
    for (uint32_t k = 0; k < i; k++) {
      if (outputs[k] == outputs[i]) {
        return 0; /* duplicate output id within one op */
      }
    }
  }
  /* An op may not consume its own output (that is a cycle). */
  for (uint32_t i = 0; i < num_inputs; i++) {
    for (uint32_t k = 0; k < num_outputs; k++) {
      if (inputs[i] == outputs[k]) {
        return 0;
      }
    }
  }

  id = ++graph->num_ops;
  op = &graph->ops[id];
  memset(op, 0, sizeof(*op));
  op->kind = kind;
  op->num_inputs = num_inputs;
  op->num_outputs = num_outputs;
  memcpy(op->inputs, inputs, sizeof(pai_graph_value_id) * num_inputs);
  memcpy(op->outputs, outputs, sizeof(pai_graph_value_id) * num_outputs);

  for (uint32_t i = 0; i < num_outputs; i++) {
    graph->values[outputs[i]].producer = id;
  }
  graph->sorted = 0;
  return id;
}

void
pai_graph_set_input(pai_graph_t *graph, pai_graph_value_id value) {
  if (graph != NULL && value != 0 && value <= graph->num_values) {
    graph->values[value].is_input = 1;
  }
}

void
pai_graph_set_output(pai_graph_t *graph, pai_graph_value_id value) {
  if (graph != NULL && value != 0 && value <= graph->num_values) {
    graph->values[value].is_output = 1;
  }
}

pai_status_t
pai_graph_topo_sort(pai_graph_t *graph) {
  uint32_t indegree[PAI_GRAPH_MAX_OPS + 1];
  uint32_t queue[PAI_GRAPH_MAX_OPS];
  uint32_t head = 0;
  uint32_t tail = 0;
  uint32_t processed = 0;
  pai_graph_op_id order[PAI_GRAPH_MAX_OPS];
  uint32_t count;

  if (!graph) {
    return PAI_ERR_INVALID_ARG;
  }

  count = graph->num_ops;
  memset(indegree, 0, sizeof(indegree));

  /* Edge producer op -> consumer op for every consumed value. */
  for (pai_graph_op_id o = 1; o <= graph->num_ops; o++) {
    const pai_graph_op_t *op = &graph->ops[o];
    for (uint32_t i = 0; i < op->num_inputs; i++) {
      uint32_t producer = graph->values[op->inputs[i]].producer;
      if (producer != 0 && producer != o) {
        indegree[o]++;
      }
    }
  }

  for (pai_graph_op_id o = 1; o <= graph->num_ops; o++) {
    if (indegree[o] == 0) {
      queue[tail++] = o;
    }
  }

  while (head < tail) {
    pai_graph_op_id o = queue[head++];
    order[processed++] = o;

    /* Consumers of o's outputs become available. */
    for (uint32_t i = 0; i < graph->ops[o].num_outputs; i++) {
      pai_graph_value_id v = graph->ops[o].outputs[i];
      for (pai_graph_op_id c = 1; c <= graph->num_ops; c++) {
        const pai_graph_op_t *op = &graph->ops[c];
        if (c == o) {
          continue;
        }
        for (uint32_t j = 0; j < op->num_inputs; j++) {
          if (op->inputs[j] == v && --indegree[c] == 0) {
            queue[tail++] = c;
          }
        }
      }
    }
  }

  if (processed != count) {
    graph->sorted = 0;
    return PAI_ERR_MISMATCH;
  }

  for (uint32_t i = 0; i < count; i++) {
    graph->ops[order[i]].epoch = i;
  }
  graph->sorted = 1;
  return PAI_OK;
}

const char *
pai_graph_op_kind_str(pai_graph_op_kind_t kind) {
  if ((int)kind < 0 || (int)kind >= (int)(sizeof(k_op_names) / sizeof(k_op_names[0]))) {
    return "?";
  }
  return k_op_names[kind];
}

pai_status_t
pai_graph_value_lifetimes_persistent(pai_graph_t *graph,
                                        uint32_t *out_start,
                                        uint32_t *out_end) {
  pai_status_t st;

  st = pai_graph_value_lifetimes(graph, out_start, out_end);
  if (st != PAI_OK) {
    return st;
  }
  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    if (graph->values[v].producer == 0) {
      out_end[v] = graph->num_ops;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_graph_value_lifetimes(pai_graph_t *graph, uint32_t *out_start,
                          uint32_t *out_end) {
  uint32_t last_use[PAI_GRAPH_MAX_VALUES + 1];
  uint32_t used[PAI_GRAPH_MAX_VALUES + 1];
  pai_status_t st = PAI_OK;

  if (graph == NULL || out_start == NULL || out_end == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out_start, 0, sizeof(uint32_t) * (PAI_GRAPH_MAX_VALUES + 1));
  memset(out_end, 0, sizeof(uint32_t) * (PAI_GRAPH_MAX_VALUES + 1));

  if (!graph->sorted) {
    st = pai_graph_topo_sort(graph);
    if (st != PAI_OK) {
      return st;
    }
  }

  /* Last consuming epoch per value. */
  memset(last_use, 0, sizeof(last_use));
  memset(used, 0, sizeof(used));
  for (pai_graph_op_id o = 1; o <= graph->num_ops; o++) {
    const pai_graph_op_t *op = &graph->ops[o];
    for (uint32_t i = 0; i < op->num_inputs; i++) {
      pai_graph_value_id v = op->inputs[i];
      used[v] = 1;
      if (op->epoch > last_use[v]) {
        last_use[v] = op->epoch;
      }
    }
  }

  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    const pai_graph_value_t *value = &graph->values[v];
    uint32_t start;
    uint32_t end;

    start = value->producer != 0 ? graph->ops[value->producer].epoch : 0;
    if (value->is_output) {
      end = graph->num_ops; /* live until execution completes */
    } else if (used[v]) {
      end = last_use[v] + 1;
    } else {
      end = start + 1; /* produced, never consumed: live for its own step */
    }

    out_start[v] = start;
    out_end[v] = end;
  }
  return st;
}

static pai_status_t
graph_memory_plan_impl(pai_graph_t *graph, pai_graph_mem_plan_t *out,
                      int persistent) {
  pai_mem_planner_t planner;
  uint32_t start[PAI_GRAPH_MAX_VALUES + 1];
  uint32_t end[PAI_GRAPH_MAX_VALUES + 1];
  pai_status_t st;

  if (!graph || !out) {
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

  pai_mem_planner_init(&planner);
  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    const pai_graph_value_t *value = &graph->values[v];
    st = pai_mem_planner_add(&planner, value->size_bytes, value->align,
                             start[v], end[v]);
    if (st != PAI_OK) {
      return st;
    }
  }

  st = pai_mem_plan_build(&planner, &out->base);
  if (st != PAI_OK) {
    return st;
  }

  /* Copy planner offsets (added in value-id order) into the id-indexed map. */
  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    out->offsets[v] = out->base.offsets[v - 1];
  }
  out->region_bytes = out->base.region_bytes;
  out->naive_bytes = out->base.naive_bytes;
  out->peak_live = out->base.peak_live;

  /* The per-id copy is authoritative; drop the transient array. */
  pai_mem_plan_free(&out->base);
  return PAI_OK;
}

pai_status_t
pai_graph_memory_plan(pai_graph_t *graph, pai_graph_mem_plan_t *out) {
  return graph_memory_plan_impl(graph, out, 0);
}

pai_status_t
pai_graph_memory_plan_persistent(pai_graph_t *graph,
                                 pai_graph_mem_plan_t *out) {
  return graph_memory_plan_impl(graph, out, 1);
}

uint64_t
pai_graph_mem_plan_offset(const pai_graph_mem_plan_t *plan,
                          pai_graph_value_id value) {
  if (plan == NULL || value == 0 || value > PAI_GRAPH_MAX_VALUES) {
    return 0;
  }
  return plan->offsets[value];
}

void
pai_graph_mem_plan_free(pai_graph_mem_plan_t *plan) {
  if (plan != NULL) {
    pai_mem_plan_free(&plan->base); /* no-op: offsets already released */
  }
}
