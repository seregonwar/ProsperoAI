/*
 * ProsperoAI — graph layer
 *
 * Generic compute graph (whitepaper §10/§11): values (tensor descriptors)
 * and ops with data dependencies. The graph provides topological ordering,
 * tensor lifetime analysis and an integrated static memory plan — the
 * §11 "memory planning" step that reuses buffers for values whose
 * lifetimes do not overlap (§16).
 */

#ifndef PAI_GRAPH_H
#define PAI_GRAPH_H

#include <pai/error.h>
#include <pai/tensor.h>
#include <planner.h>

#include <stdint.h>

#define PAI_GRAPH_MAX_VALUES 1024u /* 32-layer llama fits (~999 values) */
#define PAI_GRAPH_MAX_OPS    1024u
#define PAI_GRAPH_MAX_ARITY  16u

typedef uint32_t pai_graph_value_id; /* 0 = invalid */
typedef uint32_t pai_graph_op_id;    /* 0 = invalid */

typedef enum pai_graph_op_kind {
  PAI_OP_NONE = 0,
  PAI_OP_ADD,
  PAI_OP_MUL,
  PAI_OP_GEMM,
  PAI_OP_GEMV,
  PAI_OP_MATMUL,
  PAI_OP_RELU,
  PAI_OP_SOFTMAX,
  PAI_OP_LAYERNORM,
  PAI_OP_RMSNORM,
  PAI_OP_ROPE,
  PAI_OP_ATTENTION,
  PAI_OP_RESHAPE,
  PAI_OP_CONCAT,
  PAI_OP_CONVERT,
  PAI_OP_COPY,
  PAI_OP_SILU,
  PAI_OP_CUSTOM
} pai_graph_op_kind_t;

typedef struct pai_graph_value {
  pai_dtype_t dtype;
  uint32_t    rank;
  uint64_t    shape[PAI_TENSOR_MAX_RANK];
  uint64_t    size_bytes; /* contiguous payload bytes        */
  uint64_t    align;      /* storage alignment               */
  uint32_t    producer;   /* producing op id, 0 for inputs   */
  uint32_t    is_input;
  uint32_t    is_output;
} pai_graph_value_t;

typedef struct pai_graph_op {
  pai_graph_op_kind_t kind;
  uint32_t num_inputs;
  uint32_t num_outputs;
  pai_graph_value_id inputs[PAI_GRAPH_MAX_ARITY];
  pai_graph_value_id outputs[PAI_GRAPH_MAX_ARITY];
  uint32_t epoch; /* topological position, set by pai_graph_topo_sort */
} pai_graph_op_t;

typedef struct pai_graph {
  pai_graph_value_t values[PAI_GRAPH_MAX_VALUES + 1]; /* index = value id */
  pai_graph_op_t ops[PAI_GRAPH_MAX_OPS + 1];          /* index = op id    */
  uint32_t num_values;
  uint32_t num_ops;
  uint32_t sorted;
} pai_graph_t;

void pai_graph_init(pai_graph_t *graph);

/* Returns a value id, or 0 on failure. align 0 selects the default (16). */
pai_graph_value_id pai_graph_add_value(pai_graph_t *graph, pai_dtype_t dtype,
                                       uint32_t rank, const uint64_t *shape,
                                       uint64_t align);

/*
 * Add an op consuming/producing existing values. Output values must not
 * have a producer yet. Returns an op id, or 0 on failure.
 */
pai_graph_op_id pai_graph_add_op(pai_graph_t *graph, pai_graph_op_kind_t kind,
                                 uint32_t num_inputs, uint32_t num_outputs,
                                 const pai_graph_value_id *inputs,
                                 const pai_graph_value_id *outputs);

void pai_graph_set_input(pai_graph_t *graph, pai_graph_value_id value);
void pai_graph_set_output(pai_graph_t *graph, pai_graph_value_id value);

/*
 * Assign topological epochs to all ops (Kahn). Returns PAI_ERR_MISMATCH
 * when the graph contains a cycle.
 */
pai_status_t pai_graph_topo_sort(pai_graph_t *graph);

const char *pai_graph_op_kind_str(pai_graph_op_kind_t kind);

typedef struct pai_graph_mem_plan {
  /* Planner output; base.offsets is transient (released by
   * pai_graph_memory_plan) — use pai_graph_mem_plan_offset instead.
   * The scalar stats (region/naive/reuse/peak) stay valid. */
  pai_mem_plan_t base;
  uint64_t offsets[PAI_GRAPH_MAX_VALUES + 1];     /* per value id         */
  uint64_t region_bytes;                          /* planned region       */
  uint64_t naive_bytes;                           /* no-reuse sum         */
  uint64_t peak_live;                             /* max live bytes       */
} pai_graph_mem_plan_t;

/*
 * Compute per-value lifetimes `[start, end)` in topo epochs, using the
 * same semantics as the memory planner: inputs/params start at 0,
 * produced values start at their producer's epoch, and a value ends
 * after its last consumer (or at execution end for outputs).
 * `out_start`/`out_end` must hold PAI_GRAPH_MAX_VALUES + 1 entries.
 * Runs pai_graph_topo_sort first if needed.
 */
pai_status_t pai_graph_value_lifetimes(pai_graph_t *graph,
                                       uint32_t *out_start,
                                       uint32_t *out_end);

/*
 * Persistent variant matching pai_graph_memory_plan_persistent:
 * producer-less values (inputs/params) live until execution end.
 * Consumers that plan sessions (the scheduler's per-step accounting)
 * should use this so their live-byte stats agree with the persistent
 * layout the executor runs against.
 */
pai_status_t pai_graph_value_lifetimes_persistent(pai_graph_t *graph,
                                                  uint32_t *out_start,
                                                  uint32_t *out_end);

/*
 * Compute the static memory plan for the graph: extracts per-value
 * lifetimes from the topo order, then reuses storage across
 * non-overlapping values. Runs pai_graph_topo_sort first if needed.
 */
pai_status_t pai_graph_memory_plan(pai_graph_t *graph,
                                   pai_graph_mem_plan_t *plan);

/*
 * Persistent variant for session execution: inputs and params (values
 * with no producer) are kept live for the whole execution, so the
 * session can re-run the plan repeatedly (e.g. one generation step per
 * call) without weights being overwritten by later activations. Only
 * the producer-less values change: activations still reuse storage.
 */
pai_status_t pai_graph_memory_plan_persistent(pai_graph_t *graph,
                                              pai_graph_mem_plan_t *plan);

/* Storage offset of a value inside the planned region (0 for invalid). */
uint64_t pai_graph_mem_plan_offset(const pai_graph_mem_plan_t *plan,
                                   pai_graph_value_id value);

void pai_graph_mem_plan_free(pai_graph_mem_plan_t *plan);

#endif /* PAI_GRAPH_H */
