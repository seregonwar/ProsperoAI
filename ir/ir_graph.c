/*
 * ProsperoAI — graph <-> IR bridge
 *
 * Converts the higher-level pai_graph (graph layer, whitepaper §10/§11)
 * into the standalone Prospero IR representation and back. The op-kind
 * sets are kept in sync by the table below; the mapping is identity by
 * construction and a test asserts completeness.
 */

#include "ir.h"

#include <graph/graph.h>

#include <string.h>

/* Identity mapping between the graph op set and the IR op set. */
static const uint8_t k_graph_to_ir[] = {
    [PAI_OP_NONE] = PAI_IR_OP_NONE,
    [PAI_OP_ADD] = PAI_IR_OP_ADD,
    [PAI_OP_MUL] = PAI_IR_OP_MUL,
    [PAI_OP_GEMM] = PAI_IR_OP_GEMM,
    [PAI_OP_GEMV] = PAI_IR_OP_GEMV,
    [PAI_OP_MATMUL] = PAI_IR_OP_MATMUL,
    [PAI_OP_RELU] = PAI_IR_OP_RELU,
    [PAI_OP_SOFTMAX] = PAI_IR_OP_SOFTMAX,
    [PAI_OP_LAYERNORM] = PAI_IR_OP_LAYERNORM,
    [PAI_OP_RMSNORM] = PAI_IR_OP_RMSNORM,
    [PAI_OP_ROPE] = PAI_IR_OP_ROPE,
    [PAI_OP_ATTENTION] = PAI_IR_OP_ATTENTION,
    [PAI_OP_RESHAPE] = PAI_IR_OP_RESHAPE,
    [PAI_OP_CONCAT] = PAI_IR_OP_CONCAT,
    [PAI_OP_CONVERT] = PAI_IR_OP_CONVERT,
    [PAI_OP_COPY] = PAI_IR_OP_COPY,
    [PAI_OP_SILU] = PAI_IR_OP_SILU,
    [PAI_OP_CUSTOM] = PAI_IR_OP_CUSTOM,
};

static const uint8_t k_ir_to_graph[] = {
    [PAI_IR_OP_NONE] = PAI_OP_NONE,
    [PAI_IR_OP_ADD] = PAI_OP_ADD,
    [PAI_IR_OP_MUL] = PAI_OP_MUL,
    [PAI_IR_OP_GEMM] = PAI_OP_GEMM,
    [PAI_IR_OP_GEMV] = PAI_OP_GEMV,
    [PAI_IR_OP_MATMUL] = PAI_OP_MATMUL,
    [PAI_IR_OP_RELU] = PAI_OP_RELU,
    [PAI_IR_OP_SOFTMAX] = PAI_OP_SOFTMAX,
    [PAI_IR_OP_LAYERNORM] = PAI_OP_LAYERNORM,
    [PAI_IR_OP_RMSNORM] = PAI_OP_RMSNORM,
    [PAI_IR_OP_ROPE] = PAI_OP_ROPE,
    [PAI_IR_OP_ATTENTION] = PAI_OP_ATTENTION,
    [PAI_IR_OP_RESHAPE] = PAI_OP_RESHAPE,
    [PAI_IR_OP_CONCAT] = PAI_OP_CONCAT,
    [PAI_IR_OP_CONVERT] = PAI_OP_CONVERT,
    [PAI_IR_OP_COPY] = PAI_OP_COPY,
    [PAI_IR_OP_SILU] = PAI_OP_SILU,
    [PAI_IR_OP_CUSTOM] = PAI_OP_CUSTOM,
};

pai_status_t
pai_ir_from_graph(const pai_graph_t *graph, pai_ir_program_t *out) {
  uint32_t value_map[PAI_GRAPH_MAX_VALUES + 1];

  if (graph == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  pai_ir_init(out);

  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    const pai_graph_value_t *gv = &graph->values[v];
    uint8_t kind;
    uint32_t ir_id;

    if (gv->is_input) {
      kind = PAI_IR_VALUE_INPUT;
    } else if (gv->is_output) {
      kind = PAI_IR_VALUE_OUTPUT;
    } else if (gv->producer == 0) {
      kind = PAI_IR_VALUE_PARAM;
    } else {
      kind = PAI_IR_VALUE_ACTIVATION;
    }

    ir_id = pai_ir_add_value(out, gv->dtype, gv->rank, gv->shape, gv->align,
                             kind, PAI_IR_DEVICE_ANY);
    if (ir_id == 0) {
      return PAI_ERR_NOMEM;
    }
    value_map[v] = ir_id;
  }

  for (pai_graph_op_id o = 1; o <= graph->num_ops; o++) {
    const pai_graph_op_t *go = &graph->ops[o];
    uint16_t inputs[PAI_IR_MAX_ARITY];
    uint16_t outputs[PAI_IR_MAX_ARITY];
    uint32_t ir_id;

    if ((int)go->kind < 0 || go->kind >= PAI_OP_CUSTOM + 1) {
      return PAI_ERR_MISMATCH;
    }
    for (uint32_t i = 0; i < go->num_inputs; i++) {
      inputs[i] = (uint16_t)value_map[go->inputs[i]];
    }
    for (uint32_t i = 0; i < go->num_outputs; i++) {
      outputs[i] = (uint16_t)value_map[go->outputs[i]];
    }
    ir_id = pai_ir_add_op(out, k_graph_to_ir[go->kind], PAI_IR_DEVICE_ANY,
                          go->num_inputs, go->num_outputs, inputs, outputs);
    if (ir_id == 0) {
      return PAI_ERR_NOMEM;
    }
  }

  for (pai_graph_value_id v = 1; v <= graph->num_values; v++) {
    if (graph->values[v].is_input) {
      pai_ir_set_input(out, value_map[v]);
    }
    if (graph->values[v].is_output) {
      pai_ir_set_output(out, value_map[v]);
    }
  }
  return PAI_OK;
}

pai_status_t
pai_ir_to_graph(const pai_ir_program_t *ir, pai_graph_t *out) {
  uint32_t value_map[PAI_IR_MAX_VALUES + 1];

  if (ir == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  pai_graph_init(out);

  for (uint32_t v = 1; v <= ir->num_values; v++) {
    const pai_ir_value_t *iv = &ir->values[v];
    pai_graph_value_id id;

    /* The graph layer is unquantized (§15 metadata lives in the IR,
     * which the model manager keeps for the loader); a quantized
     * value is rebuilt as its logical f32 tensor. */
    id = pai_graph_add_value(out, iv->dtype, iv->rank, iv->shape, iv->align);
    if (id == 0) {
      return PAI_ERR_NOMEM;
    }
    value_map[v] = id;
  }

  for (uint32_t o = 1; o <= ir->num_ops; o++) {
    const pai_ir_op_t *io = &ir->ops[o];
    pai_graph_value_id inputs[PAI_IR_MAX_ARITY];
    pai_graph_value_id outputs[PAI_IR_MAX_ARITY];
    pai_graph_op_id id;

    if (io->kind >= PAI_IR_OP_KIND_COUNT) {
      return PAI_ERR_MISMATCH;
    }
    for (uint32_t i = 0; i < io->num_inputs; i++) {
      inputs[i] = value_map[io->inputs[i]];
    }
    for (uint32_t i = 0; i < io->num_outputs; i++) {
      outputs[i] = value_map[io->outputs[i]];
    }
    id = pai_graph_add_op(out, (pai_graph_op_kind_t)k_ir_to_graph[io->kind],
                          io->num_inputs, io->num_outputs, inputs, outputs);
    if (id == 0) {
      return PAI_ERR_MISMATCH;
    }
  }

  for (uint32_t i = 0; i < ir->num_inputs; i++) {
    pai_graph_set_input(out, value_map[ir->input_ids[i]]);
  }
  for (uint32_t i = 0; i < ir->num_outputs; i++) {
    pai_graph_set_output(out, value_map[ir->output_ids[i]]);
  }
  return PAI_OK;
}
