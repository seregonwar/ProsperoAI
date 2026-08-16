#include "test.h"

#include <ir/ir.h>
#include <graph/graph.h>

#include <string.h>

/*
 * Synthetic network (the Phase 1 target graph):
 *   x[16] -> GEMM(w1[16,16]) -> h1 -> ADD(b[16]) -> h2 -> RELU -> h3
 *         -> GEMM(w2[16,16]) -> h4 -> SOFTMAX -> y[16]  (y is output)
 */
static pai_graph_value_id
build_net(pai_graph_t *graph) {
  pai_graph_value_id x;
  pai_graph_value_id w1;
  pai_graph_value_id h1;
  pai_graph_value_id b;
  pai_graph_value_id h2;
  pai_graph_value_id h3;
  pai_graph_value_id w2;
  pai_graph_value_id h4;
  pai_graph_value_id y;
  const uint64_t v16[1] = {16};
  const uint64_t m16[2] = {16, 16};

  pai_graph_init(graph);

  x = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  w1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m16, 0);
  h1 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  b = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  h2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  h3 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  w2 = pai_graph_add_value(graph, PAI_DTYPE_F32, 2, m16, 0);
  h4 = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);
  y = pai_graph_add_value(graph, PAI_DTYPE_F32, 1, v16, 0);

  {
    pai_graph_value_id in1[] = {x, w1};
    pai_graph_value_id out1[] = {h1};
    pai_graph_value_id in2[] = {h1, b};
    pai_graph_value_id out2[] = {h2};
    pai_graph_value_id in3[] = {h2};
    pai_graph_value_id out3[] = {h3};
    pai_graph_value_id in4[] = {h3, w2};
    pai_graph_value_id out4[] = {h4};
    pai_graph_value_id in5[] = {h4};
    pai_graph_value_id out5[] = {y};
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, in1, out1) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_ADD, 2, 1, in2, out2) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_RELU, 1, 1, in3, out3) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_GEMM, 2, 1, in4, out4) != 0);
    CHECK(pai_graph_add_op(graph, PAI_OP_SOFTMAX, 1, 1, in5, out5) != 0);
  }

  pai_graph_set_input(graph, x);
  pai_graph_set_output(graph, y);
  return x;
}

TEST_MAIN_BEGIN()

{
  /* graph -> IR mapping: counts, kinds, value categories */
  static pai_graph_t graph;
  static pai_ir_program_t ir;
  build_net(&graph);

  CHECK(pai_ir_from_graph(&graph, &ir) == PAI_OK);
  CHECK_EQ_UINT(ir.num_values, 9);
  CHECK_EQ_UINT(ir.num_ops, 5);
  CHECK_EQ_UINT(ir.num_inputs, 1);
  CHECK_EQ_UINT(ir.num_outputs, 1);

  /* value categories */
  CHECK_EQ_UINT(ir.values[1].kind, PAI_IR_VALUE_INPUT);   /* x  */
  CHECK_EQ_UINT(ir.values[2].kind, PAI_IR_VALUE_PARAM);   /* w1 */
  CHECK_EQ_UINT(ir.values[4].kind, PAI_IR_VALUE_PARAM);   /* b  */
  CHECK_EQ_UINT(ir.values[9].kind, PAI_IR_VALUE_OUTPUT);  /* y  */
  CHECK_EQ_UINT(ir.values[3].kind, PAI_IR_VALUE_ACTIVATION);

  /* op kinds survive the mapping */
  CHECK_EQ_UINT(ir.ops[1].kind, PAI_IR_OP_GEMM);
  CHECK_EQ_UINT(ir.ops[3].kind, PAI_IR_OP_RELU);
  CHECK_EQ_UINT(ir.ops[5].kind, PAI_IR_OP_SOFTMAX);
  CHECK_EQ_UINT(ir.ops[1].num_inputs, 2);
  CHECK_EQ_UINT(ir.ops[5].num_inputs, 1);

  /* shapes and sizes are preserved */
  CHECK_EQ_UINT(ir.values[2].rank, 2);
  CHECK_EQ_UINT(ir.values[2].shape[0], 16);
  CHECK_EQ_UINT(ir.values[2].shape[1], 16);
  CHECK_EQ_UINT(ir.values[2].size_bytes, 16 * 16 * 4);
}

{
  /* encode -> decode -> to_graph round trip is faithful */
  static pai_graph_t graph;
  static pai_graph_t graph2;
  static pai_ir_program_t ir;
  static pai_ir_program_t ir2;
  pai_graph_mem_plan_t p1;
  pai_graph_mem_plan_t p2;
  static uint8_t blob[65536];
  uint32_t nbytes;
  build_net(&graph);

  CHECK(pai_ir_from_graph(&graph, &ir) == PAI_OK);
  CHECK(pai_ir_encode(&ir, blob, sizeof(blob), &nbytes) == PAI_OK);
  CHECK(nbytes == pai_ir_encoded_size(&ir));
  CHECK(nbytes > PAI_IR_HEADER_SIZE);

  CHECK(pai_ir_decode(blob, nbytes, &ir2) == PAI_OK);
  CHECK_EQ_UINT(ir2.num_values, 9);
  CHECK_EQ_UINT(ir2.num_ops, 5);
  CHECK_EQ_UINT(ir2.num_inputs, 1);
  CHECK_EQ_UINT(ir2.num_outputs, 1);
  CHECK_EQ_UINT(ir2.input_ids[0], 1);
  CHECK_EQ_UINT(ir2.output_ids[0], 9);

  CHECK(pai_ir_to_graph(&ir2, &graph2) == PAI_OK);
  CHECK_EQ_UINT(graph2.num_values, graph.num_values);
  CHECK_EQ_UINT(graph2.num_ops, graph.num_ops);
  for (pai_graph_value_id v = 1; v <= graph.num_values; v++) {
    CHECK_EQ_UINT(graph2.values[v].dtype, graph.values[v].dtype);
    CHECK_EQ_UINT(graph2.values[v].rank, graph.values[v].rank);
    CHECK_EQ_UINT(graph2.values[v].size_bytes, graph.values[v].size_bytes);
    CHECK_EQ_UINT(graph2.values[v].is_input, graph.values[v].is_input);
    CHECK_EQ_UINT(graph2.values[v].is_output, graph.values[v].is_output);
  }
  for (pai_graph_op_id o = 1; o <= graph.num_ops; o++) {
    CHECK_EQ_UINT(graph2.ops[o].kind, graph.ops[o].kind);
    CHECK_EQ_UINT(graph2.ops[o].num_inputs, graph.ops[o].num_inputs);
    CHECK_EQ_UINT(graph2.ops[o].num_outputs, graph.ops[o].num_outputs);
    for (uint32_t i = 0; i < graph.ops[o].num_inputs; i++) {
      CHECK_EQ_UINT(graph2.ops[o].inputs[i], graph.ops[o].inputs[i]);
    }
    for (uint32_t i = 0; i < graph.ops[o].num_outputs; i++) {
      CHECK_EQ_UINT(graph2.ops[o].outputs[i], graph.ops[o].outputs[i]);
    }
  }

  /* identical memory plans before and after the round trip */
  CHECK(pai_graph_memory_plan(&graph, &p1) == PAI_OK);
  CHECK(pai_graph_memory_plan(&graph2, &p2) == PAI_OK);
  CHECK_EQ_UINT(p1.region_bytes, p2.region_bytes);
  CHECK_EQ_UINT(p1.naive_bytes, p2.naive_bytes);
  CHECK_EQ_UINT(p1.peak_live, p2.peak_live);
}

{
  /* corruption and malformed blobs are rejected */
  static pai_graph_t graph;
  static pai_ir_program_t ir;
  static uint8_t blob[65536];
  static uint8_t bad[65536];
  uint32_t nbytes;
  pai_status_t st;
  build_net(&graph);
  CHECK(pai_ir_from_graph(&graph, &ir) == PAI_OK);
  CHECK(pai_ir_encode(&ir, blob, sizeof(blob), &nbytes) == PAI_OK);

  /* flip one payload byte -> CRC mismatch */
  memcpy(bad, blob, nbytes);
  bad[PAI_IR_HEADER_SIZE + 40] ^= 0xFF;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  /* flip the CRC field itself */
  memcpy(bad, blob, nbytes);
  bad[24] ^= 0x01;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  /* truncation */
  CHECK(pai_ir_decode(blob, nbytes - 1, &ir) != PAI_OK);
  /* magic / version */
  memcpy(bad, blob, nbytes);
  bad[0] ^= 0xFF;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  memcpy(bad, blob, nbytes);
  bad[5] ^= 0xFF;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  /* nonzero flags / reserved fields are rejected in v1 */
  memcpy(bad, blob, nbytes);
  bad[6] = 0x01;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  memcpy(bad, blob, nbytes);
  bad[28] = 0x01;
  CHECK(pai_ir_decode(bad, nbytes, &ir) == PAI_ERR_PROTOCOL);
  /* too small for a header */
  st = pai_ir_decode(blob, 8, &ir);
  CHECK(st == PAI_ERR_INVALID_ARG);
  /* encode into a too-small buffer */
  CHECK(pai_ir_encode(&ir, blob, 64, &nbytes) == PAI_ERR_INVALID_ARG);
}

{
  /* quantization metadata survives a round trip (§15) */
  static pai_graph_t graph;
  static pai_ir_program_t ir;
  static pai_ir_program_t ir2;
  static uint8_t blob[65536];
  uint32_t nbytes;
  pai_ir_quant_t q;
  build_net(&graph);
  CHECK(pai_ir_from_graph(&graph, &ir) == PAI_OK);

  memset(&q, 0, sizeof(q));
  q.present = 1;
  q.bit_width = 8;
  q.group_size = 32;
  q.scale_repr = PAI_IR_QUANT_REPR_F32;
  q.zero_point_repr = PAI_IR_QUANT_REPR_U8;
  q.is_signed = 0;
  q.block_structure = PAI_IR_QUANT_BLOCK_GROUP;
  CHECK(pai_ir_value_set_quant(&ir, 2, &q) == PAI_OK);

  CHECK(pai_ir_encode(&ir, blob, sizeof(blob), &nbytes) == PAI_OK);
  CHECK(pai_ir_decode(blob, nbytes, &ir2) == PAI_OK);
  CHECK_EQ_UINT(ir2.values[2].quant.present, 1);
  CHECK_EQ_UINT(ir2.values[2].quant.bit_width, 8);
  CHECK_EQ_UINT(ir2.values[2].quant.group_size, 32);
  CHECK_EQ_UINT(ir2.values[2].quant.block_structure, PAI_IR_QUANT_BLOCK_GROUP);
  CHECK_EQ_UINT(ir2.values[9].quant.present, 0);

  /* invalid quant rejected */
  q.present = 1;
  q.bit_width = 0;
  CHECK(pai_ir_value_set_quant(&ir, 2, &q) == PAI_ERR_INVALID_ARG);
  q.bit_width = 8;
  q.block_structure = 99;
  CHECK(pai_ir_value_set_quant(&ir, 2, &q) == PAI_ERR_INVALID_ARG);
  /* per-group quant without a group size is rejected */
  q.block_structure = PAI_IR_QUANT_BLOCK_GROUP;
  q.group_size = 0;
  CHECK(pai_ir_value_set_quant(&ir, 2, &q) == PAI_ERR_INVALID_ARG);
  q.group_size = 32;
  CHECK(pai_ir_value_set_quant(&ir, 2, &q) == PAI_OK);
}

{
  /* empty program round trip is valid */
  pai_ir_program_t a;
  pai_ir_program_t b;
  static uint8_t blob[256];
  uint32_t nbytes;
  pai_ir_init(&a);
  CHECK(pai_ir_encode(&a, blob, sizeof(blob), &nbytes) == PAI_OK);
  CHECK_EQ_UINT(nbytes, PAI_IR_HEADER_SIZE);
  CHECK(pai_ir_decode(blob, nbytes, &b) == PAI_OK);
  CHECK_EQ_UINT(b.num_values, 0);
  CHECK_EQ_UINT(b.num_ops, 0);
}

{
  /* builder validation */
  static pai_ir_program_t ir;
  const uint64_t v4[1] = {4};
  uint16_t a_in[] = {1};
  uint16_t a_out[] = {2};
  uint16_t loop_in[] = {2};
  uint16_t dup_out[] = {3, 3};
  pai_ir_init(&ir);
  CHECK(pai_ir_add_value(&ir, PAI_DTYPE_F32, 1, v4, 0, PAI_IR_VALUE_INPUT,
                         PAI_IR_DEVICE_ANY) == 1);
  CHECK(pai_ir_add_value(&ir, PAI_DTYPE_F32, 1, v4, 0, PAI_IR_VALUE_ACTIVATION,
                         PAI_IR_DEVICE_ANY) == 2);
  CHECK(pai_ir_add_value(&ir, PAI_DTYPE_F32, 1, v4, 0, PAI_IR_VALUE_ACTIVATION,
                         PAI_IR_DEVICE_ANY) == 3);
  CHECK(pai_ir_add_op(&ir, PAI_IR_OP_COPY, PAI_IR_DEVICE_ANY, 1, 1, a_in,
                      a_out) == 1);
  /* self-loop and duplicate outputs rejected */
  CHECK(pai_ir_add_op(&ir, PAI_IR_OP_COPY, PAI_IR_DEVICE_ANY, 1, 1, loop_in,
                      loop_in) == 0);
  CHECK(pai_ir_add_op(&ir, PAI_IR_OP_COPY, PAI_IR_DEVICE_ANY, 0, 2, NULL,
                      dup_out) == 0);
  /* out-of-range ids rejected */
  {
    uint16_t bad[] = {99};
    CHECK(pai_ir_add_op(&ir, PAI_IR_OP_COPY, PAI_IR_DEVICE_ANY, 1, 1, bad,
                        a_out) == 0);
  }
  /* invalid value args */
  CHECK(pai_ir_add_value(&ir, PAI_DTYPE_F32, 0, v4, 0, PAI_IR_VALUE_INPUT,
                         PAI_IR_DEVICE_ANY) == 0);
  CHECK(pai_ir_add_value(&ir, PAI_DTYPE_F32, 1, v4, 3, PAI_IR_VALUE_INPUT,
                         PAI_IR_DEVICE_ANY) == 0); /* non-pow2 align */
}

{
  /* kernel IR: naive per-op lowering (§10.2) */
  static pai_graph_t graph;
  static pai_ir_program_t ir;
  static pai_kir_kernel_t kernels[PAI_IR_MAX_OPS];
  uint32_t nk = 0;
  build_net(&graph);
  CHECK(pai_ir_from_graph(&graph, &ir) == PAI_OK);
  CHECK(pai_kir_lower_program(&ir, kernels, PAI_IR_MAX_OPS, &nk) == PAI_OK);
  CHECK_EQ_UINT(nk, 5);
  CHECK(strcmp(kernels[0].name, "k1_gemm") == 0);
  CHECK(strcmp(kernels[4].name, "k5_softmax") == 0);
  CHECK_EQ_UINT(kernels[0].num_ops, 1);
  CHECK_EQ_UINT(kernels[0].ops[0].kind, PAI_IR_OP_GEMM);
  CHECK_EQ_UINT(kernels[0].vector_width, 4);  /* 16-element buffers */
  CHECK_EQ_UINT(kernels[0].workgroup.x, 64);
  CHECK_EQ_UINT(kernels[0].fusion, PAI_KIR_FUSION_NONE);

  /* builder validation */
  {
    pai_kir_kernel_t k;
    uint16_t in1[] = {1};
    uint16_t out1[] = {2};
    pai_kir_init(&k, "test");
    CHECK(pai_kir_add_op(&k, PAI_IR_OP_GEMM, 1, 1, in1, out1, 4,
                         PAI_KIR_MEM_ROW_MAJOR) == PAI_OK);
    CHECK_EQ_UINT(k.ops[0].vector_width, 4);
    CHECK(pai_kir_set_workgroup(&k, 256, 1, 1) == PAI_OK);
    CHECK_EQ_UINT(k.workgroup.x, 256);
    CHECK(pai_kir_set_tile(&k, 0, 1, 1) == PAI_ERR_INVALID_ARG);
    CHECK(pai_kir_add_op(&k, PAI_IR_OP_GEMM, 1, 1, in1, out1, 0,
                         PAI_KIR_MEM_ROW_MAJOR) == PAI_ERR_INVALID_ARG);
    CHECK(pai_kir_lower_program(&ir, NULL, 1, &nk) == PAI_ERR_INVALID_ARG);
  }
}

TEST_MAIN_END()
