/*
 * ProsperoAI — Kernel IR (§10.2)
 *
 * Kernel descriptors close to GPU execution (tiling, vector width,
 * layout, workgroups). The lowering here is deliberately naive (one
 * kernel per IR op, no fusion); fusion passes emit richer kernels
 * through the same API.
 */

#include "ir.h"

#include <stdio.h>
#include <string.h>

void
pai_kir_init(pai_kir_kernel_t *kernel, const char *name) {
  memset(kernel, 0, sizeof(*kernel));
  if (name != NULL) {
    size_t n = strlen(name);
    if (n >= PAI_KIR_NAME_MAX) {
      n = PAI_KIR_NAME_MAX - 1;
    }
    memcpy(kernel->name, name, n);
    kernel->name[n] = '\0';
  }
  kernel->workgroup.x = 64; /* one default wavefront-sized workgroup */
  kernel->workgroup.y = 1;
  kernel->workgroup.z = 1;
  kernel->tile.x = 1;
  kernel->tile.y = 1;
  kernel->tile.z = 1;
  kernel->vector_width = 1;
}

pai_status_t
pai_kir_add_op(pai_kir_kernel_t *kernel, uint8_t kind, uint32_t num_inputs,
               uint32_t num_outputs, const uint16_t *in, const uint16_t *out,
               uint8_t vector_width, uint8_t mem_layout) {
  pai_kir_op_t *op;

  if (kernel == NULL || kind >= PAI_IR_OP_KIND_COUNT ||
      num_inputs > PAI_KIR_OP_MAX_IN || num_outputs > PAI_KIR_OP_MAX_OUT ||
      (num_inputs > 0 && in == NULL) || (num_outputs > 0 && out == NULL) ||
      vector_width == 0 || mem_layout > PAI_KIR_MEM_PACKED ||
      kernel->num_ops >= PAI_KIR_MAX_OPS) {
    return PAI_ERR_INVALID_ARG;
  }

  op = &kernel->ops[kernel->num_ops++];
  memset(op, 0, sizeof(*op));
  op->kind = kind;
  op->num_inputs = (uint8_t)num_inputs;
  op->num_outputs = (uint8_t)num_outputs;
  op->vector_width = vector_width;
  op->mem_layout = mem_layout;
  memcpy(op->in, in, sizeof(uint16_t) * num_inputs);
  memcpy(op->out, out, sizeof(uint16_t) * num_outputs);
  return PAI_OK;
}

pai_status_t
pai_kir_set_workgroup(pai_kir_kernel_t *kernel, uint32_t x, uint32_t y,
                      uint32_t z) {
  if (kernel == NULL || x == 0 || y == 0 || z == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  kernel->workgroup.x = x;
  kernel->workgroup.y = y;
  kernel->workgroup.z = z;
  return PAI_OK;
}

pai_status_t
pai_kir_set_tile(pai_kir_kernel_t *kernel, uint32_t x, uint32_t y,
                 uint32_t z) {
  if (kernel == NULL || x == 0 || y == 0 || z == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  kernel->tile.x = x;
  kernel->tile.y = y;
  kernel->tile.z = z;
  return PAI_OK;
}

pai_status_t
pai_kir_lower_program(const pai_ir_program_t *prog, pai_kir_kernel_t *kernels,
                      uint32_t max_kernels, uint32_t *out_num_kernels) {
  uint32_t count = 0;

  if (prog == NULL || kernels == NULL || out_num_kernels == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_num_kernels = 0;

  for (uint32_t o = 1; o <= prog->num_ops; o++) {
    const pai_ir_op_t *ir_op = &prog->ops[o];
    pai_kir_kernel_t *k;
    char name[PAI_KIR_NAME_MAX];
    uint32_t vec = 4;
    pai_status_t st;

    if (count >= max_kernels) {
      return PAI_ERR_NOMEM;
    }
    if (ir_op->num_inputs > PAI_KIR_OP_MAX_IN ||
        ir_op->num_outputs > PAI_KIR_OP_MAX_OUT) {
      return PAI_ERR_UNSUPPORTED;
    }

    /* Vector width: 4 when every involved buffer holds a multiple of
     * four elements (a whole float4 per lane); otherwise scalar. */
    for (uint32_t i = 0; i < ir_op->num_inputs; i++) {
      const pai_ir_value_t *v = &prog->values[ir_op->inputs[i]];
      uint64_t nelem = v->size_bytes / (uint64_t)pai_dtype_size(v->dtype);
      if (nelem % 4 != 0) {
        vec = 1;
      }
    }
    for (uint32_t i = 0; i < ir_op->num_outputs; i++) {
      const pai_ir_value_t *v = &prog->values[ir_op->outputs[i]];
      uint64_t nelem = v->size_bytes / (uint64_t)pai_dtype_size(v->dtype);
      if (nelem % 4 != 0) {
        vec = 1;
      }
    }

    snprintf(name, sizeof(name), "k%u_%s", o, pai_ir_op_kind_name(ir_op->kind));
    k = &kernels[count];
    pai_kir_init(k, name);
    k->vector_width = (uint8_t)vec;
    st = pai_kir_add_op(k, ir_op->kind, ir_op->num_inputs, ir_op->num_outputs,
                        ir_op->inputs, ir_op->outputs, (uint8_t)vec,
                        PAI_KIR_MEM_ROW_MAJOR);
    if (st != PAI_OK) {
      return st;
    }
    count++;
  }

  *out_num_kernels = count;
  return PAI_OK;
}
