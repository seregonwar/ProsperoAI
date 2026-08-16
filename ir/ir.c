#include "ir.h"

#include <protocol/protocol.h> /* pai_proto_crc32 for blob integrity */

#include <string.h>

static const char *const k_op_names[] = {
    "none",   "add",     "mul",     "gemm",     "gemv",   "matmul",
    "relu",   "softmax", "layernorm", "rmsnorm", "rope",  "attention",
    "reshape", "concat", "convert", "copy",     "silu",   "custom",
};

static const char *const k_value_kind_names[] = {
    "activation", "input", "param", "constant", "output",
};

static const char *const k_device_names[] = {
    "any", "cpu", "gpu",
};

void
pai_ir_init(pai_ir_program_t *prog) {
  memset(prog, 0, sizeof(*prog));
}

uint32_t
pai_ir_add_value(pai_ir_program_t *prog, pai_dtype_t dtype, uint32_t rank,
                 const uint64_t *shape, uint64_t align, uint8_t kind,
                 uint8_t device) {
  pai_ir_value_t *value;
  pai_status_t st;
  uint64_t bytes;
  uint32_t id;

  if (prog == NULL || shape == NULL || rank == 0 || rank > PAI_TENSOR_MAX_RANK ||
      prog->num_values >= PAI_IR_MAX_VALUES || (int)dtype < 0 ||
      dtype >= PAI_DTYPE_COUNT || kind >= PAI_IR_VALUE_KIND_COUNT ||
      device >= PAI_IR_DEVICE_COUNT ||
      (align != 0 && (align & (align - 1)) != 0)) {
    return 0;
  }

  st = pai_tensor_contiguous_size(dtype, rank, shape, &bytes);
  if (st != PAI_OK || bytes == 0) {
    return 0;
  }

  id = ++prog->num_values;
  value = &prog->values[id];
  memset(value, 0, sizeof(*value));
  value->dtype = dtype;
  value->rank = (uint8_t)rank;
  value->kind = kind;
  value->device = device;
  value->size_bytes = bytes;
  value->align = align != 0 ? align : 16u;
  memcpy(value->shape, shape, sizeof(uint64_t) * rank);
  return id;
}

pai_status_t
pai_ir_value_set_quant(pai_ir_program_t *prog, uint32_t value_id,
                       const pai_ir_quant_t *quant) {
  if (prog == NULL || quant == NULL || value_id == 0 ||
      value_id > prog->num_values) {
    return PAI_ERR_INVALID_ARG;
  }
  if (quant->present) {
    if (quant->bit_width == 0 || quant->bit_width > 64 ||
        quant->scale_repr >= PAI_IR_QUANT_REPR_COUNT ||
        quant->zero_point_repr >= PAI_IR_QUANT_REPR_COUNT ||
        quant->block_structure >= PAI_IR_QUANT_BLOCK_COUNT ||
        (quant->block_structure == PAI_IR_QUANT_BLOCK_GROUP &&
         quant->group_size == 0)) {
      return PAI_ERR_INVALID_ARG;
    }
  }
  prog->values[value_id].quant = *quant;
  return PAI_OK;
}

uint32_t
pai_ir_add_op(pai_ir_program_t *prog, uint8_t kind, uint8_t device,
              uint32_t num_inputs, uint32_t num_outputs,
              const uint16_t *inputs, const uint16_t *outputs) {
  pai_ir_op_t *op;
  uint32_t id;

  if (prog == NULL || num_inputs > PAI_IR_MAX_ARITY ||
      num_outputs > PAI_IR_MAX_ARITY || (num_inputs > 0 && inputs == NULL) ||
      (num_outputs > 0 && outputs == NULL) || kind >= PAI_IR_OP_KIND_COUNT ||
      device >= PAI_IR_DEVICE_COUNT || prog->num_ops >= PAI_IR_MAX_OPS) {
    return 0;
  }

  for (uint32_t i = 0; i < num_inputs; i++) {
    if (inputs[i] == 0 || inputs[i] > prog->num_values) {
      return 0;
    }
  }
  for (uint32_t i = 0; i < num_outputs; i++) {
    if (outputs[i] == 0 || outputs[i] > prog->num_values) {
      return 0;
    }
    for (uint32_t k = 0; k < i; k++) {
      if (outputs[k] == outputs[i]) {
        return 0;
      }
    }
    for (uint32_t k = 0; k < num_inputs; k++) {
      if (outputs[i] == inputs[k]) {
        return 0; /* self-loop: an op consuming its own output */
      }
    }
  }

  id = ++prog->num_ops;
  op = &prog->ops[id];
  memset(op, 0, sizeof(*op));
  op->kind = kind;
  op->device = device;
  op->num_inputs = (uint8_t)num_inputs;
  op->num_outputs = (uint8_t)num_outputs;
  memcpy(op->inputs, inputs, sizeof(uint16_t) * num_inputs);
  memcpy(op->outputs, outputs, sizeof(uint16_t) * num_outputs);
  return id;
}

void
pai_ir_set_input(pai_ir_program_t *prog, uint32_t value_id) {
  if (prog == NULL || value_id == 0 || value_id > prog->num_values ||
      prog->num_inputs >= PAI_IR_MAX_VALUES) {
    return;
  }
  prog->input_ids[prog->num_inputs++] = value_id;
}

void
pai_ir_set_output(pai_ir_program_t *prog, uint32_t value_id) {
  if (prog == NULL || value_id == 0 || value_id > prog->num_values ||
      prog->num_outputs >= PAI_IR_MAX_VALUES) {
    return;
  }
  prog->output_ids[prog->num_outputs++] = value_id;
}

const char *
pai_ir_op_kind_name(uint8_t kind) {
  if (kind >= PAI_IR_OP_KIND_COUNT ||
      kind >= sizeof(k_op_names) / sizeof(k_op_names[0])) {
    return "?";
  }
  return k_op_names[kind];
}

const char *
pai_ir_value_kind_name(uint8_t kind) {
  if (kind >= PAI_IR_VALUE_KIND_COUNT ||
      kind >= sizeof(k_value_kind_names) / sizeof(k_value_kind_names[0])) {
    return "?";
  }
  return k_value_kind_names[kind];
}

const char *
pai_ir_device_name(uint8_t device) {
  if (device >= PAI_IR_DEVICE_COUNT ||
      device >= sizeof(k_device_names) / sizeof(k_device_names[0])) {
    return "?";
  }
  return k_device_names[device];
}

/* ------------------------------------------------------------------ */
/* Flat binary serialization                                           */
/* ------------------------------------------------------------------ */

static void
le_put_u16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)(v >> 8);
}

static void
le_put_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF);
  p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static void
le_put_u64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    p[i] = (uint8_t)(v >> (8 * i));
  }
}

static uint16_t
le_get_u16(const uint8_t *p) {
  return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t
le_get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static uint64_t
le_get_u64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) {
    v = (v << 8) | p[i];
  }
  return v;
}

/* Size of one op record. */
static uint32_t
op_record_size(const pai_ir_op_t *op) {
  return 8u + 2u * op->num_inputs + 2u * op->num_outputs;
}

uint32_t
pai_ir_encoded_size(const pai_ir_program_t *prog) {
  uint64_t total;
  uint32_t op_bytes = 0;

  if (prog == NULL) {
    return 0;
  }
  total = PAI_IR_HEADER_SIZE + (uint64_t)prog->num_values * PAI_IR_VALUE_REC_SIZE;
  for (uint32_t o = 1; o <= prog->num_ops; o++) {
    op_bytes += op_record_size(&prog->ops[o]);
  }
  total += op_bytes;
  total += (uint64_t)(prog->num_inputs + prog->num_outputs) * 4u;
  if (total > UINT32_MAX) {
    return 0;
  }
  return (uint32_t)total;
}

pai_status_t
pai_ir_encode(const pai_ir_program_t *prog, uint8_t *out, uint32_t cap,
              uint32_t *out_nbytes) {
  uint32_t total;
  uint8_t *body;
  uint32_t crc;

  if (prog == NULL || out == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  total = pai_ir_encoded_size(prog);
  if (total == 0 || total > cap) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(out, 0, PAI_IR_HEADER_SIZE);
  le_put_u32(out + 0, PAI_IR_MAGIC);
  le_put_u16(out + 4, PAI_IR_VERSION);
  le_put_u32(out + 8, prog->num_values);
  le_put_u32(out + 12, prog->num_ops);
  le_put_u32(out + 16, prog->num_inputs);
  le_put_u32(out + 20, prog->num_outputs);
  /* crc at +24 filled below; reserved at +28 stays 0 */

  body = out + PAI_IR_HEADER_SIZE;

  for (uint32_t v = 1; v <= prog->num_values; v++) {
    const pai_ir_value_t *val = &prog->values[v];
    uint8_t *r = body + (uint64_t)(v - 1) * PAI_IR_VALUE_REC_SIZE;
    r[0] = (uint8_t)val->dtype;
    r[1] = val->rank;
    r[2] = val->kind;
    r[3] = val->device;
    r[4] = val->quant.present;
    r[5] = val->quant.bit_width;
    r[6] = val->quant.scale_repr;
    r[7] = val->quant.zero_point_repr;
    r[8] = val->quant.is_signed;
    r[9] = val->quant.block_structure;
    le_put_u16(r + 10, val->quant.group_size);
    le_put_u64(r + 12, val->align);
    le_put_u64(r + 20, val->size_bytes);
    for (uint32_t d = 0; d < PAI_TENSOR_MAX_RANK; d++) {
      le_put_u64(r + 28 + 8u * d, val->shape[d]);
    }
  }

  {
    uint8_t *op = body + (uint64_t)prog->num_values * PAI_IR_VALUE_REC_SIZE;
    for (uint32_t o = 1; o <= prog->num_ops; o++) {
      const pai_ir_op_t *ir_op = &prog->ops[o];
      op[0] = ir_op->kind;
      op[1] = ir_op->device;
      op[2] = ir_op->num_inputs;
      op[3] = ir_op->num_outputs;
      le_put_u16(op + 4, 0);
      le_put_u16(op + 6, 0);
      for (uint32_t i = 0; i < ir_op->num_inputs; i++) {
        le_put_u16(op + 8 + 2u * i, ir_op->inputs[i]);
      }
      for (uint32_t i = 0; i < ir_op->num_outputs; i++) {
        le_put_u16(op + 8 + 2u * ir_op->num_inputs + 2u * i,
                   ir_op->outputs[i]);
      }
      op += op_record_size(ir_op);
    }

    for (uint32_t i = 0; i < prog->num_inputs; i++) {
      le_put_u32(op, prog->input_ids[i]);
      op += 4;
    }
    for (uint32_t i = 0; i < prog->num_outputs; i++) {
      le_put_u32(op, prog->output_ids[i]);
      op += 4;
    }
  }

  /* CRC-32 over header bytes [0,24) and the body (the CRC and reserved
   * fields are excluded), using the protocol's streaming triple so the
   * blob hashes identically to the concatenated stream. */
  crc = pai_proto_crc32_init();
  crc = pai_proto_crc32_upd(crc, out, PAI_IR_CRC_COVER);
  crc = pai_proto_crc32_upd(crc, body, (uint32_t)(total - PAI_IR_HEADER_SIZE));
  crc = pai_proto_crc32_fin(crc);
  le_put_u32(out + 24, crc);

  *out_nbytes = total;
  return PAI_OK;
}

pai_status_t
pai_ir_decode(const uint8_t *data, uint32_t nbytes, pai_ir_program_t *out) {
  const uint8_t *body;
  const uint8_t *p;
  uint32_t num_values;
  uint32_t num_ops;
  uint32_t num_inputs;
  uint32_t num_outputs;
  uint32_t expected_crc;
  uint32_t crc;
  uint64_t body_bytes;

  if (data == NULL || out == NULL || nbytes < PAI_IR_HEADER_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }

  if (le_get_u32(data + 0) != PAI_IR_MAGIC) {
    return PAI_ERR_PROTOCOL;
  }
  if (le_get_u16(data + 4) != PAI_IR_VERSION) {
    return PAI_ERR_PROTOCOL;
  }
  if (le_get_u16(data + 6) != 0 || le_get_u32(data + 28) != 0) {
    /* flags and reserved must be 0 in v1 */
    return PAI_ERR_PROTOCOL;
  }

  num_values = le_get_u32(data + 8);
  num_ops = le_get_u32(data + 12);
  num_inputs = le_get_u32(data + 16);
  num_outputs = le_get_u32(data + 20);
  expected_crc = le_get_u32(data + 24);

  if (num_values > PAI_IR_MAX_VALUES || num_ops > PAI_IR_MAX_OPS ||
      num_inputs > PAI_IR_MAX_VALUES || num_outputs > PAI_IR_MAX_VALUES) {
    return PAI_ERR_PROTOCOL;
  }

  /* Integrity first: recompute CRC over [0,24) + body. */
  body = data + PAI_IR_HEADER_SIZE;
  body_bytes = (uint64_t)nbytes - PAI_IR_HEADER_SIZE;
  crc = pai_proto_crc32_init();
  crc = pai_proto_crc32_upd(crc, data, PAI_IR_CRC_COVER);
  crc = pai_proto_crc32_upd(crc, body, (uint32_t)body_bytes);
  crc = pai_proto_crc32_fin(crc);
  if (crc != expected_crc) {
    return PAI_ERR_PROTOCOL;
  }

  /* Bounds-check the record area before touching anything. */
  if ((uint64_t)num_values * PAI_IR_VALUE_REC_SIZE > body_bytes) {
    return PAI_ERR_PROTOCOL;
  }
  p = body + (uint64_t)num_values * PAI_IR_VALUE_REC_SIZE;
  for (uint32_t o = 0; o < num_ops; o++) {
    uint32_t ni;
    uint32_t no;
    if ((uint64_t)(p - body) + 8 > body_bytes) {
      return PAI_ERR_PROTOCOL;
    }
    ni = p[2];
    no = p[3];
    if ((uint64_t)(p - body) + 8 + 2u * (ni + no) > body_bytes) {
      return PAI_ERR_PROTOCOL;
    }
    p += 8 + 2u * (ni + no);
  }
  if ((uint64_t)(p - body) + 4u * (uint64_t)(num_inputs + num_outputs) >
      body_bytes) {
    return PAI_ERR_PROTOCOL;
  }

  memset(out, 0, sizeof(*out));
  out->num_values = num_values;
  out->num_ops = num_ops;
  out->num_inputs = num_inputs;
  out->num_outputs = num_outputs;

  /* Rewind to the start of the op records (the bounds-check pass above
   * advanced p past them). */
  p = body + (uint64_t)num_values * PAI_IR_VALUE_REC_SIZE;

  for (uint32_t v = 1; v <= num_values; v++) {
    const uint8_t *r = body + (uint64_t)(v - 1) * PAI_IR_VALUE_REC_SIZE;
    pai_ir_value_t *val = &out->values[v];
    uint64_t bytes;

    if (r[0] >= PAI_DTYPE_COUNT || r[1] == 0 || r[1] > PAI_TENSOR_MAX_RANK ||
        r[2] >= PAI_IR_VALUE_KIND_COUNT || r[3] >= PAI_IR_DEVICE_COUNT) {
      return PAI_ERR_PROTOCOL;
    }
    val->dtype = (pai_dtype_t)r[0];
    val->rank = r[1];
    val->kind = r[2];
    val->device = r[3];
    val->quant.present = r[4];
    val->quant.bit_width = r[5];
    val->quant.scale_repr = r[6];
    val->quant.zero_point_repr = r[7];
    val->quant.is_signed = r[8];
    val->quant.block_structure = r[9];
    val->quant.group_size = le_get_u16(r + 10);
    if (val->quant.present &&
        (val->quant.bit_width == 0 || val->quant.bit_width > 64 ||
         val->quant.scale_repr >= PAI_IR_QUANT_REPR_COUNT ||
         val->quant.zero_point_repr >= PAI_IR_QUANT_REPR_COUNT ||
         val->quant.block_structure >= PAI_IR_QUANT_BLOCK_COUNT ||
         (val->quant.block_structure == PAI_IR_QUANT_BLOCK_GROUP &&
          val->quant.group_size == 0))) {
      return PAI_ERR_PROTOCOL;
    }
    val->align = le_get_u64(r + 12);
    val->size_bytes = le_get_u64(r + 20);
    for (uint32_t d = 0; d < PAI_TENSOR_MAX_RANK; d++) {
      val->shape[d] = le_get_u64(r + 28 + 8u * d);
    }

    /* Cross-check size against the shape/dtype (catches inconsistent
     * blobs that a CRC alone would miss). */
    if (pai_tensor_contiguous_size(val->dtype, val->rank, val->shape,
                                   &bytes) != PAI_OK ||
        bytes != val->size_bytes || val->size_bytes == 0) {
      return PAI_ERR_PROTOCOL;
    }
    if (val->align == 0 || (val->align & (val->align - 1)) != 0) {
      return PAI_ERR_PROTOCOL;
    }
  }

  for (uint32_t o = 1; o <= num_ops; o++) {
    const uint8_t *rec = p;
    pai_ir_op_t *op = &out->ops[o];
    uint32_t ni = rec[2];
    uint32_t no = rec[3];

    if (rec[0] >= PAI_IR_OP_KIND_COUNT || rec[1] >= PAI_IR_DEVICE_COUNT ||
        ni == 0 || ni > PAI_IR_MAX_ARITY || no == 0 ||
        no > PAI_IR_MAX_ARITY) {
      return PAI_ERR_PROTOCOL;
    }
    op->kind = rec[0];
    op->device = rec[1];
    op->num_inputs = (uint8_t)ni;
    op->num_outputs = (uint8_t)no;
    for (uint32_t i = 0; i < ni; i++) {
      uint16_t id = le_get_u16(rec + 8 + 2u * i);
      if (id == 0 || id > num_values) {
        return PAI_ERR_PROTOCOL;
      }
      op->inputs[i] = id;
    }
    for (uint32_t i = 0; i < no; i++) {
      uint16_t id = le_get_u16(rec + 8 + 2u * ni + 2u * i);
      if (id == 0 || id > num_values) {
        return PAI_ERR_PROTOCOL;
      }
      op->outputs[i] = id;
      for (uint32_t k = 0; k < i; k++) {
        if (op->outputs[k] == id) {
          return PAI_ERR_PROTOCOL;
        }
      }
      for (uint32_t k = 0; k < ni; k++) {
        if (op->inputs[k] == id) {
          return PAI_ERR_PROTOCOL;
        }
      }
    }
    p = rec + 8 + 2u * (ni + no);
  }

  {
    const uint8_t *ids = p;
    for (uint32_t i = 0; i < num_inputs; i++) {
      uint32_t id = le_get_u32(ids + 4u * i);
      if (id == 0 || id > num_values) {
        return PAI_ERR_PROTOCOL;
      }
      out->input_ids[i] = id;
    }
    ids += 4u * num_inputs;
    for (uint32_t i = 0; i < num_outputs; i++) {
      uint32_t id = le_get_u32(ids + 4u * i);
      if (id == 0 || id > num_values) {
        return PAI_ERR_PROTOCOL;
      }
      out->output_ids[i] = id;
    }
  }

  return PAI_OK;
}
