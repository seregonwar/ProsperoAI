/*
 * ProsperoAI — Prospero IR (whitepaper §10.1) and Kernel IR (§10.2)
 *
 * Graph-level IR between model import and the graph compiler:
 * tensors, shapes, dtypes, quantization metadata (§15), operators,
 * memory alignment and device constraints — hardware-aware but not
 * PS5-kernel-specific. Serializes to a flat little-endian blob (magic
 * + version + counts + CRC-32 + value/op records) that can be embedded
 * in .pai containers or transferred over the protocol (§20/§24).
 *
 * Kernel IR (§10.2) is the lower level below fusion; the kir/ part
 * ships descriptor + builder + a naive per-op lowering.
 */

#ifndef PAI_IR_H
#define PAI_IR_H

#include <pai/dtype.h>
#include <pai/error.h>
#include <pai/tensor.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_IR_MAX_VALUES 1024u /* mirrors PAI_GRAPH_MAX_VALUES */
#define PAI_IR_MAX_OPS    1024u
#define PAI_IR_MAX_ARITY  16u

struct pai_graph; /* forward: conversion helpers accept graph layer types */

/* IR op kinds (mirrors the graph op set; conversion lives in
 * ir_graph.c). */

typedef enum pai_ir_op_kind {
  PAI_IR_OP_NONE = 0,
  PAI_IR_OP_ADD,
  PAI_IR_OP_MUL,
  PAI_IR_OP_GEMM,
  PAI_IR_OP_GEMV,
  PAI_IR_OP_MATMUL,
  PAI_IR_OP_RELU,
  PAI_IR_OP_SOFTMAX,
  PAI_IR_OP_LAYERNORM,
  PAI_IR_OP_RMSNORM,
  PAI_IR_OP_ROPE,
  PAI_IR_OP_ATTENTION,
  PAI_IR_OP_RESHAPE,
  PAI_IR_OP_CONCAT,
  PAI_IR_OP_CONVERT,
  PAI_IR_OP_COPY,
  PAI_IR_OP_SILU,
  PAI_IR_OP_CUSTOM,
  PAI_IR_OP_KIND_COUNT,
} pai_ir_op_kind_t;

/* Value categories and device placement constraints (§10.1). */

typedef enum pai_ir_value_kind {
  PAI_IR_VALUE_ACTIVATION = 0, /* intermediate tensor                  */
  PAI_IR_VALUE_INPUT,          /* graph input                          */
  PAI_IR_VALUE_PARAM,          /* weight / bias (producer-less input)  */
  PAI_IR_VALUE_CONSTANT,       /* folded constant (desktop-computed)   */
  PAI_IR_VALUE_OUTPUT,         /* graph output                         */
  PAI_IR_VALUE_KIND_COUNT,
} pai_ir_value_kind_t;

typedef enum pai_ir_device {
  PAI_IR_DEVICE_ANY = 0, /* no constraint; scheduler decides (§18)      */
  PAI_IR_DEVICE_CPU,     /* required on CPU                              */
  PAI_IR_DEVICE_GPU,     /* required on GPU                              */
  PAI_IR_DEVICE_COUNT,
} pai_ir_device_t;

/* Quantization metadata (§15) — describes the scheme; the runtime is
 * quantization-agnostic (no fixed scheme list). */

typedef enum pai_ir_quant_repr {
  PAI_IR_QUANT_REPR_F32 = 0,
  PAI_IR_QUANT_REPR_F16,
  PAI_IR_QUANT_REPR_F8,
  PAI_IR_QUANT_REPR_U8,
  PAI_IR_QUANT_REPR_COUNT,
} pai_ir_quant_repr_t;

typedef enum pai_ir_quant_block {
  PAI_IR_QUANT_BLOCK_TENSOR = 0, /* one scale/zp for the whole tensor  */
  PAI_IR_QUANT_BLOCK_CHANNEL,    /* per-channel                         */
  PAI_IR_QUANT_BLOCK_GROUP,      /* per-group (group_size elements)     */
  PAI_IR_QUANT_BLOCK_COUNT,
} pai_ir_quant_block_t;

typedef struct pai_ir_quant {
  uint8_t present;          /* 0 = unquantized (f32 storage)           */
  uint8_t bit_width;        /* storage bits per element (e.g. 4/8/16)  */
  uint8_t scale_repr;       /* pai_ir_quant_repr_t                     */
  uint8_t zero_point_repr;  /* pai_ir_quant_repr_t                     */
  uint8_t is_signed;
  uint8_t block_structure;  /* pai_ir_quant_block_t                    */
  uint16_t group_size;      /* elements per group; 0 when per-tensor   */
} pai_ir_quant_t;

/* ------------------------------------------------------------------ */
/* In-memory program representation                                    */
/* ------------------------------------------------------------------ */

typedef struct pai_ir_value {
  pai_dtype_t dtype;
  uint8_t     rank;
  uint8_t     kind;    /* pai_ir_value_kind_t                          */
  uint8_t     device;  /* pai_ir_device_t                              */
  uint8_t     reserved;
  uint64_t    shape[PAI_TENSOR_MAX_RANK];
  uint64_t    size_bytes; /* contiguous payload bytes                  */
  uint64_t    align;      /* storage alignment                         */
  pai_ir_quant_t quant;
} pai_ir_value_t;

typedef struct pai_ir_op {
  uint8_t  kind;        /* pai_ir_op_kind_t                            */
  uint8_t  device;      /* pai_ir_device_t placement constraint        */
  uint8_t  num_inputs;
  uint8_t  num_outputs;
  uint16_t inputs[PAI_IR_MAX_ARITY];  /* value indexes (1-based)       */
  uint16_t outputs[PAI_IR_MAX_ARITY]; /* value indexes (1-based)       */
} pai_ir_op_t;

typedef struct pai_ir_program {
  pai_ir_value_t values[PAI_IR_MAX_VALUES + 1]; /* index = value id    */
  pai_ir_op_t    ops[PAI_IR_MAX_OPS + 1];       /* index = op id       */
  uint32_t num_values;
  uint32_t num_ops;
  uint32_t num_inputs;                       /* graph input value ids   */
  uint32_t num_outputs;                      /* graph output value ids  */
  uint32_t input_ids[PAI_IR_MAX_VALUES];
  uint32_t output_ids[PAI_IR_MAX_VALUES];
} pai_ir_program_t;

void pai_ir_init(pai_ir_program_t *prog);

/* Value builder. Returns the value id (1-based), or 0 on failure. */
uint32_t pai_ir_add_value(pai_ir_program_t *prog, pai_dtype_t dtype,
                          uint32_t rank, const uint64_t *shape,
                          uint64_t align, uint8_t kind, uint8_t device);

/* Attach quantization metadata to an existing value. */
pai_status_t pai_ir_value_set_quant(pai_ir_program_t *prog, uint32_t value_id,
                                    const pai_ir_quant_t *quant);

/* Op builder. Inputs/outputs must reference existing value ids; no
 * single-assignment rule is enforced here (that is a graph invariant),
 * but duplicate output ids within one op are rejected. */
uint32_t pai_ir_add_op(pai_ir_program_t *prog, uint8_t kind, uint8_t device,
                       uint32_t num_inputs, uint32_t num_outputs,
                       const uint16_t *inputs, const uint16_t *outputs);

void pai_ir_set_input(pai_ir_program_t *prog, uint32_t value_id);
void pai_ir_set_output(pai_ir_program_t *prog, uint32_t value_id);

/* Stable names for diagnostics. */
const char *pai_ir_op_kind_name(uint8_t kind);
const char *pai_ir_value_kind_name(uint8_t kind);
const char *pai_ir_device_name(uint8_t device);

/* ------------------------------------------------------------------ */
/* Flat binary serialization                                           */
/* ------------------------------------------------------------------ */

#define PAI_IR_MAGIC       0x52495041u /* "PAIR"                       */
#define PAI_IR_VERSION     1u
#define PAI_IR_HEADER_SIZE 32u
#define PAI_IR_VALUE_REC_SIZE 76u
#define PAI_IR_CRC_COVER   24u /* header bytes hashed by the CRC        */

/*
 * Wire layout (little-endian):
 *
 *   [0]   u32 magic           0x52495041 ("PAIR")
 *   [4]   u16 version         PAI_IR_VERSION
 *   [6]   u16 flags           0
 *   [8]   u32 num_values
 *   [12]  u32 num_ops
 *   [16]  u32 num_input_ids
 *   [20]  u32 num_output_ids
 *   [24]  u32 crc32           over bytes [0,24) + body
 *   [28]  u32 reserved        0
 *   [32]  body:
 *           value records     num_values x PAI_IR_VALUE_REC_SIZE:
 *             [0]  u8  dtype
 *             [1]  u8  rank
 *             [2]  u8  kind
 *             [3]  u8  device
 *             [4]  u8  quant.present
 *             [5]  u8  quant.bit_width
 *             [6]  u8  quant.scale_repr
 *             [7]  u8  quant.zero_point_repr
 *             [8]  u8  quant.is_signed
 *             [9]  u8  quant.block_structure
 *             [10] u16 quant.group_size
 *             [12] u64 align
 *             [20] u64 size_bytes
 *             [28] u64 shape[6]  (unused dims are 0)
 *           op records     variable size:
 *             [0]  u8  kind
 *             [1]  u8  device
 *             [2]  u8  num_inputs
 *             [3]  u8  num_outputs
 *             [4]  u16 reserved
 *             [6]  u16 reserved
 *             [8]  u16 inputs[num_inputs]
 *                  u16 outputs[num_outputs]
 *           input ids     num_input_ids x u32 (value ids)
 *           output ids    num_output_ids x u32 (value ids)
 */

/* Size of the encoded blob; 0 when the program cannot be encoded. */
uint32_t pai_ir_encoded_size(const pai_ir_program_t *prog);

pai_status_t pai_ir_encode(const pai_ir_program_t *prog, uint8_t *out,
                           uint32_t cap, uint32_t *out_nbytes);

/* Validates magic, version, counts, CRC and record bounds; corrupted
 * or truncated blobs yield PAI_ERR_PROTOCOL. */
pai_status_t pai_ir_decode(const uint8_t *data, uint32_t nbytes,
                           pai_ir_program_t *out);

/* ------------------------------------------------------------------ */
/* Graph <-> IR conversion (ir_graph.c)                                */
/* ------------------------------------------------------------------ */

/* Convert a pai_graph into Prospero IR (single-assignment ops map 1:1;
 * value kinds derive from input/output flags and producers). */
pai_status_t pai_ir_from_graph(const struct pai_graph *graph,
                               pai_ir_program_t *out);


/* Rebuild a pai_graph from IR. Returns PAI_ERR_MISMATCH when the IR
 * violates graph invariants (duplicate producers, self-loops). */
pai_status_t pai_ir_to_graph(const pai_ir_program_t *ir,
                             struct pai_graph *out);

/* ------------------------------------------------------------------ */
/* Kernel IR (§10.2)                                                   */
/* ------------------------------------------------------------------ */

#define PAI_KIR_MAX_OPS 16u
#define PAI_KIR_OP_MAX_IN  8u
#define PAI_KIR_OP_MAX_OUT 2u
#define PAI_KIR_NAME_MAX   32u

typedef struct pai_kir_tile {
  uint32_t x;
  uint32_t y;
  uint32_t z;
} pai_kir_tile_t;

typedef enum pai_kir_fusion {
  PAI_KIR_FUSION_NONE = 0,
  PAI_KIR_FUSION_ELEM,     /* element-wise chain fused into one kernel   */
  PAI_KIR_FUSION_GEMM_ACT, /* GEMM + activation (epilogue fused)         */
} pai_kir_fusion_t;

typedef enum pai_kir_mem_layout {
  PAI_KIR_MEM_ROW_MAJOR = 0,
  PAI_KIR_MEM_TILED,       /* 2D tile-padded layout                      */
  PAI_KIR_MEM_PACKED,      /* quantized packed layout                    */
} pai_kir_mem_layout_t;

typedef struct pai_kir_op {
  uint8_t  kind;        /* pai_ir_op_kind_t                              */
  uint8_t  num_inputs;
  uint8_t  num_outputs;
  uint8_t  vector_width;   /* SIMD lane packing (1/2/4)                  */
  uint8_t  mem_layout;     /* pai_kir_mem_layout_t                       */
  uint8_t  reserved;
  uint16_t in[PAI_KIR_OP_MAX_IN];
  uint16_t out[PAI_KIR_OP_MAX_OUT];
} pai_kir_op_t;

typedef struct pai_kir_kernel {
  char          name[PAI_KIR_NAME_MAX];
  pai_kir_tile_t workgroup;  /* wavefront/workgroup configuration        */
  pai_kir_tile_t tile;       /* per-thread compute tile                  */
  uint8_t       vector_width;
  uint8_t       fusion;      /* pai_kir_fusion_t                         */
  uint8_t       num_ops;
  pai_kir_op_t  ops[PAI_KIR_MAX_OPS];
} pai_kir_kernel_t;

void pai_kir_init(pai_kir_kernel_t *kernel, const char *name);

pai_status_t pai_kir_add_op(pai_kir_kernel_t *kernel, uint8_t kind,
                            uint32_t num_inputs, uint32_t num_outputs,
                            const uint16_t *in, const uint16_t *out,
                            uint8_t vector_width, uint8_t mem_layout);

pai_status_t pai_kir_set_workgroup(pai_kir_kernel_t *kernel, uint32_t x,
                                   uint32_t y, uint32_t z);
pai_status_t pai_kir_set_tile(pai_kir_kernel_t *kernel, uint32_t x,
                              uint32_t y, uint32_t z);

/*
 * Naive lowering: one kernel per IR op (fusion is a later pass).
 * vector_width is derived from the value shapes (4 when all involved
 * buffers hold a multiple of 4 elements, else 1). `kernels` must hold
 * at least PAI_IR_MAX_OPS entries.
 */
pai_status_t pai_kir_lower_program(const pai_ir_program_t *prog,
                                   pai_kir_kernel_t *kernels,
                                   uint32_t max_kernels,
                                   uint32_t *out_num_kernels);

#ifdef __cplusplus
}
#endif

#endif /* PAI_IR_H */
