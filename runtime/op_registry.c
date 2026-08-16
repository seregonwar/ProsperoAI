/*
 * ProsperoAI — op registry (T5A).
 *
 * Builtin Phase-1 op set: name/version/arity/dtype/caps/ref-fn/kernel-id
 * per op, plus the eager CPU fallback that runs the pai_ref_* oracle on
 * dense row-major tensors.
 */

#include <pai/op.h>

#include <ref_ops.h>

#include <string.h>

/* --- kernel id vocabulary ----------------------------------------------- */

const char *
pai_op_kernel_id_str(pai_kernel_id_t id) {
  switch (id) {
  case PAI_KERNEL_ADD1D:
    return "add1d";
  case PAI_KERNEL_SUB1D:
    return "sub1d";
  case PAI_KERNEL_MUL1D:
    return "mul1d";
  case PAI_KERNEL_RELU:
    return "relu";
  case PAI_KERNEL_CLIP:
    return "clip";
  case PAI_KERNEL_BIASADD:
    return "biasadd";
  case PAI_KERNEL_MATMUL:
    return "matmul";
  case PAI_KERNEL_MUL1D_U32:
    return "mul1d_u32";
  case PAI_KERNEL_SUB1D_U32:
    return "sub1d_u32";
  case PAI_KERNEL_RELU_U32:
    return "relu_u32";
  case PAI_KERNEL_CLIP_U32:
    return "clip_u32";
  case PAI_KERNEL_ADD2D_U32:
    return "add2d_u32";
  case PAI_KERNEL_MATMUL_U32:
    return "matmul_u32";
  default:
    return "unknown";
  }
}

/* --- builtin op table ---------------------------------------------------- */

/* The registry stores pai_ref_* functions with heterogeneous signatures
 * as an opaque pai_op_ref_fn; the cast is by design (dispatched per
 * kernel id). Silence the incompatible-cast diagnostic around the
 * table. */
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-function-type-mismatch"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

static const pai_op_descriptor_t s_builtin_ops[] = {
  /* Float serial GPU set (G42-G48, T4A) — CPU ref + GPU serial + host
   * mirror all present. */
  {.name = "vecadd_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_vecadd_f32,
   .kernel_id = PAI_KERNEL_ADD1D},
  {.name = "vecsub_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_vecsub_f32,
   .kernel_id = PAI_KERNEL_SUB1D},
  {.name = "vecmul_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_vecmul_f32,
   .kernel_id = PAI_KERNEL_MUL1D},
  {.name = "relu_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_relu_f32,
   .kernel_id = PAI_KERNEL_RELU},
  {.name = "clip_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_clip_f32,
   .kernel_id = PAI_KERNEL_CLIP},
  {.name = "biasadd_f32",
   .version = 1,
   .num_inputs = 2, /* a (rows x cols), bias (cols) */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_biasadd_f32,
   .kernel_id = PAI_KERNEL_BIASADD},
  {.name = "matmul_f32",
   .version = 1,
   .num_inputs = 2, /* a (m x k), b (k x n) */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL | PAI_OP_CAP_HOST_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_gemm_f32,
   .kernel_id = PAI_KERNEL_MATMUL},

  /* Reference-only ops (no serial GPU kernel yet). */
  {.name = "gemv_f32",
   .version = 1,
   .num_inputs = 2, /* a (m x k), x (k) */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_gemv_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "scale_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_scale_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "softmax_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_softmax_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "silu_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_silu_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "copy_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_copy_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "dot_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1, /* scalar (1 element) */
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_dot_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "l1norm_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1, /* scalar */
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_l1norm_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "l2norm_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1, /* scalar */
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_l2norm_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "concat_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_concat_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "layernorm_f32",
   .version = 1,
   .num_inputs = 3, /* a, gamma (opt), beta (opt) */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_layernorm_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "rmsnorm_f32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_rmsnorm_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "rmsnorm_gamma_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_rmsnorm_gamma_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "rope_f32",
   .version = 1,
   .num_inputs = 6, /* x, cos_t, sin_t (scalars via params) */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_rope_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "attention_f32",
   .version = 1,
   .num_inputs = 3, /* q, k, v */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_attention_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "gemm_f32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_gemm_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "gemm_w8_f32",
   .version = 1,
   .num_inputs = 3, /* a, w_q, b_scales */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_gemm_w8_f32,
   .kernel_id = PAI_KERNEL_NONE},
  {.name = "gemm_w4_f32",
   .version = 1,
   .num_inputs = 3, /* a, w_q4, b_scales */
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_F32,
   .out_dtype = PAI_DTYPE_F32,
   .caps = PAI_OP_CAP_CPU_REF,
   .ref_fn = (pai_op_ref_fn)pai_ref_gemm_w4_f32,
   .kernel_id = PAI_KERNEL_NONE},

  /* Integer serial vocabulary — reserved for T4B (ref fns land with the
   * kernels; caps stay 0 until then). */
  {.name = "mul1d_u32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_MUL1D_U32},
  {.name = "sub1d_u32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_SUB1D_U32},
  {.name = "relu_u32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_RELU_U32},
  {.name = "clip_u32",
   .version = 1,
   .num_inputs = 1,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_CLIP_U32},
  {.name = "add2d_u32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_ADD2D_U32},
  {.name = "matmul_u32",
   .version = 1,
   .num_inputs = 2,
   .num_outputs = 1,
   .in_dtype = PAI_DTYPE_U32,
   .out_dtype = PAI_DTYPE_U32,
   .caps = 0,
   .ref_fn = NULL,
   .kernel_id = PAI_KERNEL_MATMUL_U32},
};

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#define PAI_BUILTIN_OP_COUNT \
  (sizeof(s_builtin_ops) / sizeof(s_builtin_ops[0]))

/* --- registry ------------------------------------------------------------- */

pai_status_t
pai_op_registry_init(pai_op_registry_t *reg) {
  if (!reg) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(reg, 0, sizeof(*reg));
  reg->version = 1;
  return PAI_OK;
}

pai_status_t
pai_op_registry_register(pai_op_registry_t *reg,
                         const pai_op_descriptor_t *desc) {
  if (!reg || !desc || !desc->name) {
    return PAI_ERR_INVALID_ARG;
  }
  if (reg->count >= PAI_OP_REGISTRY_MAX_OPS) {
    return PAI_ERR_NOMEM;
  }
  reg->ops[reg->count++] = desc;
  return PAI_OK;
}

pai_status_t
pai_op_registry_builtin(pai_op_registry_t *reg) {
  pai_status_t st;
  uint32_t i;
  if (!reg) {
    return PAI_ERR_INVALID_ARG;
  }
  st = pai_op_registry_init(reg);
  if (st != PAI_OK) {
    return st;
  }
  for (i = 0; i < PAI_BUILTIN_OP_COUNT; i++) {
    st = pai_op_registry_register(reg, &s_builtin_ops[i]);
    if (st != PAI_OK) {
      return st;
    }
  }
  return PAI_OK;
}

const pai_op_descriptor_t *
pai_op_registry_lookup(const pai_op_registry_t *reg, const char *name) {
  uint32_t i;
  if (!reg || !name) {
    return NULL;
  }
  for (i = 0; i < reg->count; i++) {
    if (strcmp(reg->ops[i]->name, name) == 0) {
      return reg->ops[i];
    }
  }
  return NULL;
}

const pai_op_descriptor_t *
pai_op_registry_lookup_kernel(const pai_op_registry_t *reg,
                              pai_kernel_id_t id) {
  uint32_t i;
  if (!reg || id == PAI_KERNEL_NONE) {
    return NULL;
  }
  for (i = 0; i < reg->count; i++) {
    if (reg->ops[i]->kernel_id == id) {
      return reg->ops[i];
    }
  }
  return NULL;
}

uint32_t
pai_op_registry_count(const pai_op_registry_t *reg) {
  return reg ? reg->count : 0;
}

uint32_t
pai_op_supports(const pai_op_descriptor_t *desc, uint32_t caps) {
  return desc && (desc->caps & caps) == caps;
}

/* --- eager CPU fallback --------------------------------------------------- */

static pai_status_t
pai_op_eval_check(const pai_op_descriptor_t *d, pai_tensor_t *const *in,
                  pai_tensor_t *const *out) {
  uint32_t i;
  for (i = 0; i < d->num_inputs; i++) {
    if (!in[i] || !in[i]->data) {
      return PAI_ERR_INVALID_ARG;
    }
    if (in[i]->dtype != d->in_dtype) {
      return PAI_ERR_MISMATCH;
    }
  }
  for (i = 0; i < d->num_outputs; i++) {
    if (!out[i] || !out[i]->data) {
      return PAI_ERR_INVALID_ARG;
    }
    if (out[i]->dtype != d->out_dtype) {
      return PAI_ERR_MISMATCH;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_op_eval_f32(const pai_op_registry_t *reg, const char *name,
                pai_tensor_t *const *inputs, uint32_t num_inputs,
                pai_tensor_t *const *outputs, uint32_t num_outputs,
                const float *params, uint32_t num_params) {
  const pai_op_descriptor_t *d;
  pai_status_t st;
  uint64_t n;

  d = pai_op_registry_lookup(reg, name);
  if (!d) {
    return PAI_ERR_UNSUPPORTED;
  }
  if (d->num_inputs != num_inputs || d->num_outputs != num_outputs) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!d->ref_fn) {
    return PAI_ERR_UNSUPPORTED; /* vocabulary-only entry (T4B reserved) */
  }
  st = pai_op_eval_check(d, inputs, outputs);
  if (st != PAI_OK) {
    return st;
  }

  switch (d->kernel_id) {
  case PAI_KERNEL_ADD1D:
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n || pai_tensor_nelem(inputs[1]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_vecadd_f32(inputs[0]->data, inputs[1]->data,
                              outputs[0]->data, n);
  case PAI_KERNEL_SUB1D:
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n || pai_tensor_nelem(inputs[1]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_vecsub_f32(inputs[0]->data, inputs[1]->data,
                              outputs[0]->data, n);
  case PAI_KERNEL_MUL1D:
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n || pai_tensor_nelem(inputs[1]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_vecmul_f32(inputs[0]->data, inputs[1]->data,
                              outputs[0]->data, n);
  case PAI_KERNEL_RELU:
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_relu_f32(inputs[0]->data, outputs[0]->data, n);
  case PAI_KERNEL_CLIP: {
    float lo = num_params > 0 ? params[0] : 0.0f;
    float hi = num_params > 1 ? params[1] : 1.0f;
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_clip_f32(inputs[0]->data, outputs[0]->data, n, lo, hi);
  }
  case PAI_KERNEL_BIASADD: {
    uint64_t cols = pai_tensor_nelem(inputs[1]);
    uint64_t rows = cols ? pai_tensor_nelem(inputs[0]) / cols : 0;
    if (cols == 0 || rows * cols != pai_tensor_nelem(inputs[0]) ||
        rows * cols != pai_tensor_nelem(outputs[0])) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_biasadd_f32(inputs[0]->data, inputs[1]->data,
                               outputs[0]->data, rows, cols);
  }
  case PAI_KERNEL_MATMUL: {
    /* Rank-2 operands only in the eager path; rank-1 row-vector
     * matmul (supported by the scheduler executor) is out of scope
     * here — the executor's run_gemm handles that shape. */
    const pai_tensor_t *a = inputs[0];
    const pai_tensor_t *b = inputs[1];
    uint64_t m, k, k2, n;
    m = a->rank >= 2 ? a->shape[0] : 0;
    k = a->rank >= 2 ? a->shape[1] : 0;
    n = b->rank >= 2 ? b->shape[1] : 0;
    k2 = b->rank >= 2 ? b->shape[0] : 0;
    if (m == 0 || n == 0 || k == 0 || k != k2) {
      return PAI_ERR_INVALID_ARG;
    }
    if (pai_tensor_nelem(a) != m * k || pai_tensor_nelem(b) != k * n ||
        pai_tensor_nelem(outputs[0]) != m * n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_gemm_f32(m, n, k, a->data, b->data, outputs[0]->data);
  }
  default:
    break;
  }

  /* Ref-only ops without a dedicated eager path use their flat-buffer
   * single-row semantics where the tensor arity matches the call. */
  if (strcmp(d->name, "gemv_f32") == 0) {
    const pai_tensor_t *a = inputs[0];
    uint64_t m = pai_tensor_nelem(outputs[0]);
    uint64_t k = m ? pai_tensor_nelem(a) / m : 0;
    if (m == 0 || k == 0 || pai_tensor_nelem(a) != m * k || pai_tensor_nelem(inputs[1]) != k) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_gemv_f32(m, k, a->data, inputs[1]->data,
                            outputs[0]->data);
  }
  if (strcmp(d->name, "softmax_f32") == 0) {
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_softmax_f32(inputs[0]->data, outputs[0]->data, n);
  }
  if (strcmp(d->name, "silu_f32") == 0) {
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_silu_f32(inputs[0]->data, outputs[0]->data, n);
  }
  if (strcmp(d->name, "copy_f32") == 0) {
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_copy_f32(inputs[0]->data, outputs[0]->data, n);
  }
  if (strcmp(d->name, "dot_f32") == 0) {
    n = pai_tensor_nelem(inputs[0]);
    if (pai_tensor_nelem(inputs[1]) != n || pai_tensor_nelem(outputs[0]) != 1) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_dot_f32(inputs[0]->data, inputs[1]->data,
                           outputs[0]->data, n);
  }
  if (strcmp(d->name, "l1norm_f32") == 0) {
    n = pai_tensor_nelem(inputs[0]);
    if (pai_tensor_nelem(outputs[0]) != 1) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_l1norm_f32(inputs[0]->data, outputs[0]->data, n);
  }
  if (strcmp(d->name, "l2norm_f32") == 0) {
    n = pai_tensor_nelem(inputs[0]);
    if (pai_tensor_nelem(outputs[0]) != 1) {
      return PAI_ERR_INVALID_ARG;
    }
    return pai_ref_l2norm_f32(inputs[0]->data, outputs[0]->data, n);
  }
  if (strcmp(d->name, "scale_f32") == 0) {
    float alpha = num_params > 0 ? params[0] : 1.0f;
    n = pai_tensor_nelem(outputs[0]);
    if (pai_tensor_nelem(inputs[0]) != n) {
      return PAI_ERR_INVALID_ARG;
    }
    if (outputs[0]->data != inputs[0]->data) {
      memcpy(outputs[0]->data, inputs[0]->data, n * sizeof(float));
    }
    return pai_ref_scale_f32(outputs[0]->data, alpha, n);
  }

  return PAI_ERR_UNSUPPORTED;
}
