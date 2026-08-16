/*
 * ProsperoAI — public SDK: operation registry (T5A, whitepaper §8/§37).
 *
 * Central description of every op the runtime knows: stable name and
 * version, tensor arity, element dtypes, device capability flags, the
 * CPU reference implementation (the correctness oracle, §37) and the
 * serial GPU kernel id (T4 vocabulary, G42-G48 + reserved T4B ids).
 *
 * The registry is read-only metadata; execution is dispatched by the
 * kernel vtable (T5B) or through the eager CPU fallback below.
 */

#ifndef PAI_OP_H
#define PAI_OP_H

#include <pai/dtype.h>
#include <pai/error.h>
#include <pai/tensor.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- device kernel vocabulary ------------------------------------------ */

typedef int32_t pai_kernel_id_t;

#define PAI_KERNEL_NONE 0

/* Serial GPU kernels. G42-G48 are the float set validated on console
 * 9021 (t4_ops.s / pai_t4_ops.inc, T4A). The _U32 ids are the reserved
 * vocabulary for the T4B integer kernels (mul1d/sub1d/relu/clip/add2d/
 * matmul_u32): registered here so lookups resolve before the kernels
 * land — T4B wires the actual shaders to these ids. */
enum {
  PAI_KERNEL_ADD1D = 1,      /* G42 c[i]    = a[i] + b[i]          (f32) */
  PAI_KERNEL_SUB1D = 2,      /* G43 c[i]    = a[i] - b[i]          (f32) */
  PAI_KERNEL_MUL1D = 3,      /* G44 c[i]    = a[i] * b[i]          (f32) */
  PAI_KERNEL_RELU  = 4,      /* G45 c[i]    = max(a[i], 0)         (f32) */
  PAI_KERNEL_CLIP  = 5,      /* G46 c[i]    = clamp(a[i], lo, hi)  (f32) */
  PAI_KERNEL_BIASADD = 6,    /* G47 c[i][j] = a[i][j] + bias[j]    (f32) */
  PAI_KERNEL_MATMUL = 7,     /* G48 c[i][j] = sum_k a[i][k]*b[k][j](f32) */
  PAI_KERNEL_MUL1D_U32 = 8,  /* T4B (reserved) */
  PAI_KERNEL_SUB1D_U32 = 9,  /* T4B (reserved) */
  PAI_KERNEL_RELU_U32  = 10, /* T4B (reserved) */
  PAI_KERNEL_CLIP_U32  = 11, /* T4B (reserved) */
  PAI_KERNEL_ADD2D_U32 = 12, /* T4B (reserved) */
  PAI_KERNEL_MATMUL_U32 = 13 /* T4B (reserved) */
};

/* Stable diagnostic name for a kernel id ("add1d", ...); returns
 * "unknown" for ids outside the vocabulary. */
const char *pai_op_kernel_id_str(pai_kernel_id_t id);

/* --- device capability flags -------------------------------------------- */

#define PAI_OP_CAP_CPU_REF    (1u << 0) /* cpu/reference implementation    */
#define PAI_OP_CAP_GPU_SERIAL (1u << 1) /* serial GPU kernel (G42-G48/T4B) */
#define PAI_OP_CAP_HOST_REF   (1u << 2) /* host-reference mirror kernel    */

/* --- descriptors --------------------------------------------------------- */

/* Reference implementations have heterogeneous signatures
 * (cpu/reference/ref_ops.h); pai_op_ref_fn is an opaque function
 * pointer that the dispatch site casts to the pai_ref_* prototype
 * matching the kernel id / op name. */
typedef pai_status_t (*pai_op_ref_fn)(void);

typedef struct pai_op_descriptor {
  const char     *name;       /* stable op name, e.g. "vecadd_f32"     */
  uint32_t        version;    /* descriptor version                    */
  uint32_t        num_inputs; /* max tensor arity (optional inputs like
                                 layernorm gamma/beta counted; eager
                                 paths may accept fewer)               */
  uint32_t        num_outputs;
  pai_dtype_t     in_dtype;   /* element dtype of inputs               */
  pai_dtype_t     out_dtype;
  uint32_t        caps;       /* PAI_OP_CAP_* bitset                   */
  pai_op_ref_fn   ref_fn;     /* pai_ref_* cast, or NULL               */
  pai_kernel_id_t kernel_id;  /* PAI_KERNEL_* or PAI_KERNEL_NONE       */
  uint32_t        reserved[2];
} pai_op_descriptor_t;

/* --- registry ------------------------------------------------------------- */

#define PAI_OP_REGISTRY_MAX_OPS 64u

typedef struct pai_op_registry {
  uint32_t version; /* registry format version (1) */
  uint32_t count;
  const pai_op_descriptor_t *ops[PAI_OP_REGISTRY_MAX_OPS];
} pai_op_registry_t;

pai_status_t pai_op_registry_init(pai_op_registry_t *reg);

/* Register a descriptor pointer (not copied; must outlive the registry). */
pai_status_t pai_op_registry_register(pai_op_registry_t *reg,
                                      const pai_op_descriptor_t *desc);

/* Populate the registry with the Phase-1 builtin op set. */
pai_status_t pai_op_registry_builtin(pai_op_registry_t *reg);

/* Lookup by stable name; NULL when absent. */
const pai_op_descriptor_t *pai_op_registry_lookup(const pai_op_registry_t *reg,
                                                  const char *name);

/* Lookup by kernel id; NULL when absent. */
const pai_op_descriptor_t *pai_op_registry_lookup_kernel(
    const pai_op_registry_t *reg, pai_kernel_id_t id);

uint32_t pai_op_registry_count(const pai_op_registry_t *reg);

/* True when the descriptor advertises every flag in `caps`. */
uint32_t pai_op_supports(const pai_op_descriptor_t *desc, uint32_t caps);

/* --- eager CPU fallback ---------------------------------------------------- */

/*
 * Execute `name` synchronously on the host through its pai_ref_*
 * implementation (the §37 oracle), accepting dense row-major
 * pai_tensor views. This is the Phase-1 CPU fallback for hosts without
 * a GPU and the differential oracle for device runs (T7).
 *
 * `params` carries scalar op parameters in a fixed per-op order:
 *   clip_f32  params[0]=lo, params[1]=hi (defaults 0.0, 1.0)
 *   scale_f32 params[0]=alpha            (default 1.0)
 *   rmsnorm_* params[0]=eps              (0 selects 1e-5)
 * Ops outside the eager dispatch set (rope, attention, layernorm,
 * quantized gemm, ...) return PAI_ERR_UNSUPPORTED; the static
 * executor (T5B) calls their pai_ref_* directly.
 */
pai_status_t pai_op_eval_f32(const pai_op_registry_t *reg, const char *name,
                             pai_tensor_t *const *inputs, uint32_t num_inputs,
                             pai_tensor_t *const *outputs, uint32_t num_outputs,
                             const float *params, uint32_t num_params);

#ifdef __cplusplus
}
#endif

#endif /* PAI_OP_H */
