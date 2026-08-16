#include "test.h"

#include <pai/op.h>
#include <pai/tensor.h>

#include <math.h>
#include <string.h>

/* Registry of the Phase-1 builtin op set: lookups, capability flags,
 * kernel-id vocabulary and the eager CPU fallback (T5A). */

static void
tensor1(pai_tensor_t *t, float *data, uint64_t n) {
  uint64_t shape[1] = {n};
  CHECK(pai_tensor_init(t, PAI_DTYPE_F32, 1, shape, data) == PAI_OK);
}

static void
tensor2(pai_tensor_t *t, float *data, uint64_t m, uint64_t k) {
  uint64_t shape[2] = {m, k};
  CHECK(pai_tensor_init(t, PAI_DTYPE_F32, 2, shape, data) == PAI_OK);
}

TEST_MAIN_BEGIN()

{
  /* builtin registry populated with the Phase-1 op set */
  pai_op_registry_t reg;
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  CHECK_EQ_UINT(reg.version, 1);
  CHECK(pai_op_registry_count(&reg) >= 24);
}

{
  /* lookup by name */
  pai_op_registry_t reg;
  const pai_op_descriptor_t *d;
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);

  d = pai_op_registry_lookup(&reg, "vecadd_f32");
  CHECK(d != NULL);
  CHECK_EQ_UINT(d->num_inputs, 2);
  CHECK_EQ_UINT(d->num_outputs, 1);
  CHECK_EQ_INT(d->in_dtype, PAI_DTYPE_F32);
  CHECK_EQ_INT(d->out_dtype, PAI_DTYPE_F32);
  CHECK_EQ_INT(d->kernel_id, PAI_KERNEL_ADD1D);
  CHECK(pai_op_supports(d, PAI_OP_CAP_CPU_REF));
  CHECK(pai_op_supports(d, PAI_OP_CAP_GPU_SERIAL));
  CHECK(pai_op_supports(d, PAI_OP_CAP_HOST_REF));
  CHECK(pai_op_supports(d, PAI_OP_CAP_CPU_REF | PAI_OP_CAP_GPU_SERIAL));
  CHECK(!pai_op_supports(d, 1u << 31));

  d = pai_op_registry_lookup(&reg, "matmul_f32");
  CHECK(d != NULL);
  CHECK_EQ_INT(d->kernel_id, PAI_KERNEL_MATMUL);

  /* ref-only op: no GPU kernel */
  d = pai_op_registry_lookup(&reg, "softmax_f32");
  CHECK(d != NULL);
  CHECK_EQ_INT(d->kernel_id, PAI_KERNEL_NONE);
  CHECK(pai_op_supports(d, PAI_OP_CAP_CPU_REF));
  CHECK(!pai_op_supports(d, PAI_OP_CAP_GPU_SERIAL));

  CHECK(pai_op_registry_lookup(&reg, "no_such_op") == NULL);
  CHECK(pai_op_registry_lookup(&reg, NULL) == NULL);
}

{
  /* lookup by kernel id, including the reserved T4B vocabulary */
  pai_op_registry_t reg;
  const pai_op_descriptor_t *d;
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);

  d = pai_op_registry_lookup_kernel(&reg, PAI_KERNEL_ADD1D);
  CHECK(d != NULL);
  CHECK(strcmp(d->name, "vecadd_f32") == 0);

  d = pai_op_registry_lookup_kernel(&reg, PAI_KERNEL_BIASADD);
  CHECK(d != NULL);
  CHECK(strcmp(d->name, "biasadd_f32") == 0);

  /* T4B reserved integer ids resolve before the kernels land */
  d = pai_op_registry_lookup_kernel(&reg, PAI_KERNEL_MUL1D_U32);
  CHECK(d != NULL);
  CHECK(strcmp(d->name, "mul1d_u32") == 0);
  CHECK(d->ref_fn == NULL);

  CHECK(pai_op_registry_lookup_kernel(&reg, PAI_KERNEL_NONE) == NULL);
  CHECK(pai_op_registry_lookup_kernel(&reg, (pai_kernel_id_t)9999) == NULL);
}

{
  /* kernel id diagnostic names */
  CHECK(strcmp(pai_op_kernel_id_str(PAI_KERNEL_ADD1D), "add1d") == 0);
  CHECK(strcmp(pai_op_kernel_id_str(PAI_KERNEL_CLIP), "clip") == 0);
  CHECK(strcmp(pai_op_kernel_id_str(PAI_KERNEL_MATMUL_U32), "matmul_u32") ==
        0);
  CHECK(strcmp(pai_op_kernel_id_str((pai_kernel_id_t)12345), "unknown") == 0);
}

{
  /* custom registration */
  pai_op_registry_t reg;
  pai_op_descriptor_t mine = {.name = "test_op",
                              .version = 2,
                              .num_inputs = 1,
                              .num_outputs = 1,
                              .in_dtype = PAI_DTYPE_F32,
                              .out_dtype = PAI_DTYPE_F32,
                              .caps = PAI_OP_CAP_CPU_REF,
                              .ref_fn = NULL,
                              .kernel_id = PAI_KERNEL_NONE};
  uint32_t before;
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  before = pai_op_registry_count(&reg);
  CHECK(pai_op_registry_register(&reg, &mine) == PAI_OK);
  CHECK_EQ_UINT(pai_op_registry_count(&reg), before + 1);
  CHECK(pai_op_registry_lookup(&reg, "test_op") == &mine);
  CHECK_EQ_UINT(pai_op_registry_lookup(&reg, "test_op")->version, 2);
  CHECK(pai_op_registry_register(&reg, NULL) == PAI_ERR_INVALID_ARG);
}

{
  /* eager vecadd */
  pai_op_registry_t reg;
  float a[4] = {1, 2, 3, 4};
  float b[4] = {10, 20, 30, 40};
  float c[4];
  pai_tensor_t ta, tb, tc;
  pai_tensor_t *in[2] = {&ta, &tb};
  pai_tensor_t *out[1] = {&tc};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor1(&ta, a, 4);
  tensor1(&tb, b, 4);
  tensor1(&tc, c, 4);
  CHECK(pai_op_eval_f32(&reg, "vecadd_f32", in, 2, out, 1, NULL, 0) ==
        PAI_OK);
  for (int i = 0; i < 4; i++) {
    CHECK(fabsf(c[i] - (a[i] + b[i])) < 1e-6f);
  }
}

{
  /* eager relu + clip with params */
  pai_op_registry_t reg;
  float a[5] = {-3, -0.5f, 0, 0.5f, 7};
  float r[5], cl[5];
  float p[2] = {0.0f, 1.0f};
  pai_tensor_t ta, tr, tcl;
  pai_tensor_t *in[1] = {&ta};
  pai_tensor_t *ro[1] = {&tr};
  pai_tensor_t *co[1] = {&tcl};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor1(&ta, a, 5);
  tensor1(&tr, r, 5);
  tensor1(&tcl, cl, 5);
  CHECK(pai_op_eval_f32(&reg, "relu_f32", in, 1, ro, 1, NULL, 0) == PAI_OK);
  CHECK(fabsf(r[0] - 0.0f) < 1e-6f);
  CHECK(fabsf(r[1] - 0.0f) < 1e-6f);
  CHECK(fabsf(r[2] - 0.0f) < 1e-6f);
  CHECK(fabsf(r[3] - 0.5f) < 1e-6f);
  CHECK(fabsf(r[4] - 7.0f) < 1e-6f);
  CHECK(pai_op_eval_f32(&reg, "clip_f32", in, 1, co, 1, p, 2) == PAI_OK);
  CHECK(fabsf(cl[0] - 0.0f) < 1e-6f);
  CHECK(fabsf(cl[1] - 0.0f) < 1e-6f);
  CHECK(fabsf(cl[2] - 0.0f) < 1e-6f);
  CHECK(fabsf(cl[3] - 0.5f) < 1e-6f);
  CHECK(fabsf(cl[4] - 1.0f) < 1e-6f);
  /* defaults lo=0, hi=1 when params absent */
  CHECK(pai_op_eval_f32(&reg, "clip_f32", in, 1, co, 1, NULL, 0) == PAI_OK);
  CHECK(fabsf(cl[4] - 1.0f) < 1e-6f);
}

{
  /* eager biasadd: 2x3 broadcast over the last dim */
  pai_op_registry_t reg;
  float a[6] = {1, 2, 3, 4, 5, 6};
  float bias[3] = {10, 20, 30};
  float c[6];
  pai_tensor_t ta, tb, tc;
  pai_tensor_t *in[2] = {&ta, &tb};
  pai_tensor_t *out[1] = {&tc};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor2(&ta, a, 2, 3);
  tensor1(&tb, bias, 3);
  tensor2(&tc, c, 2, 3);
  CHECK(pai_op_eval_f32(&reg, "biasadd_f32", in, 2, out, 1, NULL, 0) ==
        PAI_OK);
  CHECK(fabsf(c[0] - 11.0f) < 1e-6f);
  CHECK(fabsf(c[1] - 22.0f) < 1e-6f);
  CHECK(fabsf(c[2] - 33.0f) < 1e-6f);
  CHECK(fabsf(c[3] - 14.0f) < 1e-6f);
  CHECK(fabsf(c[4] - 25.0f) < 1e-6f);
  CHECK(fabsf(c[5] - 36.0f) < 1e-6f);
}

{
  /* eager matmul: 2x2 * 2x3 */
  pai_op_registry_t reg;
  float a[4] = {1, 2, 3, 4};
  float b[6] = {1, 0, 2, 0, 1, 2}; /* 2x3 */
  float c[6];
  pai_tensor_t ta, tb, tc;
  pai_tensor_t *in[2] = {&ta, &tb};
  pai_tensor_t *out[1] = {&tc};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor2(&ta, a, 2, 2);
  tensor2(&tb, b, 2, 3);
  tensor2(&tc, c, 2, 3);
  CHECK(pai_op_eval_f32(&reg, "matmul_f32", in, 2, out, 1, NULL, 0) ==
        PAI_OK);
  CHECK(fabsf(c[0] - 1.0f) < 1e-6f);  /* 1*1 + 2*0 */
  CHECK(fabsf(c[1] - 2.0f) < 1e-6f);  /* 1*0 + 2*1 */
  CHECK(fabsf(c[2] - 6.0f) < 1e-6f);  /* 1*2 + 2*2 */
  CHECK(fabsf(c[3] - 3.0f) < 1e-6f);  /* 3*1 + 4*0 */
  CHECK(fabsf(c[4] - 4.0f) < 1e-6f);  /* 3*0 + 4*1 */
  CHECK(fabsf(c[5] - 14.0f) < 1e-6f); /* 3*2 + 4*2 */
}

{
  /* eager gemv: 3x2 * x */
  pai_op_registry_t reg;
  float a[6] = {1, 2, 3, 4, 5, 6};
  float x[2] = {10, 100};
  float y[3];
  pai_tensor_t ta, tx, ty;
  pai_tensor_t *in[2] = {&ta, &tx};
  pai_tensor_t *out[1] = {&ty};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor2(&ta, a, 3, 2);
  tensor1(&tx, x, 2);
  tensor1(&ty, y, 3);
  CHECK(pai_op_eval_f32(&reg, "gemv_f32", in, 2, out, 1, NULL, 0) ==
        PAI_OK);
  CHECK(fabsf(y[0] - 210.0f) < 1e-6f);
  CHECK(fabsf(y[1] - 430.0f) < 1e-6f);
  CHECK(fabsf(y[2] - 650.0f) < 1e-6f);
}

{
  /* eager error paths */
  pai_op_registry_t reg;
  float a[4] = {1, 2, 3, 4};
  float b[4] = {1, 1, 1, 1};
  float c[4];
  pai_tensor_t ta, tb, tc;
  pai_tensor_t *in[2] = {&ta, &tb};
  pai_tensor_t *out[1] = {&tc};
  CHECK(pai_op_registry_builtin(&reg) == PAI_OK);
  tensor1(&ta, a, 4);
  tensor1(&tb, b, 4);
  tensor1(&tc, c, 4);

  /* unknown op */
  CHECK(pai_op_eval_f32(&reg, "nope_f32", in, 2, out, 1, NULL, 0) ==
        PAI_ERR_UNSUPPORTED);
  /* wrong arity */
  CHECK(pai_op_eval_f32(&reg, "vecadd_f32", in, 1, out, 1, NULL, 0) ==
        PAI_ERR_INVALID_ARG);
  /* dtype mismatch: relu expects f32, feed u32 view */
  {
    uint32_t u32[4] = {1, 2, 3, 4};
    uint64_t sh[1] = {4};
    pai_tensor_t tu;
    pai_tensor_t *ui[1] = {&tu};
    CHECK(pai_tensor_init(&tu, PAI_DTYPE_U32, 1, sh, u32) == PAI_OK);
    CHECK(pai_op_eval_f32(&reg, "relu_f32", ui, 1, out, 1, NULL, 0) ==
          PAI_ERR_MISMATCH);
  }
  /* reserved T4B op has no ref fn yet */
  CHECK(pai_op_eval_f32(&reg, "mul1d_u32", in, 2, out, 1, NULL, 0) ==
        PAI_ERR_UNSUPPORTED);
  /* op registered with a ref fn but outside the eager dispatch set */
  {
    float g[4] = {1, 1, 1, 1};
    float beta[4] = {0, 0, 0, 0};
    pai_tensor_t tg, tb2;
    pai_tensor_t *li[3] = {&ta, &tg, &tb2};
    tensor1(&tg, g, 4);
    tensor1(&tb2, beta, 4);
    CHECK(pai_op_eval_f32(&reg, "layernorm_f32", li, 3, out, 1, NULL, 0) ==
          PAI_ERR_UNSUPPORTED);
  }
}

TEST_MAIN_END()
