#include "test.h"

#include <opt_ops.h>

#include <quantize.h>
#include <ref_ops.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float
rnd(void) {
  return (float)(rand() % 4000) / 1000.0f - 2.0f;
}

static void
fill_random(float *buf, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    buf[i] = rnd();
  }
}

static void
check_vecs(const float *got, const float *exp, uint64_t n, const char *what) {
  uint64_t first = 0;
  pai_status_t st = pai_ref_compare_f32(got, exp, n, 1e-5f, 1e-5f, &first);
  if (st != PAI_OK) {
    printf("  FAIL %s: first mismatch at %llu (got %f exp %f)\n", what,
           (unsigned long long)first, got[first], exp[first]);
    g_pai_test_failures++;
  }
}

TEST_MAIN_BEGIN()

{
  /* Feature detection is sane. */
  pai_opt_init();
  CHECK(pai_opt_backend_name() != NULL);
  CHECK((pai_opt_features() & (PAI_OPT_F_SSE2 | PAI_OPT_F_AVX2)) != 0 ||
        (pai_opt_features() == 0)); /* portable is valid too */
}

{
  /* Element-wise differential tests across assorted sizes. */
  const uint64_t sizes[] = {1, 3, 7, 8, 13, 64, 100};
  for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
    uint64_t n = sizes[s];
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    float *c = (float *)malloc((size_t)n * sizeof(float));
    float *e = (float *)malloc((size_t)n * sizeof(float));

    CHECK(a != NULL && b != NULL && c != NULL && e != NULL);
    fill_random(a, n);
    fill_random(b, n);

    CHECK_EQ_INT(pai_ref_vecadd_f32(a, b, e, n), PAI_OK);
    CHECK_EQ_INT(pai_opt_vecadd_f32(a, b, c, n), PAI_OK);
    check_vecs(c, e, n, "vecadd");

    CHECK_EQ_INT(pai_ref_vecmul_f32(a, b, e, n), PAI_OK);
    CHECK_EQ_INT(pai_opt_vecmul_f32(a, b, c, n), PAI_OK);
    check_vecs(c, e, n, "vecmul");

    memcpy(c, a, (size_t)n * sizeof(float));
    CHECK_EQ_INT(pai_ref_scale_f32(c, 0.5f, n), PAI_OK);
    CHECK_EQ_INT(pai_opt_scale_f32(a, 0.5f, n), PAI_OK);
    check_vecs(a, c, n, "scale");

    CHECK_EQ_INT(pai_ref_relu_f32(a, e, n), PAI_OK);
    CHECK_EQ_INT(pai_opt_relu_f32(a, c, n), PAI_OK);
    check_vecs(c, e, n, "relu");

    CHECK_EQ_INT(pai_ref_softmax_f32(a, e, n), PAI_OK);
    CHECK_EQ_INT(pai_opt_softmax_f32(a, c, n), PAI_OK);
    check_vecs(c, e, n, "softmax");

    free(a);
    free(b);
    free(c);
    free(e);
  }
}

{
  /* GEMM + GEMV differential tests. */
  const struct {
    uint64_t m, n, k;
  } shapes[] = {{1, 1, 1}, {1, 5, 7}, {3, 5, 7}, {8, 8, 8},
                {16, 3, 9}, {7, 9, 11}, {13, 1, 6}};

  for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
    uint64_t m = shapes[s].m;
    uint64_t n = shapes[s].n;
    uint64_t k = shapes[s].k;
    float *a = (float *)malloc((size_t)(m * k) * sizeof(float));
    float *b = (float *)malloc((size_t)(k * n) * sizeof(float));
    float *c = (float *)malloc((size_t)(m * n) * sizeof(float));
    float *e = (float *)malloc((size_t)(m * n) * sizeof(float));
    char what[32];

    CHECK(a != NULL && b != NULL && c != NULL && e != NULL);
    fill_random(a, m * k);
    fill_random(b, k * n);

    CHECK_EQ_INT(pai_ref_gemm_f32(m, n, k, a, b, e), PAI_OK);
    CHECK_EQ_INT(pai_opt_gemm_f32(m, n, k, a, b, c), PAI_OK);
    snprintf(what, sizeof(what), "gemm %llux%llux%llu",
             (unsigned long long)m, (unsigned long long)n,
             (unsigned long long)k);
    check_vecs(c, e, m * n, what);

    /* GEMV: y = A x == GEMM with n == 1. */
    {
      float *x = (float *)malloc((size_t)k * sizeof(float));
      float *y = (float *)malloc((size_t)m * sizeof(float));
      CHECK(x != NULL && y != NULL);
      fill_random(x, k);
      CHECK_EQ_INT(pai_opt_gemv_f32(m, k, a, x, y), PAI_OK);
      CHECK_EQ_INT(pai_ref_gemm_f32(m, 1, k, a, x, e), PAI_OK);
      check_vecs(y, e, m, "gemv");
      free(x);
      free(y);
    }

    free(a);
    free(b);
    free(c);
    free(e);
  }
}

{
  /* Quantized-weights GEMM differential (opt vs ref). */
  enum { M = 5, N = 7, K = 16 };
  float a[M * K];
  float w_f32[K * N];
  int8_t w_q[K * N];
  float scales[2 * N];
  float c1[M * N];
  float c2[M * N];
  pai_quant_scheme_t scheme;

  fill_random(a, M * K);
  fill_random(w_f32, K * N);
  scheme.bit_width = 8;
  scheme.is_signed = 1;
  scheme.group_size = 8;
  CHECK_EQ_INT(pai_quantize_f32(w_f32, K * N, &scheme, w_q, scales, NULL),
               PAI_OK);
  CHECK_EQ_INT(pai_ref_gemm_w8_f32(M, N, K, a, w_q, scales, 8, c1), PAI_OK);
  CHECK_EQ_INT(pai_opt_gemm_w8_f32(M, N, K, a, w_q, scales, 8, c2), PAI_OK);
  check_vecs(c2, c1, M * N, "gemm_w8");

  /* group_size 0 (per-tensor). */
  CHECK_EQ_INT(pai_opt_gemm_w8_f32(M, N, K, a, w_q, scales, 0, c2), PAI_OK);
  CHECK_EQ_INT(pai_ref_gemm_w8_f32(M, N, K, a, w_q, scales, 0, c1), PAI_OK);
  check_vecs(c2, c1, M * N, "gemm_w8 per-tensor");
}

{
  /* Invalid arguments. */
  float x[4] = {1, 2, 3, 4};
  float y[4];

  CHECK_EQ_INT(pai_opt_vecadd_f32(NULL, x, y, 4), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_opt_gemm_f32(0, 4, 4, x, x, y), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_opt_gemv_f32(4, 0, x, x, y), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_opt_softmax_f32(x, NULL, 4), PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
