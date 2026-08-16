#include "opt_ops.h"

#include <immintrin.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Feature detection                                                   */
/* ------------------------------------------------------------------ */

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||          \
    defined(_M_IX86)
#define PAI_OPT_X86 1
#else
#define PAI_OPT_X86 0
#endif

#if PAI_OPT_X86 && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h> /* __get_cpuid / __get_cpuid_count: no libgcc dep    */
#endif

#if defined(__clang__) || defined(__GNUC__)
#define PAI_OPT_GNUCC 1
#define PAI_OPT_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define PAI_OPT_GNUCC 0
#define PAI_OPT_TARGET_AVX2
#endif

static int g_opt_detected;
static uint32_t g_opt_features;

static void
pai_opt_detect(void) {
  uint32_t f = 0;

  if (g_opt_detected) {
    return;
  }
#if PAI_OPT_X86
  f |= PAI_OPT_F_SSE2; /* baseline on x86-64 / SSE2 on x86            */
#endif
#if PAI_OPT_GNUCC && PAI_OPT_X86
  {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
      /* AVX + FMA + OSXSAVE present, OS has enabled YMM state.      */
      if ((ecx & ((1u << 27) | (1u << 28) | (1u << 12))) ==
          ((1u << 27) | (1u << 28) | (1u << 12))) {
        unsigned int lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        if ((lo & 0x6u) == 0x6u) {
          unsigned int eax2, ebx2, ecx2, edx2;
          if (__get_cpuid_count(7, 0, &eax2, &ebx2, &ecx2, &edx2) &&
              (ebx2 & (1u << 5))) {
            f |= PAI_OPT_F_AVX2 | PAI_OPT_F_FMA;
          }
        }
      }
    }
  }
#endif
  g_opt_features = f;
  g_opt_detected = 1;
}

void
pai_opt_init(void) {
  pai_opt_detect();
}

uint32_t
pai_opt_features(void) {
  pai_opt_detect();
  return g_opt_features;
}

const char *
pai_opt_backend_name(void) {
  pai_opt_detect();
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return "avx2+fma";
  }
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return "sse2";
  }
  return "portable";
}

/* ------------------------------------------------------------------ */
/* Horizontal reductions                                               */
/* ------------------------------------------------------------------ */

#if PAI_OPT_GNUCC
PAI_OPT_TARGET_AVX2
static inline float
hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_add_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E)); /* [1,0,3,2]       */
  lo = _mm_add_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1)); /* [2,3,0,1]       */
  return _mm_cvtss_f32(lo);
}

PAI_OPT_TARGET_AVX2
static inline float
hmax256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_max_ps(lo, hi);
  lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x4E));
  lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0xB1));
  return _mm_cvtss_f32(lo);
}
#endif /* PAI_OPT_GNUCC */

static inline float
hsum128_ps(__m128 v) {
  v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0x4E));
  v = _mm_add_ps(v, _mm_shuffle_ps(v, v, 0xB1));
  return _mm_cvtss_f32(v);
}

static inline float
hmax128_ps(__m128 v) {
  v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0x4E));
  v = _mm_max_ps(v, _mm_shuffle_ps(v, v, 0xB1));
  return _mm_cvtss_f32(v);
}

/* ------------------------------------------------------------------ */
/* Portable (fallback) tier                                            */
/* ------------------------------------------------------------------ */

static pai_status_t
opt_vecadd_portable(const float *a, const float *b, float *c, uint64_t n) {
  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_vecmul_portable(const float *a, const float *b, float *c, uint64_t n) {
  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] * b[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_scale_portable(float *a, float alpha, uint64_t n) {
  if (!a && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < n; i++) {
    a[i] = alpha * a[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_relu_portable(const float *a, float *c, uint64_t n) {
  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < n; i++) {
    c[i] = a[i] > 0.0f ? a[i] : 0.0f;
  }
  return PAI_OK;
}

static pai_status_t
opt_softmax_portable(const float *a, float *c, uint64_t n) {
  float maxv;
  float sum = 0.0f;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }
  maxv = a[0];
  for (uint64_t i = 1; i < n; i++) {
    if (a[i] > maxv) {
      maxv = a[i];
    }
  }
  for (uint64_t i = 0; i < n; i++) {
    float e = expf(a[i] - maxv);
    c[i] = e;
    sum += e;
  }
  for (uint64_t i = 0; i < n; i++) {
    c[i] /= sum;
  }
  return PAI_OK;
}

static pai_status_t
opt_gemm_portable(uint64_t m, uint64_t n, uint64_t k, const float *a,
                  const float *b, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !b || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t l = 0; l < k; l++) {
        sum += a[i * k + l] * b[l * n + j];
      }
      c[i * n + j] = sum;
    }
  }
  return PAI_OK;
}

static pai_status_t
opt_gemm_w8_portable(uint64_t m, uint64_t n, uint64_t k, const float *a,
                     const int8_t *w_q, const float *b_scales,
                     uint64_t group_size, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !w_q || !b_scales || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  if (group_size == 0 || group_size > k) {
    group_size = k; /* per-tensor block (§15)                          */
  }
  for (uint64_t i = 0; i < m; i++) {
    for (uint64_t j = 0; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        float s = b_scales[(t / group_size) * n + j];
        sum += a[i * k + t] * (float)w_q[t * n + j] * s;
      }
      c[i * n + j] = sum;
    }
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* SSE2 tier (x86-64 baseline)                                         */
/* ------------------------------------------------------------------ */

static pai_status_t
opt_vecadd_sse2(const float *a, const float *b, float *c, uint64_t n) {
  uint64_t i = 0;

  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 4 <= n; i += 4) {
    __m128 av = _mm_loadu_ps(a + i);
    __m128 bv = _mm_loadu_ps(b + i);
    _mm_storeu_ps(c + i, _mm_add_ps(av, bv));
  }
  for (; i < n; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_vecmul_sse2(const float *a, const float *b, float *c, uint64_t n) {
  uint64_t i = 0;

  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 4 <= n; i += 4) {
    __m128 av = _mm_loadu_ps(a + i);
    __m128 bv = _mm_loadu_ps(b + i);
    _mm_storeu_ps(c + i, _mm_mul_ps(av, bv));
  }
  for (; i < n; i++) {
    c[i] = a[i] * b[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_scale_sse2(float *a, float alpha, uint64_t n) {
  __m128 al = _mm_set1_ps(alpha);
  uint64_t i = 0;

  if (!a && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 4 <= n; i += 4) {
    _mm_storeu_ps(a + i, _mm_mul_ps(_mm_loadu_ps(a + i), al));
  }
  for (; i < n; i++) {
    a[i] = alpha * a[i];
  }
  return PAI_OK;
}

static pai_status_t
opt_relu_sse2(const float *a, float *c, uint64_t n) {
  __m128 zero = _mm_setzero_ps();
  uint64_t i = 0;

  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 4 <= n; i += 4) {
    _mm_storeu_ps(c + i, _mm_max_ps(_mm_loadu_ps(a + i), zero));
  }
  for (; i < n; i++) {
    c[i] = a[i] > 0.0f ? a[i] : 0.0f;
  }
  return PAI_OK;
}

static pai_status_t
opt_softmax_sse2(const float *a, float *c, uint64_t n) {
  __m128 m = _mm_set1_ps(-INFINITY);
  __m128 s = _mm_setzero_ps();
  uint64_t i = 0;
  float maxv;
  float sum;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 4 <= n; i += 4) {
    m = _mm_max_ps(m, _mm_loadu_ps(a + i));
  }
  maxv = hmax128_ps(m);
  for (; i < n; i++) {
    if (a[i] > maxv) {
      maxv = a[i];
    }
  }
  for (i = 0; i + 4 <= n; i += 4) {
    __m128 e = _mm_set_ps(expf(a[i + 3] - maxv), expf(a[i + 2] - maxv),
                          expf(a[i + 1] - maxv), expf(a[i] - maxv));
    _mm_storeu_ps(c + i, e);
    s = _mm_add_ps(s, e);
  }
  sum = hsum128_ps(s);
  for (; i < n; i++) {
    float e = expf(a[i] - maxv);
    c[i] = e;
    sum += e;
  }
  for (i = 0; i < n; i++) {
    c[i] /= sum;
  }
  return PAI_OK;
}

static pai_status_t
opt_gemm_sse2(uint64_t m, uint64_t n, uint64_t k, const float *a,
              const float *b, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !b || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < m; i++) {
    const float *arow = a + i * k;
    float *crow = c + i * n;
    uint64_t j = 0;
    for (; j + 4 <= n; j += 4) {
      __m128 acc = _mm_setzero_ps();
      for (uint64_t l = 0; l < k; l++) {
        __m128 bj = _mm_loadu_ps(b + l * n + j);
        acc = _mm_add_ps(acc, _mm_mul_ps(_mm_set1_ps(arow[l]), bj));
      }
      _mm_storeu_ps(crow + j, acc);
    }
    for (; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t l = 0; l < k; l++) {
        sum += arow[l] * b[l * n + j];
      }
      crow[j] = sum;
    }
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* AVX2 + FMA tier (function-level target attribute; runtime-gated)    */
/* ------------------------------------------------------------------ */

#if PAI_OPT_GNUCC

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_vecadd_avx2(const float *a, const float *b, float *c, uint64_t n) {
  uint64_t i = 0;

  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(c + i, _mm256_add_ps(av, bv));
  }
  for (; i < n; i++) {
    c[i] = a[i] + b[i];
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_vecmul_avx2(const float *a, const float *b, float *c, uint64_t n) {
  uint64_t i = 0;

  if ((!a || !b || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    _mm256_storeu_ps(c + i, _mm256_mul_ps(av, bv));
  }
  for (; i < n; i++) {
    c[i] = a[i] * b[i];
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_scale_avx2(float *a, float alpha, uint64_t n) {
  __m256 al = _mm256_set1_ps(alpha);
  uint64_t i = 0;

  if (!a && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), al));
  }
  for (; i < n; i++) {
    a[i] = alpha * a[i];
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_relu_avx2(const float *a, float *c, uint64_t n) {
  __m256 zero = _mm256_setzero_ps();
  uint64_t i = 0;

  if ((!a || !c) && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(c + i, _mm256_max_ps(_mm256_loadu_ps(a + i), zero));
  }
  for (; i < n; i++) {
    c[i] = a[i] > 0.0f ? a[i] : 0.0f;
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_softmax_avx2(const float *a, float *c, uint64_t n) {
  __m256 m = _mm256_set1_ps(-INFINITY);
  __m256 s = _mm256_setzero_ps();
  uint64_t i = 0;
  float maxv;
  float sum;

  if (n == 0 || (!a || !c)) {
    return PAI_ERR_INVALID_ARG;
  }
  for (; i + 8 <= n; i += 8) {
    m = _mm256_max_ps(m, _mm256_loadu_ps(a + i));
  }
  maxv = hmax256_ps(m);
  for (; i < n; i++) {
    if (a[i] > maxv) {
      maxv = a[i];
    }
  }
  for (i = 0; i + 8 <= n; i += 8) {
    __m256 e = _mm256_set_ps(expf(a[i + 7] - maxv), expf(a[i + 6] - maxv),
                             expf(a[i + 5] - maxv), expf(a[i + 4] - maxv),
                             expf(a[i + 3] - maxv), expf(a[i + 2] - maxv),
                             expf(a[i + 1] - maxv), expf(a[i] - maxv));
    _mm256_storeu_ps(c + i, e);
    s = _mm256_add_ps(s, e);
  }
  sum = hsum256_ps(s);
  for (; i < n; i++) {
    float e = expf(a[i] - maxv);
    c[i] = e;
    sum += e;
  }
  for (i = 0; i < n; i++) {
    c[i] /= sum;
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_gemm_avx2(uint64_t m, uint64_t n, uint64_t k, const float *a,
              const float *b, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !b || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < m; i++) {
    const float *arow = a + i * k;
    float *crow = c + i * n;
    uint64_t j = 0;
    for (; j + 8 <= n; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (uint64_t l = 0; l < k; l++) {
        __m256 bj = _mm256_loadu_ps(b + l * n + j);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(arow[l]), bj, acc);
      }
      _mm256_storeu_ps(crow + j, acc);
    }
    for (; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t l = 0; l < k; l++) {
        sum += arow[l] * b[l * n + j];
      }
      crow[j] = sum;
    }
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_gemv_avx2(uint64_t m, uint64_t k, const float *a, const float *x,
              float *y) {
  if (m == 0 || k == 0 || !a || !x || !y) {
    return PAI_ERR_INVALID_ARG;
  }
  for (uint64_t i = 0; i < m; i++) {
    const float *arow = a + i * k;
    __m256 acc = _mm256_setzero_ps();
    uint64_t l = 0;
    float s;
    for (; l + 8 <= k; l += 8) {
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(arow + l),
                            _mm256_loadu_ps(x + l), acc);
    }
    s = hsum256_ps(acc);
    for (; l < k; l++) {
      s += arow[l] * x[l];
    }
    y[i] = s;
  }
  return PAI_OK;
}

PAI_OPT_TARGET_AVX2
static pai_status_t
opt_gemm_w8_avx2(uint64_t m, uint64_t n, uint64_t k, const float *a,
                 const int8_t *w_q, const float *b_scales,
                 uint64_t group_size, float *c) {
  if (m == 0 || n == 0 || k == 0 || !a || !w_q || !b_scales || !c) {
    return PAI_ERR_INVALID_ARG;
  }
  if (group_size == 0 || group_size > k) {
    group_size = k;
  }
  for (uint64_t i = 0; i < m; i++) {
    const float *arow = a + i * k;
    float *crow = c + i * n;
    uint64_t j = 0;
    for (; j + 8 <= n; j += 8) {
      __m256 acc = _mm256_setzero_ps();
      for (uint64_t t0 = 0; t0 < k; t0 += group_size) {
        uint64_t t1 = t0 + group_size < k ? t0 + group_size : k;
        __m256 sg = _mm256_loadu_ps(b_scales + (t0 / group_size) * n + j);
        for (uint64_t t = t0; t < t1; t++) {
          __m128i b8 =
              _mm_loadl_epi64((const __m128i *)(const void *)(w_q + t * n + j));
          __m256 bf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b8));
          acc = _mm256_fmadd_ps(_mm256_set1_ps(arow[t]),
                                _mm256_mul_ps(bf, sg), acc);
        }
      }
      _mm256_storeu_ps(crow + j, acc);
    }
    for (; j < n; j++) {
      float sum = 0.0f;
      for (uint64_t t = 0; t < k; t++) {
        float s = b_scales[(t / group_size) * n + j];
        sum += arow[t] * (float)w_q[t * n + j] * s;
      }
      crow[j] = sum;
    }
  }
  return PAI_OK;
}

#endif /* PAI_OPT_GNUCC */

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

pai_status_t
pai_opt_vecadd_f32(const float *a, const float *b, float *c, uint64_t n) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_vecadd_avx2(a, b, c, n);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_vecadd_sse2(a, b, c, n);
  }
  return opt_vecadd_portable(a, b, c, n);
}

pai_status_t
pai_opt_vecmul_f32(const float *a, const float *b, float *c, uint64_t n) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_vecmul_avx2(a, b, c, n);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_vecmul_sse2(a, b, c, n);
  }
  return opt_vecmul_portable(a, b, c, n);
}

pai_status_t
pai_opt_scale_f32(float *a, float alpha, uint64_t n) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_scale_avx2(a, alpha, n);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_scale_sse2(a, alpha, n);
  }
  return opt_scale_portable(a, alpha, n);
}

pai_status_t
pai_opt_relu_f32(const float *a, float *c, uint64_t n) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_relu_avx2(a, c, n);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_relu_sse2(a, c, n);
  }
  return opt_relu_portable(a, c, n);
}

pai_status_t
pai_opt_softmax_f32(const float *a, float *c, uint64_t n) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_softmax_avx2(a, c, n);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_softmax_sse2(a, c, n);
  }
  return opt_softmax_portable(a, c, n);
}

pai_status_t
pai_opt_gemm_f32(uint64_t m, uint64_t n, uint64_t k, const float *a,
                 const float *b, float *c) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_gemm_avx2(m, n, k, a, b, c);
  }
#endif
  if (g_opt_features & PAI_OPT_F_SSE2) {
    return opt_gemm_sse2(m, n, k, a, b, c);
  }
  return opt_gemm_portable(m, n, k, a, b, c);
}

pai_status_t
pai_opt_gemv_f32(uint64_t m, uint64_t k, const float *a, const float *x,
                 float *y) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_gemv_avx2(m, k, a, x, y);
  }
#endif
  /* GEMV is GEMM with n == 1. */
  return pai_opt_gemm_f32(m, 1, k, a, x, y);
}

pai_status_t
pai_opt_gemm_w8_f32(uint64_t m, uint64_t n, uint64_t k, const float *a,
                    const int8_t *w_q, const float *b_scales,
                    uint64_t group_size, float *c) {
  pai_opt_detect();
#if PAI_OPT_GNUCC
  if (g_opt_features & PAI_OPT_F_AVX2) {
    return opt_gemm_w8_avx2(m, n, k, a, w_q, b_scales, group_size, c);
  }
#endif
  return opt_gemm_w8_portable(m, n, k, a, w_q, b_scales, group_size, c);
}
