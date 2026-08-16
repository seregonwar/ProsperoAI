/*
 * ProsperoAI — optimized CPU backend (whitepaper §6/§12, §37).
 * SIMD operators for the Zen 2-class PS5 CPU and host builds, with
 * runtime dispatch: portable -> SSE2 (x86-64 baseline) -> AVX2 + FMA
 * (when detected). The reference backend stays the correctness oracle:
 * every optimized op must be validated differentially against its
 * reference twin (§37); semantics and return codes are identical.
 * AVX2 paths use function-level target attributes (no -march needed).
 */

#ifndef PAI_CPU_OPTIMIZED_H
#define PAI_CPU_OPTIMIZED_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Detected feature flags. */
#define PAI_OPT_F_SSE2 (1u << 0)
#define PAI_OPT_F_AVX2 (1u << 1)
#define PAI_OPT_F_FMA  (1u << 2)

/* Run CPU feature detection (idempotent; also called lazily). */
void pai_opt_init(void);

/* Bitmask of pai_opt_features detected on this CPU. */
uint32_t pai_opt_features(void);

/* Name of the active implementation tier for diagnostics. */
const char *pai_opt_backend_name(void);

/* Element-wise ops (same semantics as pai_ref_*_f32) */

/* c[i] = a[i] + b[i] */
pai_status_t pai_opt_vecadd_f32(const float *a, const float *b, float *c,
                                uint64_t n);

/* c[i] = a[i] * b[i] */
pai_status_t pai_opt_vecmul_f32(const float *a, const float *b, float *c,
                                uint64_t n);

/* a[i] = alpha * a[i] (in place) */
pai_status_t pai_opt_scale_f32(float *a, float alpha, uint64_t n);

/* c[i] = max(a[i], 0) */
pai_status_t pai_opt_relu_f32(const float *a, float *c, uint64_t n);

/* Single-row softmax. */
pai_status_t pai_opt_softmax_f32(const float *a, float *c, uint64_t n);

/* Dense ops */

/* c = a * b, row-major: a[m x k], b[k x n], c[m x n]. */
pai_status_t pai_opt_gemm_f32(uint64_t m, uint64_t n, uint64_t k,
                              const float *a, const float *b, float *c);

/* y[m] = A[m,k] * x[k] (GEMM with n == 1, specialized). */
pai_status_t pai_opt_gemv_f32(uint64_t m, uint64_t k, const float *a,
                              const float *x, float *y);

/*
 * Quantized-weights GEMM (W8A16, §15): c = a * w_q * scale with the
 * weight matrix stored as per-group signed int8 plus one f32 scale per
 * (group, column):
 *
 *   w_q      k x n int8 (row-major: w_q[t * n + j])
 *   b_scales (ceil(k/group_size) * n) f32 scales; the scale for element
 *            (t, j) is b_scales[(t / group_size) * n + j]
 *   group_size  elements per group; 0 selects one group over the whole
 *            k axis (per-tensor, §15 block structure)
 *
 * c[i][j] = sum_t a[i][t] * w_q[t*n+j] * b_scales[(t/group_size)*n+j]
 */
pai_status_t pai_opt_gemm_w8_f32(uint64_t m, uint64_t n, uint64_t k,
                                 const float *a, const int8_t *w_q,
                                 const float *b_scales, uint64_t group_size,
                                 float *c);

#ifdef __cplusplus
}
#endif

#endif /* PAI_CPU_OPTIMIZED_H */
