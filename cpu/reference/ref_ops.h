/*
 * ProsperoAI — CPU reference backend.
 * Deliberately simple implementations; the correctness oracle for the
 * optimized CPU/GPU backends (whitepaper §37). Clarity over speed.
 */

#ifndef PAI_CPU_REFERENCE_H
#define PAI_CPU_REFERENCE_H

#include <pai/error.h>

#include <stdint.h>

/* c[i] = a[i] + b[i], element-wise, f32, contiguous buffers. */
pai_status_t pai_ref_vecadd_f32(const float *a, const float *b, float *c,
                                uint64_t n);

/* c[i] = a[i] * b[i], element-wise, f32, contiguous buffers. */
pai_status_t pai_ref_vecmul_f32(const float *a, const float *b, float *c,
                                uint64_t n);

/* c[i] = alpha * a[i] (in place when a == c). */
pai_status_t pai_ref_scale_f32(float *a, float alpha, uint64_t n);

/* c[i] = max(a[i], 0). */
pai_status_t pai_ref_relu_f32(const float *a, float *c, uint64_t n);

/* Row softmax over the flat buffer (single row). */
pai_status_t pai_ref_softmax_f32(const float *a, float *c, uint64_t n);

/*
 * RMS norm over the flat buffer (single row):
 *   out[i] = a[i] * rsqrt(mean(a^2) + eps)
 * eps 0 selects the default (1e-5).
 */
pai_status_t pai_ref_rmsnorm_f32(const float *a, float *c, uint64_t n,
                                 float eps);

/*
 * RMS norm with per-element gain (LLaMA-style, §9):
 *   out[i] = a[i] * gamma[i] * rsqrt(mean(a^2) + eps)
 * gamma must hold n elements. eps 0 selects the default (1e-5).
 */
pai_status_t pai_ref_rmsnorm_gamma_f32(const float *a, float *c, uint64_t n,
                                       const float *gamma, float eps);

/* c[i] = a[i] / (1 + exp(-a[i])) (SiLU / swish). */
pai_status_t pai_ref_silu_f32(const float *a, float *c, uint64_t n);

/*
 * Rotary position embeddings (LLaMA-style, §9). x is `rows` x `hd`
 * row-major with rows = seq * heads in position-major order (position
 * p occupies rows p*heads..(p+1)*heads, one row per head); the
 * position of row r is r / heads. cos_t/sin_t are `ctx` x `r2` tables
 * (r2 = rotary dim / 2); the first r2 pairs (i, i + r2) of each row
 * are rotated:
 *
 *   out[i]      = x[i] * cos_t[p][i] - x[i + r2] * sin_t[p][i]
 *   out[i + r2] = x[i] * sin_t[p][i] + x[i + r2] * cos_t[p][i]
 *
 * Elements beyond 2*r2 are copied unchanged. Requires hd >= 2*r2.
 * heads = 1 degenerates to the single-head case (position = r % seq).
 */
pai_status_t pai_ref_rope_f32(const float *x, uint64_t rows, uint64_t hd,
                              uint64_t seq, uint64_t heads,
                              const float *cos_t, const float *sin_t,
                              uint64_t r2, float *out);

/*
 * Causal multi-head self-attention over a sequence (§9). All four
 * buffers are position-major: q is seq x H x hd, k/v are
 * seq x HK x hd, out is seq x H x hd (position p occupies rows
 * p*H..(p+1)*H, head h at column block h*hd..(h+1)*hd). This matches
 * the [seq, n_embd] activation with per-head column blocks, so a flat
 * reshape to/from [seq, n_embd] interleaves heads correctly. Head h
 * reads KV head h * HK / H (grouped-query attention; H % HK must be
 * 0). Position p attends to positions 0..p (causal); scores are
 * scaled by 1/sqrt(hd) and softmaxed per row.
 */
pai_status_t pai_ref_attention_f32(uint64_t h, uint64_t hk, uint64_t seq,
                                   uint64_t hd, const float *q,
                                   const float *k, const float *v, float *out);


/*
 * Layer norm over the flat buffer (single row):
 *   out[i] = (a[i] - mean) / sqrt(var + eps) * gamma[i] + beta[i]
 * gamma/beta may be NULL (treated as 1/0). eps 0 selects 1e-5.
 */
pai_status_t pai_ref_layernorm_f32(const float *a, float *c, uint64_t n,
                                   const float *gamma, const float *beta,
                                   float eps);

/* c = concat(a, b): na elements of a followed by nb of b. */
pai_status_t pai_ref_concat_f32(const float *a, const float *b, float *c,
                                uint64_t na, uint64_t nb);

/* c[i] = a[i] (memcpy of n floats; c may alias a). */
pai_status_t pai_ref_copy_f32(const float *a, float *c, uint64_t n);

/* Classic naive GEMM: c = a * b, row-major, no blocking. */
pai_status_t pai_ref_gemm_f32(uint64_t m, uint64_t n, uint64_t k,
                              const float *a, const float *b, float *c);

/*
 * Quantized-weights GEMM (W8A16, §15): c = a * w_q * scale with the
 * weight matrix stored as per-group signed int8 plus one f32 scale per
 * (group, column):
 *
 *   w_q      k x n int8 (row-major: w_q[t * n + j])
 *   b_scales (ceil(k/group_size) * n) f32 scales; element (t, j) uses
 *            b_scales[(t / group_size) * n + j]
 *   group_size  elements per group; 0 = one group over k (per-tensor)
 *
 * c[i][j] = sum_t a[i][t] * w_q[t*n+j] * b_scales[(t/group_size)*n+j]
 */
pai_status_t pai_ref_gemm_w8_f32(uint64_t m, uint64_t n, uint64_t k,
                                 const float *a, const int8_t *w_q,
                                 const float *b_scales, uint64_t group_size,
                                 float *c);

/*
 * Quantized-weights GEMM with 4-bit weights (W4A16): the weight matrix
 * is packed two values per byte over the flat layout (element (t, j)
 * at flat index t*n+j -> byte [flat/2], low nibble when flat is even,
 * high nibble otherwise, sign-extended), with one f32 scale per
 * (group, column) exactly as in pai_ref_gemm_w8_f32.
 */
pai_status_t pai_ref_gemm_w4_f32(uint64_t m, uint64_t n, uint64_t k,
                                 const float *a, const int8_t *w_q4,
                                 const float *b_scales, uint64_t group_size,
                                 float *c);

/* memset-like fill of 16-byte patterns for GPU memset kernels. */
pai_status_t pai_ref_memset16(void *dst, const uint32_t pattern[4],
                              uint64_t blocks);

/* Compare two f32 buffers with relative+absolute tolerance. */
pai_status_t pai_ref_compare_f32(const float *a, const float *b, uint64_t n,
                                 float abs_tol, float rel_tol,
                                 uint64_t *out_first_mismatch);

#endif /* PAI_CPU_REFERENCE_H */
