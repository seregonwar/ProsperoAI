/*
 * ProsperoAI — CPU reference backend
 *
 * Deliberately simple reference implementations used as the correctness
 * oracle for optimized CPU/GPU backends (whitepaper §37).
 *
 * Everything here favors clarity over performance.
 */

#ifndef PAI_CPU_REFERENCE_H
#define PAI_CPU_REFERENCE_H

#include <pai/error.h>

#include <stdint.h>

/* c[i] = a[i] + b[i], element-wise, f32, contiguous buffers. */
pai_status_t pai_ref_vecadd_f32(const float *a, const float *b, float *c,
                                uint64_t n);

/* c[i] = alpha * a[i] (in place when a == c). */
pai_status_t pai_ref_scale_f32(float *a, float alpha, uint64_t n);

/* Classic naive GEMM: c = a * b, row-major, no blocking. */
pai_status_t pai_ref_gemm_f32(uint64_t m, uint64_t n, uint64_t k,
                              const float *a, const float *b, float *c);

/* memset-like fill of 16-byte patterns for GPU memset kernels. */
pai_status_t pai_ref_memset16(void *dst, const uint32_t pattern[4],
                              uint64_t blocks);

/* Compare two f32 buffers with relative+absolute tolerance. */
pai_status_t pai_ref_compare_f32(const float *a, const float *b, uint64_t n,
                                 float abs_tol, float rel_tol,
                                 uint64_t *out_first_mismatch);

#endif /* PAI_CPU_REFERENCE_H */
