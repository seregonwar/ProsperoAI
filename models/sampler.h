/*
 * ProsperoAI — sampler (whitepaper §3.2 CPU components)
 *
 * Deterministic sampling from a logits or probability vector with the
 * standard knobs: temperature, top-k and top-p (nucleus). Greedy
 * decoding is temperature == 0 (or top_k == 1). The PRNG is a seeded
 * xorshift64* so generation is reproducible for the same seed — the
 * deterministic behaviour tests and benchmarking rely on (§31).
 *
 * Sampling runs on the CPU by design (§3.2).
 */

#ifndef PAI_MODELS_SAMPLER_H
#define PAI_MODELS_SAMPLER_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pai_sampler {
  uint64_t seed;      /* PRNG state                                          */
  float    temperature; /* 0 = greedy argmax; else logits /= temperature     */
  uint32_t top_k;     /* 0 = disabled; keep the top-k candidates             */
  float    top_p;     /* 1.0 = disabled; nucleus mass threshold              */
} pai_sampler_t;

void pai_sampler_init(pai_sampler_t *sampler, uint64_t seed);

/* Next xorshift64* value (raw). */
uint64_t pai_sampler_next_u64(pai_sampler_t *sampler);

/* Uniform double in [0, 1). */
double pai_sampler_next_double(pai_sampler_t *sampler);

/*
 * Sample from logits. Applies temperature (unless 0), top-k (unless 0)
 * and top-p (unless >= 1.0), then softmax + categorical sampling.
 * `n` must be >= 1. Sets *out_id to the chosen index. Advances the
 * PRNG state in `sampler` (so the seed evolves across draws).
 */
pai_status_t pai_sampler_sample_logits(pai_sampler_t *sampler,
                                       const float *logits, uint32_t n,
                                       uint32_t *out_id);

/*
 * Sample from an already-normalized probability vector (sum 1). Applies
 * top-k / top-p only. Greedy (temperature 0) picks the argmax.
 * Advances the PRNG state in `sampler`.
 */
pai_status_t pai_sampler_sample_probs(pai_sampler_t *sampler,
                                      const float *probs, uint32_t n,
                                      uint32_t *out_id);

/* Stable name for diagnostics. */
const char *pai_sampler_mode_name(const pai_sampler_t *sampler);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_SAMPLER_H */
