#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void
pai_sampler_init(pai_sampler_t *sampler, uint64_t seed) {
  if (sampler == NULL) {
    return;
  }
  memset(sampler, 0, sizeof(*sampler));
  sampler->seed = seed != 0 ? seed : 0x9E3779B97F4A7C15ull;
  sampler->temperature = 0.0f; /* greedy by default */
  sampler->top_k = 0;
  sampler->top_p = 1.0f;
}

uint64_t
pai_sampler_next_u64(pai_sampler_t *sampler) {
  uint64_t x;

  if (sampler == NULL) {
    return 0;
  }
  x = sampler->seed;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  sampler->seed = x;
  return x * 0x2545F4914F6CDD1Dull;
}

double
pai_sampler_next_double(pai_sampler_t *sampler) {
  return (double)(pai_sampler_next_u64(sampler) >> 11) *
         (1.0 / 9007199254740992.0);
}

const char *
pai_sampler_mode_name(const pai_sampler_t *sampler) {
  if (sampler == NULL) {
    return "?";
  }
  if (sampler->temperature == 0.0f) {
    return "greedy";
  }
  if (sampler->top_p < 1.0f) {
    return "nucleus";
  }
  return "temperature";
}

/* Index sort (descending) of the first n values. */
static void
sort_desc(const float *vals, uint32_t n, uint32_t *order) {
  uint32_t i;
  uint32_t j;

  for (i = 0; i < n; i++) {
    order[i] = i;
  }
  for (i = 0; i < n; i++) {
    for (j = i + 1; j < n; j++) {
      if (vals[order[j]] > vals[order[i]]) {
        uint32_t tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }
    }
  }
}

pai_status_t
pai_sampler_sample_logits(pai_sampler_t *sampler, const float *logits,
                          uint32_t n, uint32_t *out_id) {
  float *w;
  float maxv;
  float sum;
  uint32_t i;
  uint32_t chosen = 0;

  if (sampler == NULL || logits == NULL || n == 0 || out_id == NULL ||
      sampler->temperature < 0.0f) {
    return PAI_ERR_INVALID_ARG;
  }

  /* Greedy shortcut: no randomness involved. */
  if (sampler->temperature == 0.0f) {
    float best = logits[0];
    chosen = 0;
    for (i = 1; i < n; i++) {
      if (logits[i] > best) {
        best = logits[i];
        chosen = i;
      }
    }
    *out_id = chosen;
    return PAI_OK;
  }

  w = (float *)malloc((size_t)n * sizeof(float));
  if (w == NULL) {
    return PAI_ERR_NOMEM;
  }

  /* Temperature-scaled logits -> softmax. */
  maxv = logits[0];
  for (i = 1; i < n; i++) {
    if (logits[i] > maxv) {
      maxv = logits[i];
    }
  }
  sum = 0.0f;
  for (i = 0; i < n; i++) {
    float v = (logits[i] - maxv) / sampler->temperature;
    w[i] = expf(v < -80.0f ? -80.0f : v);
    sum += w[i];
  }
  for (i = 0; i < n; i++) {
    w[i] /= sum;
  }

  pai_sampler_sample_probs(sampler, w, n, &chosen);
  free(w);
  *out_id = chosen;
  return PAI_OK;
}

pai_status_t
pai_sampler_sample_probs(pai_sampler_t *sampler, const float *probs,
                         uint32_t n, uint32_t *out_id) {
  float *w;
  float *cdf;
  uint32_t *order;
  uint32_t keep;
  double r;
  uint32_t i;
  uint32_t chosen = 0;

  if (sampler == NULL || probs == NULL || n == 0 || out_id == NULL ||
      sampler->temperature < 0.0f) {
    return PAI_ERR_INVALID_ARG;
  }

  /* Greedy: argmax. */
  if (sampler->temperature == 0.0f) {
    float best = probs[0];
    for (i = 1; i < n; i++) {
      if (probs[i] > best) {
        best = probs[i];
        chosen = i;
      }
    }
    *out_id = chosen;
    return PAI_OK;
  }

  w = (float *)malloc((size_t)n * sizeof(float));
  cdf = (float *)malloc((size_t)n * sizeof(float));
  order = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
  if (w == NULL || cdf == NULL || order == NULL) {
    free(w);
    free(cdf);
    free(order);
    return PAI_ERR_NOMEM;
  }

  memcpy(w, probs, (size_t)n * sizeof(float));
  sort_desc(w, n, order);

  /* top-k: keep the first `keep` in descending order. */
  keep = n;
  if (sampler->top_k > 0 && sampler->top_k < keep) {
    keep = sampler->top_k;
  }

  /* top-p: keep the smallest prefix whose mass >= top_p. */
  if (sampler->top_p < 1.0f) {
    float acc = 0.0f;
    uint32_t k = 0;
    while (k < keep && acc < sampler->top_p) {
      acc += w[order[k]];
      k++;
    }
    if (k > 0) {
      keep = k;
    }
  }

  /* Renormalize the kept prefix. */
  {
    float sum = 0.0f;
    for (i = 0; i < keep; i++) {
      sum += w[order[i]];
    }
    if (sum <= 0.0f) {
      free(w);
      free(cdf);
      free(order);
      return PAI_ERR_INVALID_ARG;
    }
    cdf[0] = w[order[0]] / sum;
    for (i = 1; i < keep; i++) {
      cdf[i] = cdf[i - 1] + w[order[i]] / sum;
    }
  }

  r = pai_sampler_next_double(sampler);
  for (i = 0; i < keep; i++) {
    if (r <= (double)cdf[i]) {
      chosen = order[i];
      break;
    }
  }
  if (i == keep) {
    chosen = order[keep - 1];
  }

  free(w);
  free(cdf);
  free(order);
  *out_id = chosen;
  return PAI_OK;
}
