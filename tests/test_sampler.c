#include "test.h"

#include <sampler.h>

#include <math.h>

TEST_MAIN_BEGIN()

{
  /* Greedy picks the argmax. */
  pai_sampler_t s;
  static const float logits[] = {-1.0f, 3.0f, 0.5f, -2.0f};
  uint32_t id = 99;

  pai_sampler_init(&s, 1);
  s.temperature = 0.0f;
  CHECK_EQ_INT(pai_sampler_sample_logits(&s, logits, 4, &id), PAI_OK);
  CHECK_EQ_UINT(id, 1);
}

{
  /* Determinism: same seed + same probs -> same draw. */
  pai_sampler_t a;
  pai_sampler_t b;
  static const float probs[] = {0.1f, 0.2f, 0.5f, 0.2f};
  uint32_t ida = 99;
  uint32_t idb = 99;
  uint32_t i;

  pai_sampler_init(&a, 0xDEADBEEF);
  pai_sampler_init(&b, 0xDEADBEEF);
  a.temperature = 1.0f;
  b.temperature = 1.0f;

  for (i = 0; i < 50; i++) {
    CHECK_EQ_INT(pai_sampler_sample_probs(&a, probs, 4, &ida), PAI_OK);
    CHECK_EQ_INT(pai_sampler_sample_probs(&b, probs, 4, &idb), PAI_OK);
    CHECK_EQ_UINT(ida, idb);
  }

  /* Different seed diverges (almost surely). */
  pai_sampler_init(&b, 0xFEEDFACE);
  b.temperature = 1.0f;
  {
    int diverged = 0;
    for (i = 0; i < 50; i++) {
      pai_sampler_sample_probs(&a, probs, 4, &ida);
      pai_sampler_sample_probs(&b, probs, 4, &idb);
      if (ida != idb) {
        diverged = 1;
        break;
      }
    }
    CHECK(diverged);
  }
}

{
  /* Top-k: only the top-k candidates can ever be drawn. */
  pai_sampler_t s;
  static const float probs[] = {0.01f, 0.60f, 0.39f, 0.00f};
  uint32_t id = 99;
  uint32_t i;

  pai_sampler_init(&s, 42);
  s.temperature = 1.0f;
  s.top_k = 2;
  for (i = 0; i < 200; i++) {
    CHECK_EQ_INT(pai_sampler_sample_probs(&s, probs, 4, &id), PAI_OK);
    CHECK(id == 1 || id == 2);
  }
}

{
  /* Top-p: a narrow nucleus still never picks outside it. */
  pai_sampler_t s;
  static const float probs[] = {0.95f, 0.03f, 0.01f, 0.01f};
  uint32_t id = 99;
  uint32_t i;

  pai_sampler_init(&s, 7);
  s.temperature = 1.0f;
  s.top_p = 0.9f;
  for (i = 0; i < 200; i++) {
    CHECK_EQ_INT(pai_sampler_sample_probs(&s, probs, 4, &id), PAI_OK);
    CHECK_EQ_UINT(id, 0);
  }
}

{
  /* Uniform probs with temperature: every id reachable over draws. */
  pai_sampler_t s;
  static const float probs[] = {0.25f, 0.25f, 0.25f, 0.25f};
  uint32_t seen = 0;
  uint32_t id = 99;
  uint32_t i;

  pai_sampler_init(&s, 1234);
  s.temperature = 1.0f;
  for (i = 0; i < 400; i++) {
    CHECK_EQ_INT(pai_sampler_sample_probs(&s, probs, 4, &id), PAI_OK);
    CHECK(id < 4);
    seen |= (1u << id);
  }
  CHECK_EQ_UINT(seen, 0xF);
}

{
  /* Validation. */
  pai_sampler_t s;
  static const float probs[] = {1.0f};
  uint32_t id = 0;

  pai_sampler_init(&s, 1);
  CHECK_EQ_INT(pai_sampler_sample_logits(&s, NULL, 1, &id),
               PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_sampler_sample_logits(&s, probs, 0, &id),
               PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_sampler_sample_probs(&s, probs, 1, NULL),
               PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
