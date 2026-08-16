#include "test.h"

#include <importer.h>

#include <model.h>
#include <pai/pai.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define V 4 /* vocab (tokens a..d) */
#define H 8 /* hidden */

static const char k_desc[] =
    "name tiny-lm\n"
    "family llm\n"
    "context 16\n"
    "layers 1\n"
    "kv_bytes 8\n"
    "vocab 4\n"
    "\n"
    "token 0 a\n"
    "token 1 b\n"
    "token 2 c\n"
    "token 3 d\n"
    "\n"
    "value 1 f32 [4] input\n"
    "value 2 f32 [8,4] param weights=tok_embed.bin\n"
    "value 3 f32 [8] activation\n"
    "value 4 f32 [4,8] param weights=w_out.bin\n"
    "value 5 f32 [4] activation\n"
    "value 6 f32 [4] output\n"
    "\n"
    "op 1 gemv 2, 1 -> 3\n" /* spaced list: 2 tokens -> multi-token */
    "op 2 gemv 4, 3 -> 5\n"
    "op 3 softmax 5 -> 6\n";

/* Weights chosen so token t -> token (t+1)%V under greedy decoding, and
 * exactly representable in q8 AND q4 (0 and 100 / 0 and 2). */
static void
build_weights(float *w_embed, float *w_out) {
  memset(w_embed, 0, (size_t)(H * V) * sizeof(float));
  for (uint32_t i = 0; i < V && i < H; i++) {
    w_embed[i * V + i] = 2.0f;
  }
  memset(w_out, 0, (size_t)(V * H) * sizeof(float));
  for (uint32_t i = 0; i < V; i++) {
    w_out[((i + 1) % V) * H + i] = 100.0f;
  }
}

static int
write_file(const char *path, const void *data, size_t nbytes) {
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    return -1;
  }
  if (fwrite(data, 1, nbytes, f) != nbytes) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

static char gen_text[256];
static uint32_t gen_text_n;

static void
on_token(const char *token, void *user) {
  (void)user;
  if (gen_text_n + (uint32_t)strlen(token) < sizeof(gen_text)) {
    gen_text_n += (uint32_t)strlen(token);
    strcat(gen_text, token);
  }
}

/* Import a desc with the given quant option and run a greedy chain. */
static int
import_and_generate(const pai_import_options_t *opts, const char *expect,
                    uint32_t expect_n) {
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_model_t *model = NULL;
  pai_session_t *session = NULL;
  pai_status_t st;

  st = pai_import_model("tiny_desc.txt", opts, &blob, &nbytes);
  if (st != PAI_OK) {
    printf("  FAIL import (quant=%s): %s\n",
           opts != NULL && opts->quant != NULL ? opts->quant : "none",
           pai_status_str(st));
    g_pai_test_failures++;
    return -1;
  }
  st = pai_model_open_blob(blob, nbytes, &model);
  if (st != PAI_OK) {
    printf("  FAIL open (quant=%s): %s\n",
           opts != NULL && opts->quant != NULL ? opts->quant : "none",
           pai_status_str(st));
    g_pai_test_failures++;
    free(blob);
    return -1;
  }
  CHECK(strcmp(pai_model_name(model), "tiny-lm") == 0);

  st = pai_session_init(model, &session);
  if (st != PAI_OK) {
    printf("  FAIL session (quant=%s): %s\n",
           opts != NULL && opts->quant != NULL ? opts->quant : "none",
           pai_status_str(st));
    g_pai_test_failures++;
    pai_model_close(model);
    free(blob);
    return -1;
  }

  /* Quantized containers: the session region must hold the exact
   * dequantized weights (2.0 / 100.0 are exactly representable in both
   * q8 and q4), proving the packed layout + scales are read back
   * correctly, not just the argmax-preserving chain. */
  if (opts != NULL && opts->quant != NULL) {
    const pai_graph_mem_plan_t *mp = &model->mem_plan;
    const float *r2 = (const float *)(const void *)(session->region +
                                                    pai_graph_mem_plan_offset(
                                                        mp, 2));
    const float *r4 = (const float *)(const void *)(session->region +
                                                    pai_graph_mem_plan_offset(
                                                        mp, 4));
    int good = 1;
    for (uint32_t i = 0; i < (uint32_t)(H * V); i++) {
      float exp2 = (i == 0 || i == 5 || i == 10 || i == 15) ? 2.0f : 0.0f;
      if (fabsf(r2[i] - exp2) > 1e-4f) {
        good = 0;
      }
    }
    for (uint32_t i = 0; i < (uint32_t)(V * H); i++) {
      float exp4 = (i == 3 || i == 8 || i == 17 || i == 26) ? 100.0f : 0.0f;
      if (fabsf(r4[i] - exp4) > 1e-4f) {
        good = 0;
      }
    }
    if (!good) {
      printf("  FAIL dequantized weights mismatch (quant=%s)\n",
             opts->quant);
      g_pai_test_failures++;
    }
  }

  CHECK_EQ_INT(pai_session_set_generation(session, 8, 0), PAI_OK);

  memset(gen_text, 0, sizeof(gen_text));
  gen_text_n = 0;
  st = pai_session_generate(session, "a", on_token, NULL);
  if (st != PAI_OK) {
    printf("  FAIL generate (quant=%s): %s\n",
           opts != NULL && opts->quant != NULL ? opts->quant : "none",
           pai_status_str(st));
    g_pai_test_failures++;
  } else {
    CHECK_EQ_UINT(gen_text_n, expect_n);
    CHECK(strcmp(gen_text, expect) == 0);
  }

  pai_session_destroy(session);
  pai_model_close(model);
  free(blob);
  return 0;
}

TEST_MAIN_BEGIN()

{
  float w_embed[H * V];
  float w_out[V * H];
  pai_import_options_t opts;
  FILE *f;

  /* Fixture: desc + raw f32 weight files in the working directory. */
  CHECK(write_file("tiny_desc.txt", k_desc, sizeof(k_desc) - 1) == 0);
  build_weights(w_embed, w_out);
  f = fopen("tok_embed.bin", "wb");
  CHECK(f != NULL);
  fwrite(w_embed, sizeof(float), H * V, f);
  fclose(f);
  f = fopen("w_out.bin", "wb");
  CHECK(f != NULL);
  fwrite(w_out, sizeof(float), V * H, f);
  fclose(f);

  /* Unquantized import: greedy chain a->b->c->d->a... */
  memset(&opts, 0, sizeof(opts));
  opts.quant = NULL;
  import_and_generate(&opts, "bcdabcda", 8);

  /* q8 per-tensor: weights are exactly representable, chain preserved. */
  memset(&opts, 0, sizeof(opts));
  opts.quant = "q8";
  opts.quant_group = 0;
  import_and_generate(&opts, "bcdabcda", 8);

  /* q4 grouped: same. */
  memset(&opts, 0, sizeof(opts));
  opts.quant = "q4";
  opts.quant_group = 4;
  import_and_generate(&opts, "bcdabcda", 8);

  /* Import to a file works too. */
  {
    memset(&opts, 0, sizeof(opts));
    opts.quant = NULL;
    CHECK_EQ_INT(pai_import_model_to_file("tiny_desc.txt", "imported.pai",
                                          &opts),
                 PAI_OK);
    {
      pai_model_t *model = NULL;
      CHECK_EQ_INT(pai_model_open(NULL, "imported.pai", &model), PAI_OK);
      CHECK(model != NULL);
      pai_model_close(model);
    }
  }

  /* Error paths. */
  memset(&opts, 0, sizeof(opts));
  opts.quant = "q8";
  CHECK_EQ_INT(pai_import_model("no_such.txt", &opts, NULL, NULL),
               PAI_ERR_INVALID_ARG);
  {
    uint8_t *blob = NULL;
    uint32_t n = 0;
    CHECK_EQ_INT(pai_import_model("no_such.txt", &opts, &blob, &n),
                 PAI_ERR_IO);
    /* Missing weight file. */
    CHECK_EQ_INT(write_file("tiny_desc.txt", k_desc, sizeof(k_desc) - 1), 0);
    {
      /* Break a weight file reference: remove tok_embed.bin. */
      remove("tok_embed.bin");
      CHECK_EQ_INT(pai_import_model("tiny_desc.txt", &opts, &blob, &n),
                   PAI_ERR_IO);
      build_weights(w_embed, w_out);
      f = fopen("tok_embed.bin", "wb");
      CHECK(f != NULL);
      fwrite(w_embed, sizeof(float), H * V, f);
      fclose(f);
    }
    /* Bad quant name. */
    opts.quant = "q9";
    CHECK_EQ_INT(pai_import_model("tiny_desc.txt", &opts, &blob, &n),
                 PAI_ERR_INVALID_ARG);
    opts.quant = NULL;
  }
}

TEST_MAIN_END()
