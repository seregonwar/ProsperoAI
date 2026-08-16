#include "test.h"

#include <graph/graph.h>
#include <ir/ir.h>
#include <model.h>
#include <pai/pai.h>
#include <tokenizer.h>

#include <stdlib.h>
#include <string.h>

#define V 4 /* vocab (tokens a..d) */
#define H 8 /* hidden */

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

/* Build the tiny LM graph: onehot[V] -> GEMV(W_embed) -> h[H] ->
 * GEMV(W_out) -> logits[V] -> SOFTMAX -> probs[V]. Weights chosen so
 * token t deterministically maps to token (t+1)%V under greedy
 * decoding: a -> b -> c -> d -> a -> ... */
static void
build_graph(pai_graph_t *g) {
  const uint64_t s_v[1] = {V};
  const uint64_t s_he[2] = {H, V};
  const uint64_t s_h[1] = {H};
  const uint64_t s_vh[2] = {V, H};
  pai_graph_value_id onehot;
  pai_graph_value_id w_embed;
  pai_graph_value_id h;
  pai_graph_value_id w_out;
  pai_graph_value_id logits;
  pai_graph_value_id probs;

  pai_graph_init(g);
  onehot = pai_graph_add_value(g, PAI_DTYPE_F32, 1, s_v, 16);
  w_embed = pai_graph_add_value(g, PAI_DTYPE_F32, 2, s_he, 16);
  h = pai_graph_add_value(g, PAI_DTYPE_F32, 1, s_h, 16);
  w_out = pai_graph_add_value(g, PAI_DTYPE_F32, 2, s_vh, 16);
  logits = pai_graph_add_value(g, PAI_DTYPE_F32, 1, s_v, 16);
  probs = pai_graph_add_value(g, PAI_DTYPE_F32, 1, s_v, 16);

  CHECK(onehot != 0 && w_embed != 0 && h != 0 && w_out != 0 && logits != 0 &&
        probs != 0);

  /* GEMV semantics: y[m] = A[m,k] * x[k] with A rank2, x rank1. */
  {
    pai_graph_value_id in1[] = {w_embed, onehot};
    pai_graph_value_id out1[] = {h};
    CHECK(pai_graph_add_op(g, PAI_OP_GEMV, 2, 1, in1, out1) != 0);
  }
  {
    pai_graph_value_id in2[] = {w_out, h};
    pai_graph_value_id out2[] = {logits};
    CHECK(pai_graph_add_op(g, PAI_OP_GEMV, 2, 1, in2, out2) != 0);
  }
  {
    pai_graph_value_id in3[] = {logits};
    pai_graph_value_id out3[] = {probs};
    CHECK(pai_graph_add_op(g, PAI_OP_SOFTMAX, 1, 1, in3, out3) != 0);
  }

  pai_graph_set_input(g, onehot);
  pai_graph_set_output(g, probs);
}

/* Canonical weights: w_embed [H,V] (column t = 2*e_t), w_out [V,H]
 * (row-major w_out[j][i] = 100 if i<V and j==(i+1)%V). */
static void
build_weights(float *w_embed, float *w_out) {
  uint32_t i;
  uint32_t j;

  memset(w_embed, 0, (size_t)(H * V) * sizeof(float));
  for (i = 0; i < V && i < H; i++) {
    w_embed[i * V + i] = 2.0f;
  }
  memset(w_out, 0, (size_t)(V * H) * sizeof(float));
  for (i = 0; i < V; i++) {
    w_out[((i + 1) % V) * H + i] = 100.0f;
  }
  (void)j;
}

/* Assemble a .pai container blob from the graph + weights + tokenizer. */
static uint8_t *
make_container(pai_graph_t *g, const float *w_embed, const float *w_out,
               uint32_t *out_nbytes) {
  pai_ir_program_t ir;
  uint8_t *ir_blob = NULL;
  uint32_t ir_n = 0;
  pai_tok_t tok;
  uint8_t *tok_blob = NULL;
  uint32_t tok_n = 0;
  pai_pai_meta_t meta;
  uint8_t meta_blob[PAI_PAI_META_SIZE];
  uint32_t meta_n = 0;
  pai_pai_tensor_t tensors[2];
  uint8_t manifest_blob[512];
  uint32_t manifest_n = 0;
  uint8_t *weights_blob;
  uint32_t weights_n = (uint32_t)((H * V + V * H) * sizeof(float));
  pai_pai_builder_t builder;
  uint8_t *container_blob = NULL;
  uint32_t total = 0;

  /* IR program from the graph. */
  CHECK_EQ_INT(pai_ir_from_graph(g, &ir), PAI_OK);
  ir_n = pai_ir_encoded_size(&ir);
  CHECK(ir_n > 0);
  ir_blob = (uint8_t *)malloc(ir_n);
  CHECK(ir_blob != NULL);
  CHECK_EQ_INT(pai_ir_encode(&ir, ir_blob, ir_n, &ir_n), PAI_OK);

  /* Tokenizer: single-char tokens a..d with ids 0..3. */
  pai_tok_init(&tok);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 0, "a"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 1, "b"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 2, "c"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 3, "d"), PAI_OK);
  pai_tok_finalize(&tok);
  tok_n = pai_tok_blob_size(&tok);
  tok_blob = (uint8_t *)malloc(tok_n);
  CHECK(tok_blob != NULL);
  CHECK_EQ_INT(pai_tok_serialize(&tok, tok_blob, tok_n, &tok_n), PAI_OK);

  /* Metadata. */
  memset(&meta, 0, sizeof(meta));
  strcpy(meta.name, "tiny-lm");
  meta.family = PAI_PAI_FAMILY_LLM;
  meta.context_len = 16;
  meta.num_layers = 1;
  meta.kv_bytes_per_token = 8;
  meta.vocab_size = V;
  CHECK_EQ_INT(pai_pai_meta_encode(&meta, meta_blob, sizeof(meta_blob),
                                   &meta_n),
               PAI_OK);

  /* Manifest: w_embed = graph value 2, w_out = graph value 4. */
  memset(tensors, 0, sizeof(tensors));
  strcpy(tensors[0].name, "w_embed");
  tensors[0].value_id = 2;
  tensors[0].dtype = PAI_DTYPE_F32;
  tensors[0].rank = 2;
  tensors[0].shape[0] = H;
  tensors[0].shape[1] = V;
  tensors[0].offset = 0;
  tensors[0].size_bytes = (uint64_t)(H * V) * sizeof(float);
  strcpy(tensors[1].name, "w_out");
  tensors[1].value_id = 4;
  tensors[1].dtype = PAI_DTYPE_F32;
  tensors[1].rank = 2;
  tensors[1].shape[0] = V;
  tensors[1].shape[1] = H;
  tensors[1].offset = (uint64_t)(H * V) * sizeof(float);
  tensors[1].size_bytes = (uint64_t)(V * H) * sizeof(float);
  CHECK_EQ_INT(pai_pai_manifest_encode(tensors, 2, manifest_blob,
                                       sizeof(manifest_blob), &manifest_n),
               PAI_OK);

  weights_blob = (uint8_t *)malloc(weights_n);
  CHECK(weights_blob != NULL);
  memcpy(weights_blob, w_embed, (size_t)(H * V) * sizeof(float));
  memcpy(weights_blob + (size_t)(H * V) * sizeof(float), w_out,
         (size_t)(V * H) * sizeof(float));

  pai_pai_builder_init(&builder);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_META, meta_blob, meta_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_MANIFEST, manifest_blob,
                      manifest_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_WEIGHTS, weights_blob, weights_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_IR, ir_blob, ir_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_TOKENIZER, tok_blob, tok_n);

  total = pai_pai_encoded_size(&builder);
  CHECK(total > 0);
  container_blob = (uint8_t *)malloc(total);
  CHECK(container_blob != NULL);
  CHECK_EQ_INT(pai_pai_build(&builder, container_blob, total, &total), PAI_OK);
  *out_nbytes = total;

  free(weights_blob);
  free(tok_blob);
  free(ir_blob);
  pai_tok_destroy(&tok);
  return container_blob;
}

TEST_MAIN_BEGIN()

{
  pai_graph_t g;
  float w_embed[H * V];
  float w_out[V * H];
  uint8_t *blob;
  uint32_t nbytes = 0;
  pai_model_t *model = NULL;
  pai_session_t *session = NULL;
  pai_status_t st;

  /* Build + write a .pai file, then open it through the runtime API. */
  build_graph(&g);
  build_weights(w_embed, w_out);
  blob = make_container(&g, w_embed, w_out, &nbytes);
  CHECK(blob != NULL);
  CHECK_EQ_INT(pai_pai_write_file("test_tiny.pai", blob, nbytes), PAI_OK);

  st = pai_model_open(NULL, "test_tiny.pai", &model);
  CHECK_EQ_INT(st, PAI_OK);
  CHECK(model != NULL);
  CHECK(strcmp(pai_model_name(model), "tiny-lm") == 0);
  CHECK_EQ_UINT(model->vocab_size, V);
  CHECK_EQ_UINT(model->num_layers, 1);
  CHECK_EQ_UINT(model->num_tensors, 2);

  /* Create a session and generate greedily from "a". */
  CHECK_EQ_INT(pai_session_create(model, &session), PAI_OK);
  CHECK(session != NULL);
  CHECK_EQ_INT(pai_session_set_generation(session, 8, 0), PAI_OK);

  memset(gen_text, 0, sizeof(gen_text));
  gen_text_n = 0;
  CHECK_EQ_INT(pai_generate(session, "a", on_token, NULL), PAI_OK);
  CHECK_EQ_UINT(session->generated_tokens, 8);
  CHECK_EQ_UINT(gen_text_n, 8);

  /* Greedy chain: a->b->c->d->a->... so the emitted text is "bcdabcda". */
  CHECK(strcmp(gen_text, "bcdabcda") == 0);

  /* EOS stops generation. d == token id 3. */
  CHECK_EQ_INT(pai_session_set_generation(session, 8, 3), PAI_OK);
  memset(gen_text, 0, sizeof(gen_text));
  gen_text_n = 0;
  CHECK_EQ_INT(pai_generate(session, "a", on_token, NULL), PAI_OK);
  CHECK_EQ_UINT(gen_text_n, 3);
  CHECK(strcmp(gen_text, "bcd") == 0);

  /* Determinism: same session, same seed -> same output. */
  {
    char again[256];
    memset(gen_text, 0, sizeof(gen_text));
    gen_text_n = 0;
    CHECK_EQ_INT(pai_generate(session, "a", on_token, NULL), PAI_OK);
    strcpy(again, gen_text);
    CHECK_EQ_UINT(gen_text_n, 3);
    CHECK(strcmp(again, "bcd") == 0);
  }

  /* Non-greedy sampling still draws from the right distribution with a
   * heavy peak (deterministic seed). */
  {
    pai_sampler_t sampler;
    pai_sampler_init(&sampler, 12345);
    sampler.temperature = 1.0f;
    CHECK_EQ_INT(pai_session_set_sampler(session, &sampler), PAI_OK);
    CHECK_EQ_INT(pai_session_set_generation(session, 8, 0), PAI_OK);
    memset(gen_text, 0, sizeof(gen_text));
    gen_text_n = 0;
    CHECK_EQ_INT(pai_generate(session, "a", on_token, NULL), PAI_OK);
    CHECK_EQ_UINT(gen_text_n, 8);
    /* The logits peak (100) dominates even at temperature 1. */
    CHECK(strcmp(gen_text, "bcdabcda") == 0);
  }

  pai_session_destroy(session);
  pai_model_close(model);

  /* Open from the in-memory blob too. */
  CHECK_EQ_INT(pai_model_open_blob(blob, nbytes, &model), PAI_OK);
  CHECK(model != NULL);
  pai_model_close(model);

  free(blob);
}

{
  /* Validation. */
  pai_model_t *model = NULL;
  pai_session_t *session = NULL;

  CHECK_EQ_INT(pai_model_open(NULL, "no_such.pai", &model), PAI_ERR_IO);
  CHECK_EQ_INT(pai_model_open_blob(NULL, 0, &model), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_session_init(NULL, &session), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_session_create(NULL, &session), PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
