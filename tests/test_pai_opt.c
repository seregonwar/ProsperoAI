#include "test.h"

#include <pai_opt.h>
#include <pai/pai.h>

#include <stdlib.h>
#include <string.h>

static const char *k_names[6] = {
    "tok_embeddings.weight",         /* embedding  [16, 8]  */
    "layers.0.attention.wq.weight",  /* attention  [8, 8]   */
    "layers.0.mlp.gate_proj.weight", /* mlp        [16, 8]  */
    "layers.0.mlp.down_proj.weight", /* mlp        [8, 16]  */
    "output.weight",                 /* output     [16, 8]  */
    "layers.0.input_layernorm.weight", /* norm      [8]     */
};

static const uint64_t k_shapes[6][2] = {
    {16, 8},
    {8, 8},
    {16, 8},
    {8, 16},
    {16, 8},
    {8, 0}, /* rank 1 */
};

static const uint32_t k_ranks[6] = {2, 2, 2, 2, 2, 1};

/* Build a f32 container with the six tensors above. */
static int
build_fixture(uint8_t **out_blob, uint32_t *out_nbytes) {
  pai_pai_meta_t meta;
  pai_pai_tensor_t tensors[6];
  pai_pai_builder_t builder;
  uint8_t meta_blob[PAI_PAI_META_SIZE];
  uint8_t manifest_blob[1024];
  uint32_t meta_n = 0;
  uint32_t manifest_n = 0;
  uint32_t total = 0;
  uint64_t offset = 0;
  uint8_t *blob;
  uint32_t i;

  memset(&meta, 0, sizeof(meta));
  strcpy(meta.name, "opt-lm");
  meta.family = PAI_PAI_FAMILY_LLM;
  meta.context_len = 32;
  meta.num_layers = 2;
  meta.kv_bytes_per_token = 8;
  meta.vocab_size = 16;

  memset(tensors, 0, sizeof(tensors));
  for (i = 0; i < 6; i++) {
    uint64_t numel = k_shapes[i][0] * (k_ranks[i] == 2 ? k_shapes[i][1] : 1);
    strcpy(tensors[i].name, k_names[i]);
    tensors[i].value_id = i + 2;
    tensors[i].dtype = PAI_DTYPE_F32;
    tensors[i].rank = k_ranks[i];
    tensors[i].shape[0] = k_shapes[i][0];
    if (k_ranks[i] == 2) {
      tensors[i].shape[1] = k_shapes[i][1];
    }
    tensors[i].offset = offset;
    tensors[i].size_bytes = numel * sizeof(float);
    offset += tensors[i].size_bytes;
  }

  CHECK_EQ_INT(pai_pai_meta_encode(&meta, meta_blob, sizeof(meta_blob),
                                   &meta_n),
               PAI_OK);
  CHECK_EQ_INT(pai_pai_manifest_encode(tensors, 6, manifest_blob,
                                       sizeof(manifest_blob), &manifest_n),
               PAI_OK);
  pai_pai_builder_init(&builder);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_META, meta_blob,
                                   meta_n),
               PAI_OK);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_MANIFEST,
                                   manifest_blob, manifest_n),
               PAI_OK);
  {
    /* Weights blob: f32 layout, offsets as recorded above. */
    uint8_t *weights = (uint8_t *)calloc(1, (size_t)offset);
    CHECK(weights != NULL);
    memset(weights, 0xAB, (size_t)offset);
    CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_WEIGHTS, weights,
                                     (uint32_t)offset),
                 PAI_OK);
    free(weights);
  }

  total = pai_pai_encoded_size(&builder);
  blob = (uint8_t *)malloc(total);
  CHECK(blob != NULL);
  CHECK_EQ_INT(pai_pai_build(&builder, blob, total, &total), PAI_OK);
  *out_blob = blob;
  *out_nbytes = total;
  return 1;
}

TEST_MAIN_BEGIN()

/* ---- role classification (§15 table) ---- */
CHECK(pai_opt_classify("tok_embeddings.weight") == PAI_OPT_ROLE_EMBEDDING);
CHECK(pai_opt_classify("model.embed_tokens.weight") == PAI_OPT_ROLE_EMBEDDING);
CHECK(pai_opt_classify("layers.0.attention.wq.weight") ==
      PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("layers.0.self_attn.k_proj.weight") ==
      PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("layers.0.mlp.gate_proj.weight") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("layers.0.mlp.down_proj.weight") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("model.feed_forward.w2") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("lm_head.weight") == PAI_OPT_ROLE_OUTPUT);
CHECK(pai_opt_classify("layers.0.input_layernorm.weight") ==
      PAI_OPT_ROLE_NORM);
CHECK(pai_opt_classify("layers.0.attn_norm.weight") == PAI_OPT_ROLE_NORM);
CHECK(pai_opt_classify("some.mystery.buffer") == PAI_OPT_ROLE_OTHER);
CHECK(pai_opt_classify(NULL) == PAI_OPT_ROLE_OTHER);
/* GGUF/LLaMA convention (the primary import path): ffn_* are MLP,
 * attn_* are attention (attn_output is the Wo projection), and the
 * per-layer norm wins over the "attn"/"output" substrings. */
CHECK(pai_opt_classify("blk.0.ffn_gate.weight") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("blk.0.ffn_up.weight") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("blk.0.ffn_down.weight") == PAI_OPT_ROLE_MLP);
CHECK(pai_opt_classify("blk.0.attn_q.weight") == PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("blk.0.attn_k.weight") == PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("blk.0.attn_v.weight") == PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("blk.0.attn_output.weight") ==
      PAI_OPT_ROLE_ATTENTION);
CHECK(pai_opt_classify("blk.0.attn_norm.weight") == PAI_OPT_ROLE_NORM);
CHECK(pai_opt_classify("output.weight") == PAI_OPT_ROLE_OUTPUT);
CHECK(pai_opt_classify("token_embd.weight") == PAI_OPT_ROLE_EMBEDDING);
CHECK(strcmp(pai_opt_role_name(PAI_OPT_ROLE_MLP), "mlp") == 0);
CHECK(strcmp(pai_opt_scheme_name(PAI_OPT_SCHEME_Q4), "q4") == 0);

{
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_pai_container_t c;
  pai_opt_plan_t plan;
  uint32_t i;

  CHECK(build_fixture(&blob, &nbytes));
  CHECK_EQ_INT(pai_pai_open(blob, nbytes, &c), PAI_OK);

  /* ---- default plan (role defaults, no budget) ---- */
  CHECK_EQ_INT(pai_opt_plan(&c, NULL, 0, 0, &plan), PAI_OK);
  CHECK(strcmp(plan.name, "opt-lm") == 0);
  CHECK_EQ_UINT(plan.num_tensors, 6);
  /* f32 totals: 512+256+512+512+512+32. */
  CHECK_EQ_UINT(plan.cur_total, 2336);
  CHECK_EQ_UINT(plan.feasible, 1);
  /* chosen schemes: embedding q8, attention q8, mlp q4, output q8,
   * norm f32. */
  CHECK_EQ_INT(plan.tensors[0].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[1].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[2].chosen, PAI_OPT_SCHEME_Q4);
  CHECK_EQ_INT(plan.tensors[3].chosen, PAI_OPT_SCHEME_Q4);
  CHECK_EQ_INT(plan.tensors[4].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[5].chosen, PAI_OPT_SCHEME_F32);
  /* q4(128) = 64 + 4; q8(128) = 128 + 4; q8(64) = 64 + 4; norm f32. */
  CHECK_EQ_UINT(plan.plan_total, 500);
  CHECK_EQ_UINT(plan.min_total, 468);

  /* ---- budget fit: greedy largest-savings from f32 ---- */
  /* 2336 -> 700 needs savings >= 1636: 4x380 + 188 = 628. */
  CHECK_EQ_INT(pai_opt_plan(&c, NULL, 700, 0, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.plan_total, 628);
  CHECK_EQ_UINT(plan.feasible, 1);
  CHECK_EQ_INT(plan.tensors[0].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[4].chosen, PAI_OPT_SCHEME_Q8);
  /* Budget below the cheapest layout: best effort + infeasible. */
  CHECK_EQ_INT(pai_opt_plan(&c, NULL, 100, 0, &plan), PAI_OK);
  CHECK_EQ_UINT(plan.plan_total, 468);
  CHECK_EQ_UINT(plan.feasible, 0);

  /* ---- role floors hold under pressure ---- */
  /* The norm stays f32 and embeddings/output stay q8 even when the
   * budget is unreachable. */
  CHECK_EQ_INT(plan.tensors[5].chosen, PAI_OPT_SCHEME_F32);
  CHECK_EQ_INT(plan.tensors[0].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[4].chosen, PAI_OPT_SCHEME_Q8);
  CHECK_EQ_INT(plan.tensors[2].chosen, PAI_OPT_SCHEME_Q4);

  /* ---- group size changes the scale overhead ---- */
  CHECK_EQ_INT(pai_opt_plan(&c, NULL, 0, 32, &plan), PAI_OK);
  /* q8(128, g32) = 128 + 4 groups*4 = 144; q4(128, g32) = 64 + 16. */
  {
    uint64_t mlp_plan = 0;
    CHECK_EQ_UINT(plan.tensors[0].size_bytes[PAI_OPT_SCHEME_Q8], 144);
    for (i = 0; i < plan.num_tensors; i++) {
      if (plan.tensors[i].role == PAI_OPT_ROLE_MLP) {
        mlp_plan += plan.tensors[i].size_bytes[plan.tensors[i].chosen];
      }
    }
    CHECK_EQ_UINT(mlp_plan, 160); /* 2 x q4(128, g32) */
  }

  /* ---- invalid arguments ---- */
  CHECK_EQ_INT(pai_opt_plan(NULL, NULL, 0, 0, &plan), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_opt_plan(&c, NULL, 0, 0, NULL), PAI_ERR_INVALID_ARG);

  free(blob);
}

TEST_MAIN_END()
