#include "model.h"

#include <quantize.h>

#include <stdlib.h>
#include <string.h>

pai_status_t
pai_model_open_path(const char *path, pai_model_t **out_model) {
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_status_t st;

  if (path == NULL || out_model == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  st = pai_pai_read_file(path, &blob, &nbytes);
  if (st != PAI_OK) {
    return st;
  }
  st = pai_model_open_blob(blob, nbytes, out_model);
  free(blob);
  return st;
}

pai_status_t
pai_model_open_blob(const uint8_t *blob, uint32_t nbytes,
                    pai_model_t **out_model) {
  pai_pai_container_t container;
  pai_pai_meta_t meta;
  const uint8_t *sec;
  uint32_t sec_size;
  pai_model_t *model;
  pai_status_t st;

  if (blob == NULL || out_model == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_model = NULL;

  st = pai_pai_open(blob, nbytes, &container);
  if (st != PAI_OK) {
    return st;
  }

  sec = pai_pai_section(&container, PAI_PAI_SEC_META, &sec_size);
  if (sec == NULL) {
    return PAI_ERR_PROTOCOL;
  }
  st = pai_pai_meta_decode(sec, sec_size, &meta);
  if (st != PAI_OK) {
    return st;
  }

  model = (pai_model_t *)calloc(1, sizeof(*model));
  if (model == NULL) {
    return PAI_ERR_NOMEM;
  }
  memcpy(model->name, meta.name, sizeof(model->name));
  model->family = meta.family;
  model->context_len = meta.context_len;
  model->num_layers = meta.num_layers;
  model->kv_bytes_per_token = meta.kv_bytes_per_token;
  model->vocab_size = meta.vocab_size;

  /* Manifest: tensor names -> value ids + weight offsets. */
  sec = pai_pai_section(&container, PAI_PAI_SEC_MANIFEST, &sec_size);
  if (sec == NULL) {
    st = PAI_ERR_PROTOCOL;
    goto fail;
  }
  st = pai_pai_manifest_decode(sec, sec_size, model->manifest,
                               PAI_PAI_MAX_TENSORS, &model->num_tensors);
  if (st != PAI_OK) {
    goto fail;
  }

  /* Weights: canonical blob, owned by the model. */
  sec = pai_pai_section(&container, PAI_PAI_SEC_WEIGHTS, &sec_size);
  if (sec == NULL || sec_size == 0) {
    st = PAI_ERR_PROTOCOL;
    goto fail;
  }
  model->weights = (uint8_t *)malloc(sec_size);
  if (model->weights == NULL) {
    st = PAI_ERR_NOMEM;
    goto fail;
  }
  memcpy(model->weights, sec, sec_size);
  model->weights_bytes = sec_size;

  /* Prospero IR: the compiled program. */
  sec = pai_pai_section(&container, PAI_PAI_SEC_IR, &sec_size);
  if (sec == NULL) {
    st = PAI_ERR_PROTOCOL;
    goto fail;
  }
  st = pai_ir_decode(sec, sec_size, &model->ir);
  if (st != PAI_OK) {
    goto fail;
  }

  /* Rebuild graph + static memory plan. Weights must stay resident
   * across generation steps: use the persistent variant (§16). */
  st = pai_ir_to_graph(&model->ir, &model->graph);
  if (st != PAI_OK) {
    goto fail;
  }
  st = pai_graph_memory_plan_persistent(&model->graph, &model->mem_plan);
  if (st != PAI_OK) {
    goto fail;
  }

  /* Tokenizer (optional section). */
  pai_tok_init(&model->tokenizer);
  sec = pai_pai_section(&container, PAI_PAI_SEC_TOKENIZER, &sec_size);
  if (sec != NULL) {
    st = pai_tok_deserialize(&model->tokenizer, sec, sec_size);
    if (st != PAI_OK) {
      goto fail;
    }
  }

  *out_model = model;
  return PAI_OK;

fail:
  if (model->weights != NULL) {
    free(model->weights);
  }
  pai_tok_destroy(&model->tokenizer);
  free(model);
  return st;
}

void
pai_model_close(pai_model_t *model) {
  if (model == NULL) {
    return;
  }
  pai_tok_destroy(&model->tokenizer);
  if (model->weights != NULL) {
    free(model->weights);
  }
  free(model);
}

const char *
pai_model_name(const pai_model_t *model) {
  return model != NULL ? model->name : NULL;
}

static uint64_t
next_pow2_u64(uint64_t v) {
  uint64_t p = 1;
  while (p < v && p < (UINT64_C(1) << 62)) {
    p <<= 1;
  }
  return p;
}

static const pai_graph_value_t *
graph_value(const pai_graph_t *graph, uint32_t id) {
  return (id > 0 && id <= graph->num_values) ? &graph->values[id] : NULL;
}

static float *
value_ptr(pai_session_t *session, uint32_t id) {
  return (float *)(void *)(session->region +
                           pai_graph_mem_plan_offset(&session->model->mem_plan,
                                                     id));
}

pai_status_t
pai_session_init(pai_model_t *model, pai_session_t **out_session) {
  pai_session_t *session;
  pai_status_t st;
  uint32_t i;

  if (model == NULL || out_session == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_session = NULL;

  session = (pai_session_t *)calloc(1, sizeof(*session));
  if (session == NULL) {
    return PAI_ERR_NOMEM;
  }
  session->model = model;
  session->seed = 0x5EEDC0DEull;
  pai_sampler_init(&session->sampler, session->seed);
  session->max_tokens = 32;
  session->eos_token = 0;

  st = pai_sched_build_persistent(&model->graph, &model->mem_plan,
                                  PAI_SCHED_INTERACTIVE, NULL,
                                  &session->plan);
  if (st != PAI_OK) {
    free(session);
    return st;
  }

  /* Planned storage region. */
  session->region = (uint8_t *)calloc(1, (size_t)model->mem_plan.region_bytes);
  if (session->region == NULL) {
    free(session);
    return PAI_ERR_NOMEM;
  }

  /* Load canonical weights into planned offsets. Quantized tensors
   * (§15) are dequantized to f32: the reference executor is f32-only,
   * so quantized storage stays in the container and f32 state in the
   * region. */
  for (i = 0; i < model->num_tensors; i++) {
    const pai_pai_tensor_t *t = &model->manifest[i];
    const pai_graph_value_t *v = graph_value(&model->graph, t->value_id);
    const pai_ir_value_t *iv;
    uint64_t off;

    if (v == NULL) {
      continue;
    }
    off = pai_graph_mem_plan_offset(&model->mem_plan, t->value_id);
    iv = t->value_id <= model->ir.num_values ? &model->ir.values[t->value_id]
                                             : NULL;

    if (iv != NULL && iv->quant.present) {
      pai_quant_scheme_t scheme;
      uint64_t n;
      uint64_t q_bytes;

      if (iv->quant.bit_width != 4 && iv->quant.bit_width != 8) {
        st = PAI_ERR_UNSUPPORTED; /* v0 executes q4/q8 only           */
        goto fail;
      }
      if (off + v->size_bytes > model->mem_plan.region_bytes ||
          t->offset + t->size_bytes > model->weights_bytes) {
        st = PAI_ERR_MISMATCH;
        goto fail;
      }
      n = v->size_bytes / sizeof(float);
      scheme.bit_width = iv->quant.bit_width;
      scheme.is_signed = 1; /* v0: signed symmetric (§15)              */
      scheme.group_size = iv->quant.group_size; /* 0 = per-tensor     */
      q_bytes = pai_quant_value_bytes(n, &scheme);
      if (q_bytes + pai_quant_scale_bytes(n, &scheme) != t->size_bytes) {
        st = PAI_ERR_MISMATCH;
        goto fail;
      }
      st = pai_dequantize_f32(
          (const int8_t *)(const void *)(model->weights + t->offset),
          (const float *)(const void *)(model->weights + t->offset + q_bytes),
          n, &scheme, (float *)(void *)(session->region + off));
      if (st != PAI_OK) {
        goto fail;
      }
      continue;
    }

    if (off + t->size_bytes > model->mem_plan.region_bytes ||
        t->offset + t->size_bytes > model->weights_bytes) {
      st = PAI_ERR_MISMATCH;
      goto fail;
    }
    memcpy(session->region + off, model->weights + t->offset, t->size_bytes);
  }

  /* Per-session KV cache (§19), sized from the model metadata. */
  if (model->num_layers > 0 && model->kv_bytes_per_token > 0) {
    uint64_t need = (uint64_t)model->context_len * model->num_layers *
                    model->kv_bytes_per_token;
    uint64_t region_size = next_pow2_u64(need < 128 ? 128 : need);

    st = pai_kv_cache_init_owned(&session->kv, region_size, model->num_layers,
                                 model->kv_bytes_per_token);
    if (st != PAI_OK) {
      goto fail;
    }
  }

  *out_session = session;
  return PAI_OK;

fail:
  if (session->region != NULL) {
    free(session->region);
  }
  free(session);
  return st;
}

void
pai_session_destroy(pai_session_t *session) {
  if (session == NULL) {
    return;
  }
  if (session->kv.region != NULL) {
    pai_kv_cache_destroy(&session->kv);
  }
  if (session->region != NULL) {
    free(session->region);
  }
  free(session);
}

pai_status_t
pai_session_set_sampler(pai_session_t *session, const pai_sampler_t *sampler) {
  if (session == NULL || sampler == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  session->sampler = *sampler;
  return PAI_OK;
}

pai_status_t
pai_session_set_generation(pai_session_t *session, uint32_t max_tokens,
                           uint32_t eos_token) {
  if (session == NULL || max_tokens == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  session->max_tokens = max_tokens;
  session->eos_token = eos_token;
  return PAI_OK;
}

pai_status_t
pai_session_generate(pai_session_t *session, const char *prompt,
                     void (*on_token)(const char *token, void *user),
                     void *user) {
  pai_model_t *model;
  const pai_ir_program_t *ir;
  uint32_t *ids;
  uint32_t n = 0;
  uint32_t cur;
  uint32_t steps;
  uint32_t input_id;
  uint32_t output_id;
  const pai_graph_value_t *in_value;
  const pai_graph_value_t *out_value;
  uint32_t vocab;
  uint32_t ctx = 0;
  int seq_mode = 0;
  char text[PAI_TOK_MAX_TOKEN_LEN * 8 + 1];
  pai_status_t st;

  if (session == NULL || prompt == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  model = session->model;
  ir = &model->ir;

  if (model->tokenizer.num_tokens == 0) {
    return PAI_ERR_UNSUPPORTED; /* model has no tokenizer section */
  }
  if (ir->num_inputs == 0 || ir->num_outputs == 0) {
    return PAI_ERR_MISMATCH;
  }
  input_id = ir->input_ids[0];
  output_id = ir->output_ids[0];
  in_value = graph_value(&model->graph, input_id);
  out_value = graph_value(&model->graph, output_id);
  if (in_value == NULL || out_value == NULL) {
    return PAI_ERR_MISMATCH;
  }

  /* Two input conventions:
   *   rank-1 [vocab]   single-token models (v0 importer): one-hot of the
   *                    current token, output sampled from the whole
   *                    output value (expected to hold probabilities).
   *   rank-2 [ctx, vocab]  sequence models (§9, GGUF adapter): every
   *                    step re-feeds the full context (real tokens at
   *                    positions 0..seq_len-1, zeros after); causal
   *                    attention makes position seq_len-1 the current
   *                    token. Output is [ctx, vocab] logits; the last
   *                    real row is sampled.
   */
  seq_mode = in_value->rank == 2;
  if (seq_mode) {
    ctx = (uint32_t)in_value->shape[0];
    vocab = model->vocab_size != 0 ? model->vocab_size
                                   : (uint32_t)in_value->shape[1];
    if (vocab == 0 || ctx == 0 || ctx > PAI_TOK_MAX_INPUT ||
        out_value->rank != 2 || out_value->shape[1] != vocab ||
        out_value->shape[0] < ctx) {
      return PAI_ERR_MISMATCH;
    }
  } else {
    ctx = 0;
    vocab = model->vocab_size != 0 ? model->vocab_size
                                   : (uint32_t)in_value->shape[0];
    if (vocab == 0) {
      return PAI_ERR_MISMATCH;
    }
  }

  /* Tokenize the prompt; the last token seeds generation. */
  ids = (uint32_t *)malloc(PAI_TOK_MAX_INPUT * sizeof(uint32_t));
  if (ids == NULL) {
    return PAI_ERR_NOMEM;
  }
  st = pai_tok_encode(&model->tokenizer, prompt, (uint32_t)strlen(prompt),
                      ids, PAI_TOK_MAX_INPUT, &n);
  if (st != PAI_OK || n == 0) {
    free(ids);
    return st != PAI_OK ? st : PAI_ERR_INVALID_ARG;
  }
  cur = ids[n - 1];

  session->generated_tokens = 0;
  if (seq_mode) {
    uint32_t seq_len = n < ctx ? n : ctx;

    for (steps = 0; steps < session->max_tokens; steps++) {
      float *in = value_ptr(session, input_id);
      float *out = value_ptr(session, output_id);
      uint32_t sampled;
      uint32_t text_n = 0;
      uint32_t i;

      if (seq_len >= ctx) {
        break; /* context exhausted: stop generating */
      }

      memset(in, 0, (size_t)in_value->size_bytes);
      for (i = 0; i < seq_len; i++) {
        if (ids[i] >= vocab) {
          free(ids);
          return PAI_ERR_MISMATCH;
        }
        in[(size_t)i * vocab + ids[i]] = 1.0f;
      }

      st = pai_sched_execute(&model->graph, &model->mem_plan, &session->plan,
                             session->region, NULL);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }

      /* Sample the last real position's logits. */
      st = pai_sampler_sample_logits(&session->sampler,
                                     out + (size_t)(seq_len - 1) * vocab,
                                     vocab, &sampled);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }

      st = pai_tok_decode(&model->tokenizer, &sampled, 1, text, sizeof(text),
                          &text_n);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }
      session->generated_tokens++;
      if (on_token != NULL) {
        on_token(text, user);
      }

      ids[seq_len++] = sampled;
      if (session->eos_token != 0 && sampled == session->eos_token) {
        break;
      }
    }
  } else {
    for (steps = 0; steps < session->max_tokens; steps++) {
      float *in = value_ptr(session, input_id);
      float *out = value_ptr(session, output_id);
      uint32_t sampled = cur;
      uint32_t text_n = 0;

      if (cur >= vocab) {
        free(ids);
        return PAI_ERR_MISMATCH; /* token id outside the model's vocab */
      }
      memset(in, 0, (size_t)in_value->size_bytes);
      in[cur] = 1.0f;

      st = pai_sched_execute(&model->graph, &model->mem_plan, &session->plan,
                             session->region, NULL);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }

      st = pai_sampler_sample_probs(&session->sampler, out,
                                    (uint32_t)out_value->shape[0], &sampled);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }

      st = pai_tok_decode(&model->tokenizer, &sampled, 1, text, sizeof(text),
                          &text_n);
      if (st != PAI_OK) {
        free(ids);
        return st;
      }
      session->generated_tokens++;
      if (on_token != NULL) {
        on_token(text, user);
      }

      cur = sampled;
      if (session->eos_token != 0 && cur == session->eos_token) {
        break;
      }
    }
  }

  free(ids);
  return PAI_OK;
}

/* Embeddings (§26 /v1/embeddings) */

/*
 * Locate the token-embedding parameter: the non-input operand of the
 * first GEMM/GEMV/MATMUL consuming the graph input. Layout is
 * row-major [vocab, dim] (GGUF adapter) or column-major [dim, vocab]
 * (importer DSL). Returns the value id, or 0 when there is none.
 */
static uint32_t
find_embed_param(const pai_model_t *model, int *out_row_major,
                 uint64_t *out_dim) {
  const pai_ir_program_t *ir = &model->ir;
  uint32_t input_id = ir->num_inputs > 0 ? ir->input_ids[0] : 0;
  uint32_t vocab = model->vocab_size;
  uint32_t oi;

  if (input_id == 0 || vocab == 0) {
    return 0;
  }
  for (oi = 1; oi <= ir->num_ops; oi++) {
    const pai_ir_op_t *op = &ir->ops[oi];
    uint32_t j;
    if (op->kind != PAI_IR_OP_GEMM && op->kind != PAI_IR_OP_GEMV &&
        op->kind != PAI_IR_OP_MATMUL) {
      continue;
    }
    for (j = 0; j < op->num_inputs; j++) {
      uint32_t k;
      if (op->inputs[j] != input_id) {
        continue;
      }
      for (k = 0; k < op->num_inputs; k++) {
        uint32_t cand;
        const pai_ir_value_t *iv;
        if (k == j) {
          continue;
        }
        cand = op->inputs[k];
        if (cand == 0 || cand > ir->num_values) {
          continue;
        }
        iv = &ir->values[cand];
        if (iv->kind != PAI_IR_VALUE_PARAM || iv->rank != 2) {
          continue;
        }
        if (iv->shape[0] == (uint64_t)vocab) {
          *out_row_major = 1;
          *out_dim = iv->shape[1];
          return cand;
        }
        if (iv->shape[1] == (uint64_t)vocab) {
          *out_row_major = 0;
          *out_dim = iv->shape[0];
          return cand;
        }
      }
    }
  }
  return 0;
}

/* Load a value's canonical weight blob as f32 (dequantizing when
 * stored quantized). Caller frees *out. */
static pai_status_t
load_value_f32(const pai_model_t *model, uint32_t value_id, float **out,
               uint64_t *out_n) {
  const pai_pai_tensor_t *t = NULL;
  const pai_ir_value_t *iv;
  uint64_t numel = 1;
  uint32_t i;
  float *buf;

  for (i = 0; i < model->num_tensors; i++) {
    if (model->manifest[i].value_id == value_id) {
      t = &model->manifest[i];
      break;
    }
  }
  if (t == NULL) {
    return PAI_ERR_MISMATCH;
  }
  iv = value_id <= model->ir.num_values ? &model->ir.values[value_id] : NULL;
  if (iv == NULL) {
    return PAI_ERR_MISMATCH;
  }
  for (i = 0; i < iv->rank; i++) {
    numel *= iv->shape[i];
  }
  if (numel == 0 || numel > (UINT64_C(1) << 32)) {
    return PAI_ERR_UNSUPPORTED;
  }

  if (iv->quant.present) {
    pai_quant_scheme_t scheme;
    uint64_t q_bytes;

    if (iv->quant.bit_width != 4 && iv->quant.bit_width != 8) {
      return PAI_ERR_UNSUPPORTED;
    }
    scheme.bit_width = iv->quant.bit_width;
    scheme.is_signed = 1;
    scheme.group_size = iv->quant.group_size;
    q_bytes = pai_quant_value_bytes(numel, &scheme);
    if (q_bytes + pai_quant_scale_bytes(numel, &scheme) != t->size_bytes) {
      return PAI_ERR_MISMATCH;
    }
    buf = (float *)malloc((size_t)numel * sizeof(float));
    if (buf == NULL) {
      return PAI_ERR_NOMEM;
    }
    {
      pai_status_t st = pai_dequantize_f32(
          (const int8_t *)(const void *)(model->weights + t->offset),
          (const float *)(const void *)(model->weights + t->offset + q_bytes),
          numel, &scheme, buf);
      if (st != PAI_OK) {
        free(buf);
        return st;
      }
    }
  } else {
    if (t->size_bytes != numel * sizeof(float) ||
        t->offset + t->size_bytes > model->weights_bytes) {
      return PAI_ERR_MISMATCH;
    }
    buf = (float *)malloc((size_t)numel * sizeof(float));
    if (buf == NULL) {
      return PAI_ERR_NOMEM;
    }
    memcpy(buf, model->weights + t->offset, (size_t)numel * sizeof(float));
  }
  *out = buf;
  *out_n = numel;
  return PAI_OK;
}

pai_status_t
pai_model_embed_dim(const pai_model_t *model, uint32_t *out_dim) {
  int row_major;
  uint64_t dim;

  if (model == NULL || out_dim == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (find_embed_param(model, &row_major, &dim) == 0 || dim == 0 ||
      dim > 0xFFFFFFFFu) {
    return PAI_ERR_UNSUPPORTED;
  }
  *out_dim = (uint32_t)dim;
  return PAI_OK;
}

pai_status_t
pai_model_embed(pai_model_t *model, const char *text, uint32_t max_elements,
                float *out, uint32_t *out_n) {
  uint32_t value_id;
  int row_major;
  uint64_t dim64;
  uint32_t dim;
  uint32_t vocab;
  float *mat = NULL;
  uint64_t mat_n = 0;
  float *acc = NULL;
  uint32_t *ids = NULL;
  uint32_t nids = 0;
  uint32_t count = 0;
  uint32_t i;
  uint64_t d;
  pai_status_t st;

  if (model == NULL || text == NULL || out == NULL || out_n == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_n = 0;
  value_id = find_embed_param(model, &row_major, &dim64);
  if (value_id == 0 || dim64 == 0 || dim64 > 0xFFFFFFFFu) {
    return PAI_ERR_UNSUPPORTED;
  }
  dim = (uint32_t)dim64;
  if (max_elements < dim) {
    return PAI_ERR_NOMEM;
  }
  vocab = model->vocab_size;

  st = load_value_f32(model, value_id, &mat, &mat_n);
  if (st != PAI_OK) {
    return st;
  }
  if (mat_n != (uint64_t)dim * vocab) {
    free(mat);
    return PAI_ERR_MISMATCH;
  }

  acc = (float *)calloc(dim, sizeof(float));
  ids = (uint32_t *)malloc(PAI_TOK_MAX_INPUT * sizeof(uint32_t));
  if (acc == NULL || ids == NULL) {
    free(acc);
    free(ids);
    free(mat);
    return PAI_ERR_NOMEM;
  }
  st = pai_tok_encode(&model->tokenizer, text, (uint32_t)strlen(text), ids,
                      PAI_TOK_MAX_INPUT, &nids);
  if (st != PAI_OK) {
    free(acc);
    free(ids);
    free(mat);
    return st;
  }

  for (i = 0; i < nids; i++) {
    uint32_t tok = ids[i];
    if (tok >= vocab) {
      continue; /* byte-fallback ids have no embedding row */
    }
    if (row_major) {
      const float *row = mat + (uint64_t)tok * dim;
      for (d = 0; d < dim; d++) {
        acc[d] += row[d];
      }
    } else {
      for (d = 0; d < dim; d++) {
        acc[d] += mat[(uint64_t)d * vocab + tok];
      }
    }
    count++;
  }
  if (count == 0) {
    free(acc);
    free(ids);
    free(mat);
    return PAI_ERR_MISMATCH;
  }

  for (d = 0; d < dim; d++) {
    out[d] = acc[d] / (float)count;
  }
  *out_n = dim;
  free(acc);
  free(ids);
  free(mat);
  return PAI_OK;
}
