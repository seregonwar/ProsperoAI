/*
 * ProsperoAI — mixed-precision quantization planner (whitepaper §15/§20).
 */

#include "pai_opt.h"

#include <quantize.h>

#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static uint64_t
tensor_numel(const pai_pai_tensor_t *t) {
  uint64_t n = 1;
  uint32_t i;

  if (t == NULL || t->rank == 0 || t->rank > PAI_TENSOR_MAX_RANK) {
    return 0;
  }
  for (i = 0; i < t->rank; i++) {
    if (t->shape[i] == 0 || n > UINT64_MAX / t->shape[i]) {
      return 0;
    }
    n *= t->shape[i];
  }
  return n;
}

const char *
pai_opt_role_name(pai_opt_role_t role) {
  switch (role) {
  case PAI_OPT_ROLE_EMBEDDING:
    return "embedding";
  case PAI_OPT_ROLE_ATTENTION:
    return "attention";
  case PAI_OPT_ROLE_MLP:
    return "mlp";
  case PAI_OPT_ROLE_OUTPUT:
    return "output";
  case PAI_OPT_ROLE_NORM:
    return "norm";
  default:
    return "other";
  }
}

const char *
pai_opt_scheme_name(pai_opt_scheme_t scheme) {
  switch (scheme) {
  case PAI_OPT_SCHEME_F32:
    return "f32";
  case PAI_OPT_SCHEME_Q8:
    return "q8";
  case PAI_OPT_SCHEME_Q4:
    return "q4";
  default:
    return "?";
  }
}

pai_opt_role_t
pai_opt_classify(const char *tensor_name) {
  char buf[64];
  size_t i;

  if (tensor_name == NULL) {
    return PAI_OPT_ROLE_OTHER;
  }
  for (i = 0; tensor_name[i] != '\0' && i + 1 < sizeof(buf); i++) {
    char ch = tensor_name[i];
    if (ch >= 'A' && ch <= 'Z') {
      ch = (char)(ch - 'A' + 'a');
    }
    buf[i] = ch;
  }
  buf[i] = '\0';

  /* Match order matters: norms win over the "attn"/"output" inside
   * composite names ("attn_norm"), attention wins over the "output"
   * of "attn_output" (the Wo projection), and so on. Covers the
   * importer DSL names, LLaMA/HF style ("layers.0.mlp.gate_proj") and
   * GGUF style ("blk.0.ffn_gate.weight", "blk.0.attn_q.weight"). */
  if (strstr(buf, "embed") != NULL || strstr(buf, "embd") != NULL ||
      strstr(buf, "wte") != NULL || strstr(buf, "tok_emb") != NULL) {
    return PAI_OPT_ROLE_EMBEDDING;
  }
  if (strstr(buf, "norm") != NULL) {
    return PAI_OPT_ROLE_NORM;
  }
  if (strstr(buf, "attn") != NULL || strstr(buf, "attention") != NULL ||
      strstr(buf, "wq") != NULL || strstr(buf, "wk") != NULL ||
      strstr(buf, "wv") != NULL || strstr(buf, "wo") != NULL ||
      strstr(buf, "q_proj") != NULL || strstr(buf, "k_proj") != NULL ||
      strstr(buf, "v_proj") != NULL || strstr(buf, "o_proj") != NULL) {
    return PAI_OPT_ROLE_ATTENTION;
  }
  if (strstr(buf, "lm_head") != NULL || strstr(buf, "output") != NULL ||
      strstr(buf, "w_out") != NULL || strstr(buf, "head") != NULL) {
    return PAI_OPT_ROLE_OUTPUT;
  }
  if (strstr(buf, "mlp") != NULL || strstr(buf, "feed_forward") != NULL ||
      strstr(buf, "ffn") != NULL || strstr(buf, "gate_proj") != NULL ||
      strstr(buf, "up_proj") != NULL || strstr(buf, "down_proj") != NULL ||
      strstr(buf, "w1") != NULL || strstr(buf, "w2") != NULL ||
      strstr(buf, "w3") != NULL) {
    return PAI_OPT_ROLE_MLP;
  }
  return PAI_OPT_ROLE_OTHER;
}

/* Whether a role may use a scheme (v0 policy, §15): norms always stay
 * exact, embeddings and the output head cap at q8, attention/mlp/other
 * may go all the way to q4. */
static int
role_allows(pai_opt_role_t role, pai_opt_scheme_t scheme) {
  if (scheme == PAI_OPT_SCHEME_F32) {
    return 1;
  }
  if (scheme == PAI_OPT_SCHEME_Q8) {
    return role != PAI_OPT_ROLE_NORM;
  }
  return role == PAI_OPT_ROLE_ATTENTION || role == PAI_OPT_ROLE_MLP ||
         role == PAI_OPT_ROLE_OTHER;
}

/* Role default scheme (whitepaper §15 example table mapped onto the
 * v0 schemes). */
static pai_opt_scheme_t
role_default(pai_opt_role_t role) {
  switch (role) {
  case PAI_OPT_ROLE_EMBEDDING:
  case PAI_OPT_ROLE_ATTENTION:
  case PAI_OPT_ROLE_OUTPUT:
    return PAI_OPT_SCHEME_Q8;
  case PAI_OPT_ROLE_MLP:
    return PAI_OPT_SCHEME_Q4;
  case PAI_OPT_ROLE_NORM:
  default:
    return PAI_OPT_SCHEME_F32;
  }
}

/* Cheapest scheme a role may ever use. */
static pai_opt_scheme_t
role_cheapest(pai_opt_role_t role) {
  if (role == PAI_OPT_ROLE_NORM) {
    return PAI_OPT_SCHEME_F32;
  }
  if (role == PAI_OPT_ROLE_EMBEDDING || role == PAI_OPT_ROLE_OUTPUT) {
    return PAI_OPT_SCHEME_Q8;
  }
  return PAI_OPT_SCHEME_Q4;
}

/* ------------------------------------------------------------------ */
/* Plan                                                                */
/* ------------------------------------------------------------------ */

pai_status_t
pai_opt_plan(const pai_pai_container_t *c, const pai_ir_program_t *ir,
             uint64_t budget, uint16_t group_size, pai_opt_plan_t *out) {
  pai_pai_meta_t meta;
  const uint8_t *sec;
  uint32_t sec_size;
  pai_pai_tensor_t tensors[PAI_PAI_MAX_TENSORS];
  uint32_t count = 0;
  uint32_t i;
  pai_status_t st;

  if (c == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));

  sec = pai_pai_section(c, PAI_PAI_SEC_META, &sec_size);
  if (sec == NULL) {
    return PAI_ERR_PROTOCOL;
  }
  st = pai_pai_meta_decode(sec, sec_size, &meta);
  if (st != PAI_OK) {
    return st;
  }
  memcpy(out->name, meta.name, sizeof(meta.name));

  sec = pai_pai_section(c, PAI_PAI_SEC_MANIFEST, &sec_size);
  if (sec == NULL) {
    return PAI_ERR_PROTOCOL;
  }
  st = pai_pai_manifest_decode(sec, sec_size, tensors, PAI_PAI_MAX_TENSORS,
                               &count);
  if (st != PAI_OK) {
    return st;
  }
  if (count > PAI_OPT_MAX_TENSORS) {
    count = PAI_OPT_MAX_TENSORS;
  }

  out->budget = budget;
  out->num_tensors = count;
  for (i = 0; i < count; i++) {
    pai_opt_tensor_t *t = &out->tensors[i];
    pai_quant_scheme_t qs;
    uint64_t n;

    memcpy(t->name, tensors[i].name, sizeof(t->name));
    t->value_id = tensors[i].value_id;
    t->role = pai_opt_classify(t->name);
    n = tensor_numel(&tensors[i]);
    t->numel = n;
    t->cur_bytes = tensors[i].size_bytes;

    t->size_bytes[PAI_OPT_SCHEME_F32] = n * sizeof(float);
    qs.bit_width = 8;
    qs.is_signed = 1;
    qs.group_size = group_size;
    t->size_bytes[PAI_OPT_SCHEME_Q8] = pai_quant_weights_bytes(n, &qs);
    qs.bit_width = 4;
    t->size_bytes[PAI_OPT_SCHEME_Q4] = pai_quant_weights_bytes(n, &qs);

    /* Current scheme: IR quant metadata first, then an inference from
     * the stored size, then f32. */
    t->cur_scheme = PAI_OPT_SCHEME_F32;
    if (ir != NULL && t->value_id > 0 && t->value_id <= ir->num_values) {
      const pai_ir_value_t *iv = &ir->values[t->value_id];
      if (iv->quant.present) {
        if (iv->quant.bit_width == 4) {
          t->cur_scheme = PAI_OPT_SCHEME_Q4;
        } else if (iv->quant.bit_width == 8) {
          t->cur_scheme = PAI_OPT_SCHEME_Q8;
        }
      }
    }
    if (t->cur_scheme == PAI_OPT_SCHEME_F32 && n > 0 &&
        t->cur_bytes < n * sizeof(float)) {
      if (t->cur_bytes == t->size_bytes[PAI_OPT_SCHEME_Q4]) {
        t->cur_scheme = PAI_OPT_SCHEME_Q4;
      } else if (t->cur_bytes == t->size_bytes[PAI_OPT_SCHEME_Q8]) {
        t->cur_scheme = PAI_OPT_SCHEME_Q8;
      } else if (t->cur_bytes <= n) {
        t->cur_scheme = PAI_OPT_SCHEME_Q4; /* dense packed values */
      } else {
        t->cur_scheme = PAI_OPT_SCHEME_Q8;
      }
    }

    out->cur_total += t->cur_bytes;
    out->min_total += t->size_bytes[role_cheapest(t->role)];
  }

  if (budget == 0) {
    /* Role-default plan, clamped so a tensor already stored more
     * cheaply than its role default is never re-inflated. */
    for (i = 0; i < count; i++) {
      pai_opt_tensor_t *t = &out->tensors[i];
      pai_opt_scheme_t def = role_default(t->role);
      t->chosen = def > t->cur_scheme ? def : t->cur_scheme;
      out->plan_total += t->size_bytes[t->chosen];
    }
    out->feasible = 1;
  } else {
    /* Greedy largest-savings fit: start from the current storage and
     * keep applying the single most profitable allowed downgrade until
     * the total fits the budget (or no cheaper layout exists). */
    for (i = 0; i < count; i++) {
      out->tensors[i].chosen = out->tensors[i].cur_scheme;
      out->plan_total += out->tensors[i].size_bytes[out->tensors[i].chosen];
    }
    while (out->plan_total > budget) {
      uint64_t best_save = 0;
      uint32_t best_idx = count; /* none */
      for (i = 0; i < count; i++) {
        pai_opt_tensor_t *t = &out->tensors[i];
        pai_opt_scheme_t next = (pai_opt_scheme_t)(t->chosen + 1);
        uint64_t save;
        if (next >= PAI_OPT_SCHEME_COUNT || !role_allows(t->role, next)) {
          continue;
        }
        save = t->size_bytes[t->chosen] - t->size_bytes[next];
        if (save > best_save || (save == best_save && i < best_idx)) {
          best_save = save;
          best_idx = i;
        }
      }
      if (best_idx == count) {
        break; /* no cheaper layout available */
      }
      out->tensors[best_idx].chosen =
          (pai_opt_scheme_t)(out->tensors[best_idx].chosen + 1);
      out->plan_total -= best_save;
    }
    out->feasible = out->plan_total <= budget;
  }

  return PAI_OK;
}
