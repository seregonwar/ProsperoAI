/*
 * ProsperoAI — LLaMA-family adapter (see llama.h)
 *
 * The translator rebuilds the model as a sequence-mode compute graph:
 *
 *   input [ctx, vocab] (one-hot) -> embed gemm -> per-layer block ->
 *   final rmsnorm -> lm_head gemm -> logits [ctx, vocab]
 *
 * Per layer (input x [ctx, n_embd]):
 *   xn   = rmsnorm(x, attn_gamma)
 *   q/k/v = gemm(xn, wq/wk/wv)          (weights stored k-major for the
 *                                        C[m,n]=A[m,k]*B[k,n] executor)
 *   q/k/v reshaped to [heads, seq, hd] for RoPE + attention
 *   a    = causal_mha(rope(q), rope(k), v)
 *   o    = gemm(a, wo)
 *   x    = x + o
 *   xn2  = rmsnorm(x, ffn_gamma)
 *   gated= silu(gemm(xn2, wgate)) * gemm(xn2, wup)
 *   x    = x + gemm(gated, wdown)
 *
 * All rank-2 GGUF weights are transposed from their native [out, in]
 * layout to [in, out]; token embeddings stay native (they already
 * match the executor's k-major convention). RoPE cos/sin tables are
 * params excluded from quantization (values in [-1, 1] would not
 * survive the q8/q4 packers).
 */

#include "llama.h"

#include "gguf.h"

#include <pai/pai.h>
#include <tokenizer.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct lla_build {
  pai_pai_meta_t meta;
  pai_container_tensor_t *tensors;
  uint32_t num_tensors;
  uint32_t cap_tensors;
  pai_container_op_t *ops;
  uint32_t num_ops;
  uint32_t cap_ops;
  pai_container_token_t *toks;
  uint32_t num_toks;
  pai_container_merge_t *merges;
  uint32_t num_merges;
  void **owned; /* malloc'd buffers (weights, op id arrays, names) */
  uint32_t num_owned;
  uint32_t cap_owned;
  pai_status_t st; /* sticky error for the own/dup helpers        */
} lla_build_t;

static char *own_str_dup(lla_build_t *b, const char *s);

static void *
xmalloc(size_t n) {
  return malloc(n > 0 ? n : 1);
}

static void
own(lla_build_t *b, void *p) {
  if (p == NULL) {
    return;
  }
  if (b->num_owned >= b->cap_owned) {
    uint32_t ncap = b->cap_owned == 0 ? 128 : b->cap_owned * 2;
    void **nw = (void **)realloc(b->owned, (size_t)ncap * sizeof(*nw));
    if (nw == NULL) {
      b->st = PAI_ERR_NOMEM;
      return;
    }
    b->owned = nw;
    b->cap_owned = ncap;
  }
  b->owned[b->num_owned++] = p;
}

static char *
own_str_dup(lla_build_t *b, const char *s) {
  char *p = (char *)xmalloc(strlen(s) + 1);
  if (p == NULL) {
    b->st = PAI_ERR_NOMEM;
    return NULL;
  }
  strcpy(p, s);
  own(b, p);
  return p;
}

static pai_container_tensor_t *
add_tensor(lla_build_t *b) {
  pai_container_tensor_t *t;

  if (b->num_tensors >= b->cap_tensors) {
    uint32_t ncap = b->cap_tensors == 0 ? 64 : b->cap_tensors * 2;
    pai_container_tensor_t *nw = (pai_container_tensor_t *)realloc(
        b->tensors, (size_t)ncap * sizeof(*nw));
    if (nw == NULL) {
      b->st = PAI_ERR_NOMEM; /* sticky: callers index b->tensors[id-1] */
      return NULL;
    }
    b->tensors = nw;
    b->cap_tensors = ncap;
  }
  t = &b->tensors[b->num_tensors];
  memset(t, 0, sizeof(*t));
  t->id = b->num_tensors + 1;
  t->dtype = PAI_DTYPE_F32;
  t->quantize = 1;
  b->num_tensors++;
  return t;
}

static pai_container_op_t *
add_op(lla_build_t *b, uint32_t kind, uint32_t num_in, uint32_t num_out,
       const uint32_t *in, const uint32_t *out) {
  pai_container_op_t *o;
  uint32_t *in_c;
  uint32_t *out_c;

  if (b->num_ops >= b->cap_ops) {
    uint32_t ncap = b->cap_ops == 0 ? 64 : b->cap_ops * 2;
    pai_container_op_t *nw =
        (pai_container_op_t *)realloc(b->ops, (size_t)ncap * sizeof(*nw));
    if (nw == NULL) {
      return NULL;
    }
    b->ops = nw;
    b->cap_ops = ncap;
  }
  o = &b->ops[b->num_ops];
  memset(o, 0, sizeof(*o));
  o->id = b->num_ops + 1;
  o->kind = kind;
  o->num_in = num_in;
  o->num_out = num_out;
  if (num_in > 0) {
    in_c = (uint32_t *)xmalloc((size_t)num_in * sizeof(uint32_t));
    if (in_c == NULL) {
      return NULL;
    }
    memcpy(in_c, in, (size_t)num_in * sizeof(uint32_t));
    o->in = in_c;
    own(b, in_c);
  }
  if (num_out > 0) {
    out_c = (uint32_t *)xmalloc((size_t)num_out * sizeof(uint32_t));
    if (out_c == NULL) {
      return NULL;
    }
    memcpy(out_c, out, (size_t)num_out * sizeof(uint32_t));
    o->out = out_c;
    own(b, out_c);
  }
  b->num_ops++;
  return o;
}

static uint32_t
act(lla_build_t *b, uint32_t rank, uint64_t d0, uint64_t d1, uint64_t d2) {
  pai_container_tensor_t *t = add_tensor(b);
  if (t == NULL) {
    return 0;
  }
  t->kind = 2;
  t->rank = rank;
  t->shape[0] = d0;
  if (rank > 1) {
    t->shape[1] = d1;
  }
  if (rank > 2) {
    t->shape[2] = d2;
  }
  return t->id;
}

static uint32_t
param(lla_build_t *b, uint32_t rank, uint64_t d0, uint64_t d1, uint64_t d2,
      const char *name, const float *data, uint64_t n, int quantize) {
  pai_container_tensor_t *t = add_tensor(b);
  if (t == NULL) {
    return 0;
  }
  t->kind = 1;
  t->rank = rank;
  t->shape[0] = d0;
  if (rank > 1) {
    t->shape[1] = d1;
  }
  if (rank > 2) {
    t->shape[2] = d2;
  }
  t->name = name;
  t->data = data;
  t->n = n;
  t->quantize = (uint8_t)(quantize != 0);
  return t->id;
}

/* Transpose a rows x cols buffer into cols x rows (new allocation). */
static float *
transpose2d(const float *src, uint64_t rows, uint64_t cols) {
  float *out = (float *)xmalloc((size_t)(rows * cols) * sizeof(float));
  if (out == NULL) {
    return NULL;
  }
  for (uint64_t r = 0; r < rows; r++) {
    for (uint64_t c = 0; c < cols; c++) {
      out[c * rows + r] = src[r * cols + c];
    }
  }
  return out;
}

/* Unique token id for `text`, or -1 when absent/ambiguous. */
static int
find_token(const char *const *tokens, uint64_t n, const char *text) {
  int found = -1;
  uint64_t i;

  for (i = 0; i < n; i++) {
    if (strcmp(tokens[i], text) == 0) {
      if (found >= 0) {
        return -1;
      }
      found = (int)i;
    }
  }
  return found;
}

pai_status_t
pai_llama_import(const char *gguf_path, const char *out_path,
                 const pai_container_quant_t *quant) {
  FILE *f = NULL;
  uint8_t *file = NULL;
  long nbytes = 0;
  pai_gguf_t gg;
  lla_build_t b;
  char arch[64];
  char key[64];
  char nm[PAI_PAI_NAME_MAX + 1];
  char tname[160];
  uint32_t n_layer = 0;
  uint32_t n_embd = 0;
  uint32_t n_ff = 0;
  uint32_t n_head = 0;
  uint32_t n_kv = 0;
  uint32_t ctx = 0;
  uint32_t n_rot = 0;
  float theta = 10000.0f;
  uint64_t vocab_n = 0;
  char **tokens = NULL;
  char **merges = NULL;
  uint64_t merges_n = 0;
  uint32_t hd;
  uint32_t kv_embd;
  uint32_t r2;
  uint32_t total_slots;
  float **slots;
  uint64_t *slot_n;
  uint32_t L;
  uint32_t i;
  pai_status_t st;

  if (gguf_path == NULL || out_path == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(&b, 0, sizeof(b));
  memset(&gg, 0, sizeof(gg));

  /* Read the whole file (v0: host tooling; streaming arrives with the
   * Desktop transfer pipeline). */
  f = fopen(gguf_path, "rb");
  if (f == NULL) {
    return PAI_ERR_IO;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    st = PAI_ERR_IO;
    goto done;
  }
  nbytes = ftell(f);
  if (nbytes < 0 || (uint64_t)nbytes > (UINT64_C(2) << 30)) {
    st = PAI_ERR_UNSUPPORTED;
    goto done;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    st = PAI_ERR_IO;
    goto done;
  }
  file = (uint8_t *)xmalloc((size_t)nbytes);
  if (file == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }
  if (fread(file, 1, (size_t)nbytes, f) != (size_t)nbytes) {
    st = PAI_ERR_IO;
    goto done;
  }
  fclose(f);
  f = NULL;

  st = pai_gguf_open(file, (uint64_t)nbytes, &gg);
  if (st != PAI_OK) {
    goto done;
  }

  /* Architecture. */
  st = pai_gguf_get_string(&gg, "general.architecture", arch, sizeof(arch));
  if (st != PAI_OK) {
    goto done;
  }
  if (strcmp(arch, PAI_LLAMA_ARCH_LLAMA) != 0 &&
      strcmp(arch, PAI_LLAMA_ARCH_MISTRAL) != 0) {
    st = PAI_ERR_UNSUPPORTED;
    goto done;
  }

  /* Configuration (keys are prefixed with the architecture). */
  snprintf(key, sizeof(key), "%s.block_count", arch);
  st = pai_gguf_get_u32(&gg, key, &n_layer);
  if (st != PAI_OK) {
    goto done;
  }
  snprintf(key, sizeof(key), "%s.embedding_length", arch);
  st = pai_gguf_get_u32(&gg, key, &n_embd);
  if (st != PAI_OK) {
    goto done;
  }
  snprintf(key, sizeof(key), "%s.feed_forward_length", arch);
  st = pai_gguf_get_u32(&gg, key, &n_ff);
  if (st != PAI_OK) {
    goto done;
  }
  snprintf(key, sizeof(key), "%s.attention.head_count", arch);
  st = pai_gguf_get_u32(&gg, key, &n_head);
  if (st != PAI_OK) {
    goto done;
  }
  snprintf(key, sizeof(key), "%s.attention.head_count_kv", arch);
  if (pai_gguf_get_u32(&gg, key, &n_kv) != PAI_OK) {
    n_kv = n_head;
  }
  snprintf(key, sizeof(key), "%s.context_length", arch);
  st = pai_gguf_get_u32(&gg, key, &ctx);
  if (st != PAI_OK) {
    goto done;
  }
  snprintf(key, sizeof(key), "%s.rope.dimension_count", arch);
  if (pai_gguf_get_u32(&gg, key, &n_rot) != PAI_OK) {
    n_rot = 0;
  }
  snprintf(key, sizeof(key), "%s.rope.freq_base", arch);
  if (pai_gguf_get_f32(&gg, key, &theta) != PAI_OK) {
    theta = 10000.0f;
  }

  /* Vocabulary. */
  st = pai_gguf_get_string_array(&gg, "tokenizer.ggml.tokens", &tokens,
                                 &vocab_n);
  if (st != PAI_OK) {
    goto done;
  }
  (void)pai_gguf_get_string_array(&gg, "tokenizer.ggml.merges", &merges,
                                  &merges_n);

  /* Shape validation. */
  if (n_layer == 0 || n_embd == 0 || n_ff == 0 || n_head == 0 || ctx == 0 ||
      vocab_n == 0 || n_embd % n_head != 0 || n_kv == 0 || n_head % n_kv != 0 ||
      ctx > PAI_TOK_MAX_INPUT) {
    st = PAI_ERR_MISMATCH;
    goto done;
  }
  hd = n_embd / n_head;
  kv_embd = hd * n_kv;
  if (n_rot == 0 || n_rot > hd || (n_rot & 1) != 0) {
    n_rot = hd;
  }
  r2 = n_rot / 2;

  /* ---- Weight slots.
   *   slot 0                     token_embd  [vocab, n_embd]  (native)
   *   slot 1 + L*9 + {0..8}      blk.L.*     (transposed 2D, vectors raw)
   *   slot 1 + n_layer*9         output      [n_embd, vocab]  (transposed)
   * ---- */
  total_slots = 1 + 9 * n_layer + 1;
  slots = (float **)calloc(total_slots, sizeof(*slots));
  slot_n = (uint64_t *)calloc(total_slots, sizeof(*slot_n));
  if (slots == NULL || slot_n == NULL) {
    free(slots);
    free(slot_n);
    st = PAI_ERR_NOMEM;
    goto done;
  }

  {
    static const char *const layer_names[] = {
        "attn_norm.weight", "attn_q.weight",     "attn_k.weight",
        "attn_v.weight",    "attn_output.weight", "ffn_norm.weight",
        "ffn_gate.weight",  "ffn_up.weight",      "ffn_down.weight",
    };
    const pai_gguf_tensor_t *gt;
    float *dq;
    uint64_t n;

    /* token_embd.weight: native k-major. */
    gt = pai_gguf_find_tensor(&gg, "token_embd.weight");
    if (gt == NULL) {
      st = PAI_ERR_MISMATCH;
      goto slots_done;
    }
    st = pai_gguf_dequant(&gg, gt, &dq, &n);
    if (st != PAI_OK) {
      goto slots_done;
    }
    slots[0] = dq;
    slot_n[0] = n;
    own(&b, dq);

    /* Layer weights. */
    for (L = 0; L < n_layer; L++) {
      for (i = 0; i < 9; i++) {
        uint32_t slot = 1 + L * 9 + i;
        snprintf(tname, sizeof(tname), "blk.%u.%s", L, layer_names[i]);
        gt = pai_gguf_find_tensor(&gg, tname);
        if (gt == NULL) {
          st = PAI_ERR_MISMATCH;
          goto slots_done;
        }
        st = pai_gguf_dequant(&gg, gt, &dq, &n);
        if (st != PAI_OK) {
          goto slots_done;
        }
        if (gt->n_dims == 1) {
          slots[slot] = dq; /* vector: no transpose */
        } else {
          slots[slot] = transpose2d(dq, gt->dims[1], gt->dims[0]);
          free(dq);
          if (slots[slot] == NULL) {
            st = PAI_ERR_NOMEM;
            goto slots_done;
          }
        }
        slot_n[slot] = n;
        own(&b, slots[slot]);
      }
    }

    /* output.weight (optional; ties to the embedding when absent). */
    {
      uint32_t slot = 1 + n_layer * 9;
      gt = pai_gguf_find_tensor(&gg, "output.weight");
      if (gt != NULL) {
        st = pai_gguf_dequant(&gg, gt, &dq, &n);
        if (st != PAI_OK) {
          goto slots_done;
        }
        slots[slot] = transpose2d(dq, gt->dims[1], gt->dims[0]);
        free(dq);
        if (slots[slot] == NULL) {
          st = PAI_ERR_NOMEM;
          goto slots_done;
        }
      } else {
        slots[slot] = transpose2d(slots[0], vocab_n, n_embd);
        n = vocab_n * n_embd;
        if (slots[slot] == NULL) {
          st = PAI_ERR_NOMEM;
          goto slots_done;
        }
      }
      slot_n[slot] = n;
      own(&b, slots[slot]);
    }
  }

slots_done:
  if (st != PAI_OK) {
    free(slots);
    free(slot_n);
    goto done;
  }

  /* ---- Graph values (declaration order = id order). ---- */
  {
    uint32_t in_id;
    uint32_t x_id;
    uint32_t embed_id;
    uint32_t cos_id;
    uint32_t sin_id;
    uint32_t logits_id;
    uint32_t xf_id;

    in_id = act(&b, 2, ctx, vocab_n, 0);
    if (in_id == 0) {
      st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
      goto build_done;
    }
    {
      pai_container_tensor_t *t = &b.tensors[in_id - 1];
      t->kind = 0; /* graph input */
    }
    x_id = act(&b, 2, ctx, n_embd, 0); /* h0: embed output */
    embed_id = param(&b, 2, vocab_n, n_embd, 0, "token_embd", slots[0],
                     slot_n[0], 1);

    /* RoPE tables must exist before the per-layer ops reference them. */
    cos_id = param(&b, 2, ctx, r2, 0, "rope.cos", NULL, 0, 0);
    sin_id = param(&b, 2, ctx, r2, 0, "rope.sin", NULL, 0, 0);
    if (b.st != PAI_OK) {
      st = b.st;
      goto build_done;
    }

    /* Embedding: one-hot [ctx, vocab] x token_embd [vocab, n_embd]. */
    {
      uint32_t in2[2];
      uint32_t out1[1];
      in2[0] = in_id;
      in2[1] = embed_id;
      out1[0] = x_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
    }

    /* Per layer. */
    for (L = 0; L < n_layer; L++) {
      uint32_t gamma_id, xn_id, wq_id, q_id, wk_id, k_id, wv_id, v_id;
      uint32_t qr_id, kr_id, vr_id, qr2_id, kr2_id, a_id, a2_id, wo_id;
      uint32_t o_id, x1_id, fgamma_id, hn_id, wgate_id, gate_id, wup_id;
      uint32_t up_id, g_id, gated_id, wdown_id, down_id, x2_id;
      uint32_t in3[3];
      uint32_t in2[2];
      uint32_t out1[1];

      if (b.st != PAI_OK) { /* any param/act allocation failed */
        st = b.st;
        goto build_done;
      }
      snprintf(nm, sizeof(nm), "blk.%u.attn_norm.weight", L);
      gamma_id = param(&b, 1, n_embd, 0, 0, NULL, NULL, 0, 1);
      if (gamma_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[gamma_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 0];
        t->n = slot_n[1 + L * 9 + 0];
      }
      xn_id = act(&b, 2, ctx, n_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.attn_q.weight", L);
      wq_id = param(&b, 2, n_embd, n_embd, 0, NULL, NULL, 0, 1);
      if (wq_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wq_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 1];
        t->n = slot_n[1 + L * 9 + 1];
      }
      q_id = act(&b, 2, ctx, n_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.attn_k.weight", L);
      wk_id = param(&b, 2, n_embd, kv_embd, 0, NULL, NULL, 0, 1);
      if (wk_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wk_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 2];
        t->n = slot_n[1 + L * 9 + 2];
      }
      k_id = act(&b, 2, ctx, kv_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.attn_v.weight", L);
      wv_id = param(&b, 2, n_embd, kv_embd, 0, NULL, NULL, 0, 1);
      if (wv_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wv_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 3];
        t->n = slot_n[1 + L * 9 + 3];
      }
      v_id = act(&b, 2, ctx, kv_embd, 0);
      /* Position-major [seq, H/K, hd]: a flat reshape of the
       * [seq, n_embd] activation (per-head column blocks) lands head
       * h at columns h*hd..(h+1)*hd of each position row. */
      qr_id = act(&b, 3, ctx, n_head, hd);
      kr_id = act(&b, 3, ctx, n_kv, hd);
      vr_id = act(&b, 3, ctx, n_kv, hd);
      qr2_id = act(&b, 3, ctx, n_head, hd);
      kr2_id = act(&b, 3, ctx, n_kv, hd);
      a_id = act(&b, 3, ctx, n_head, hd);
      a2_id = act(&b, 2, ctx, n_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.attn_output.weight", L);
      wo_id = param(&b, 2, n_embd, n_embd, 0, NULL, NULL, 0, 1);
      if (wo_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wo_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 4];
        t->n = slot_n[1 + L * 9 + 4];
      }
      o_id = act(&b, 2, ctx, n_embd, 0);
      x1_id = act(&b, 2, ctx, n_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.ffn_norm.weight", L);
      fgamma_id = param(&b, 1, n_embd, 0, 0, NULL, NULL, 0, 1);
      if (fgamma_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[fgamma_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 5];
        t->n = slot_n[1 + L * 9 + 5];
      }
      hn_id = act(&b, 2, ctx, n_embd, 0);
      snprintf(nm, sizeof(nm), "blk.%u.ffn_gate.weight", L);
      wgate_id = param(&b, 2, n_embd, n_ff, 0, NULL, NULL, 0, 1);
      if (wgate_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wgate_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 6];
        t->n = slot_n[1 + L * 9 + 6];
      }
      gate_id = act(&b, 2, ctx, n_ff, 0);
      snprintf(nm, sizeof(nm), "blk.%u.ffn_up.weight", L);
      wup_id = param(&b, 2, n_embd, n_ff, 0, NULL, NULL, 0, 1);
      if (wup_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wup_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 7];
        t->n = slot_n[1 + L * 9 + 7];
      }
      up_id = act(&b, 2, ctx, n_ff, 0);
      g_id = act(&b, 2, ctx, n_ff, 0);
      gated_id = act(&b, 2, ctx, n_ff, 0);
      snprintf(nm, sizeof(nm), "blk.%u.ffn_down.weight", L);
      wdown_id = param(&b, 2, n_ff, n_embd, 0, NULL, NULL, 0, 1);
      if (wdown_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[wdown_id - 1];
        t->name = own_str_dup(&b, nm);
        t->data = slots[1 + L * 9 + 8];
        t->n = slot_n[1 + L * 9 + 8];
      }
      down_id = act(&b, 2, ctx, n_embd, 0);
      x2_id = act(&b, 2, ctx, n_embd, 0);

      /* ---- Ops. ---- */
      in2[0] = x_id;
      in2[1] = gamma_id;
      out1[0] = xn_id;
      if (add_op(&b, PAI_OP_RMSNORM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = xn_id;
      in2[1] = wq_id;
      out1[0] = q_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[1] = wk_id;
      out1[0] = k_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[1] = wv_id;
      out1[0] = v_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      out1[0] = qr_id;
      if (add_op(&b, PAI_OP_RESHAPE, 1, 1, &q_id, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      out1[0] = kr_id;
      if (add_op(&b, PAI_OP_RESHAPE, 1, 1, &k_id, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      out1[0] = vr_id;
      if (add_op(&b, PAI_OP_RESHAPE, 1, 1, &v_id, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in3[0] = qr_id;
      in3[1] = cos_id;
      in3[2] = sin_id;
      out1[0] = qr2_id;
      if (add_op(&b, PAI_OP_ROPE, 3, 1, in3, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in3[0] = kr_id;
      out1[0] = kr2_id;
      if (add_op(&b, PAI_OP_ROPE, 3, 1, in3, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in3[0] = qr2_id;
      in3[1] = kr2_id;
      in3[2] = vr_id;
      out1[0] = a_id;
      if (add_op(&b, PAI_OP_ATTENTION, 3, 1, in3, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      out1[0] = a2_id;
      if (add_op(&b, PAI_OP_RESHAPE, 1, 1, &a_id, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = a2_id;
      in2[1] = wo_id;
      out1[0] = o_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = x_id;
      in2[1] = o_id;
      out1[0] = x1_id;
      if (add_op(&b, PAI_OP_ADD, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = x1_id;
      in2[1] = fgamma_id;
      out1[0] = hn_id;
      if (add_op(&b, PAI_OP_RMSNORM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = hn_id;
      in2[1] = wgate_id;
      out1[0] = gate_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[1] = wup_id;
      out1[0] = up_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      out1[0] = g_id;
      if (add_op(&b, PAI_OP_SILU, 1, 1, &gate_id, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = g_id;
      in2[1] = up_id;
      out1[0] = gated_id;
      if (add_op(&b, PAI_OP_MUL, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = gated_id;
      in2[1] = wdown_id;
      out1[0] = down_id;
      if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      in2[0] = x1_id;
      in2[1] = down_id;
      out1[0] = x2_id;
      if (add_op(&b, PAI_OP_ADD, 2, 1, in2, out1) == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      x_id = x2_id;
    }

    /* Final rmsnorm + LM head + logits. */
    {
      uint32_t out_gamma_id;
      uint32_t lm_id;
      float *cos_buf;
      float *sin_buf;

      out_gamma_id = param(&b, 1, n_embd, 0, 0, NULL, NULL, 0, 1);
      if (b.st != PAI_OK) {
        st = b.st;
        goto build_done;
      }
      /* The final norm gamma is the "output_norm.weight" GGUF tensor. */
      {
        const pai_gguf_tensor_t *gt =
            pai_gguf_find_tensor(&gg, "output_norm.weight");
        float *dq;
        uint64_t n;
        if (gt == NULL) {
          st = PAI_ERR_MISMATCH;
          goto build_done;
        }
        st = pai_gguf_dequant(&gg, gt, &dq, &n);
        if (st != PAI_OK) {
          goto build_done;
        }
        own(&b, dq);
        {
          pai_container_tensor_t *t = &b.tensors[out_gamma_id - 1];
          t->name = "output_norm.weight";
          t->data = dq;
          t->n = n;
        }
      }
      xf_id = act(&b, 2, ctx, n_embd, 0);
      lm_id = param(&b, 2, n_embd, vocab_n, 0, "output.weight",
                    slots[1 + n_layer * 9], slot_n[1 + n_layer * 9], 1);
      logits_id = act(&b, 2, ctx, vocab_n, 0);
      if (logits_id == 0) {
        st = b.st != PAI_OK ? b.st : PAI_ERR_NOMEM;
        goto build_done;
      }
      {
        pai_container_tensor_t *t = &b.tensors[logits_id - 1];
        t->kind = 3; /* graph output */
      }

      cos_buf = (float *)xmalloc((size_t)ctx * r2 * sizeof(float));
      sin_buf = (float *)xmalloc((size_t)ctx * r2 * sizeof(float));
      if (cos_buf == NULL || sin_buf == NULL) {
        free(cos_buf);
        free(sin_buf);
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      for (uint32_t p = 0; p < ctx; p++) {
        for (i = 0; i < r2; i++) {
          float freq = powf(theta, -(float)(2 * i) / (float)n_rot);
          cos_buf[p * r2 + i] = cosf((float)p * freq);
          sin_buf[p * r2 + i] = sinf((float)p * freq);
        }
      }
      {
        pai_container_tensor_t *t = &b.tensors[cos_id - 1];
        t->data = cos_buf;
        t->n = (uint64_t)ctx * r2;
      }
      {
        pai_container_tensor_t *t = &b.tensors[sin_id - 1];
        t->data = sin_buf;
        t->n = (uint64_t)ctx * r2;
      }
      own(&b, cos_buf);
      own(&b, sin_buf);

      /* Final ops. */
      {
        uint32_t in2[2];
        uint32_t out1[1];
        in2[0] = x_id;
        in2[1] = out_gamma_id;
        out1[0] = xf_id;
        if (add_op(&b, PAI_OP_RMSNORM, 2, 1, in2, out1) == NULL) {
          st = PAI_ERR_NOMEM;
          goto build_done;
        }
        in2[0] = xf_id;
        in2[1] = lm_id;
        out1[0] = logits_id;
        if (add_op(&b, PAI_OP_GEMM, 2, 1, in2, out1) == NULL) {
          st = PAI_ERR_NOMEM;
          goto build_done;
        }
      }
    }

    /* ---- Tokenizer descriptors. ----
     * Skip tokens the container tokenizer cannot hold (empty text or
     * > PAI_TOK_MAX_TOKEN_LEN chars): the blob stores each text length
     * in a u8. Sparse ids are fine (the tokenizer keeps an id per
     * token); merges referencing skipped tokens are dropped below. */
    {
      uint64_t tok_count = 0;
      for (i = 0; i < vocab_n; i++) {
        size_t tl = strlen(tokens[i]);
        if (tl > 0 && tl <= PAI_TOK_MAX_TOKEN_LEN) {
          tok_count++;
        }
      }
      b.toks = (pai_container_token_t *)calloc(tok_count > 0 ? tok_count : 1,
                                               sizeof(*b.toks));
      if (b.toks == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      for (i = 0; i < vocab_n; i++) {
        size_t tl = strlen(tokens[i]);
        if (tl > 0 && tl <= PAI_TOK_MAX_TOKEN_LEN) {
          b.toks[b.num_toks].id = (uint32_t)i;
          b.toks[b.num_toks].text = tokens[i];
          b.num_toks++;
        }
      }
    }
    if (merges_n > 0) {
      uint32_t cap_merges = merges_n < PAI_TOK_MAX_MERGES
                                ? (uint32_t)merges_n
                                : PAI_TOK_MAX_MERGES;
      b.merges = (pai_container_merge_t *)calloc(cap_merges > 0 ? cap_merges : 1,
                                                 sizeof(*b.merges));
      if (b.merges == NULL) {
        st = PAI_ERR_NOMEM;
        goto build_done;
      }
      for (uint64_t mi = 0; mi < merges_n && b.num_merges < PAI_TOK_MAX_MERGES;
           mi++) {
        const char *m = merges[mi];
        const char *sp = strchr(m, ' ');
        char left[256];
        char right[256];
        char merged[512];
        int l_id;
        int r_id;
        int m_id;

        if (sp == NULL || (size_t)(sp - m) == 0 ||
            (size_t)(sp - m) >= sizeof(left)) {
          continue;
        }
        memcpy(left, m, (size_t)(sp - m));
        left[sp - m] = '\0';
        if (sp[1] == '\0' || strlen(sp + 1) >= sizeof(right)) {
          continue;
        }
        strcpy(right, sp + 1);
        if (strlen(left) + strlen(right) >= sizeof(merged)) {
          continue;
        }
        l_id = find_token(tokens, vocab_n, left);
        r_id = find_token(tokens, vocab_n, right);
        if (l_id < 0 || r_id < 0) {
          continue;
        }
        strcpy(merged, left);
        strcat(merged, right);
        m_id = find_token(tokens, vocab_n, merged);
        if (m_id < 0) {
          continue;
        }
        b.merges[b.num_merges].left = (uint32_t)l_id;
        b.merges[b.num_merges].right = (uint32_t)r_id;
        b.merges[b.num_merges].result = (uint32_t)m_id;
        b.num_merges++;
      }
    }

    /* ---- Metadata. ---- */
    memset(&b.meta, 0, sizeof(b.meta));
    snprintf(b.meta.name, sizeof(b.meta.name), "%s", "llama");
    if (pai_gguf_get_string(&gg, "general.name", nm, sizeof(nm)) == PAI_OK &&
        nm[0] != '\0') {
      snprintf(b.meta.name, sizeof(b.meta.name), "%s", nm);
    }
    b.meta.family = PAI_PAI_FAMILY_LLM;
    b.meta.context_len = ctx;
    b.meta.num_layers = n_layer;
    b.meta.kv_bytes_per_token = 0; /* v0: full-context recompute */
    b.meta.vocab_size = (uint32_t)vocab_n;

    /* ---- Container build + write. ---- */
    {
      uint8_t *blob = NULL;
      uint32_t blob_n = 0;
      st = pai_container_build(&b.meta, b.tensors, b.num_tensors, b.ops,
                               b.num_ops, b.toks, b.num_toks, b.merges,
                               b.num_merges, quant, &blob, &blob_n);
      if (st == PAI_OK) {
        st = pai_pai_write_file(out_path, blob, blob_n);
      }
      free(blob);
    }
  }

build_done:
  free(slots);
  free(slot_n);

done:
  if (f != NULL) {
    fclose(f);
  }
  free(file);
  free(tokens);
  free(merges);
  for (uint32_t oi = 0; oi < b.num_owned; oi++) {
    free(b.owned[oi]);
  }
  free(b.owned);
  free(b.tensors);
  free(b.ops);
  free(b.toks);
  free(b.merges);
  pai_gguf_close(&gg);
  return st;
}

int
pai_llama_is_gguf(const uint8_t *head4) {
  if (head4 == NULL) {
    return 0;
  }
  return ((uint32_t)head4[0] | ((uint32_t)head4[1] << 8) |
          ((uint32_t)head4[2] << 16) | ((uint32_t)head4[3] << 24)) ==
         PAI_GGUF_MAGIC;
}
