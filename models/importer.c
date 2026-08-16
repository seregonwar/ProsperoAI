#include "importer.h"

#include <container.h>

#include <graph/graph.h>
#include <ir/ir.h>
#include <pai/pai.h>

#include <quantize.h>
#include <tokenizer.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMP_MAX_LINE 512u
#define IMP_MAX_TEXT 64u

typedef struct imp_value {
  uint32_t id;
  uint32_t dtype; /* pai_dtype_t */
  uint32_t rank;
  uint64_t shape[PAI_TENSOR_MAX_RANK];
  uint32_t kind; /* 0 input, 1 param, 2 activation, 3 output */
  char weight_file[IMP_MAX_TEXT];
  int has_weight;
  float *data; /* loaded raw weights */
  uint64_t n;
} imp_value_t;

typedef struct imp_op {
  uint32_t id;
  uint32_t kind; /* pai_graph_op_kind_t */
  uint32_t num_in;
  uint32_t num_out;
  uint32_t in[PAI_GRAPH_MAX_ARITY];
  uint32_t out[PAI_GRAPH_MAX_ARITY];
} imp_op_t;

typedef struct imp_tok {
  uint32_t id;
  char text[IMP_MAX_TEXT];
} imp_tok_t;

typedef struct imp_model {
  char name[PAI_PAI_NAME_MAX + 1];
  uint32_t family;
  uint32_t context_len;
  uint32_t num_layers;
  uint32_t kv_bytes_per_token;
  uint32_t vocab_size;

  imp_value_t values[PAI_GRAPH_MAX_VALUES];
  uint32_t num_values;
  imp_op_t ops[PAI_GRAPH_MAX_OPS];
  uint32_t num_ops;
  imp_tok_t tokens[PAI_TOK_MAX_TOKENS];
  uint32_t num_tokens;
} imp_model_t;

/* ------------------------------------------------------------------ */
/* Line/token helpers                                                  */
/* ------------------------------------------------------------------ */

static char *
skip_ws(char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\r') {
    p++;
  }
  return p;
}

/* Read one whitespace-delimited token into out. Returns 0 at EOL. */
static int
read_tok(char **pp, char *out, size_t cap) {
  char *p = skip_ws(*pp);
  size_t i = 0;

  if (*p == '\0' || *p == '\n') {
    *pp = p;
    return 0;
  }
  while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') {
    if (i + 1 < cap) {
      out[i++] = *p;
    }
    p++;
  }
  out[i] = '\0';
  *pp = p;
  return 1;
}

static int
parse_u32(const char *s, uint32_t *out) {
  char *end = NULL;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0') {
    return -1;
  }
  *out = (uint32_t)v;
  return 0;
}

/* Split a comma-separated token into u32 ids. */
static int
parse_id_list(const char *s, uint32_t *out, uint32_t max, uint32_t *out_n) {
  uint32_t n = 0;
  const char *p = s;

  while (*p) {
    char tmp[16];
    size_t i = 0;
    while (*p && *p != ',') {
      if (i + 1 < sizeof(tmp)) {
        tmp[i++] = *p;
      }
      p++;
    }
    tmp[i] = '\0';
    if (n >= max || parse_u32(tmp, &out[n]) != 0) {
      return -1;
    }
    n++;
    if (*p == ',') {
      p++;
    }
  }
  *out_n = n;
  return 0;
}

/* Parse a "[d0,d1,...]" token into a shape. */
static int
parse_shape(const char *tok, uint64_t *shape, uint32_t *out_rank) {
  const char *p = tok;
  uint32_t rank = 0;

  if (*p != '[') {
    return -1;
  }
  p++;
  if (*p == ']') {
    return -1; /* empty shape */
  }
  while (*p && *p != ']') {
    char tmp[24];
    size_t i = 0;
    while (*p && *p != ',' && *p != ']') {
      if (i + 1 < sizeof(tmp)) {
        tmp[i++] = *p;
      }
      p++;
    }
    tmp[i] = '\0';
    if (rank >= PAI_TENSOR_MAX_RANK || tmp[0] == '\0') {
      return -1;
    }
    {
      char *end = NULL;
      unsigned long long v = strtoull(tmp, &end, 10);
      if (end == tmp || *end != '\0') {
        return -1;
      }
      shape[rank++] = (uint64_t)v;
    }
    if (*p == ',') {
      p++;
    }
  }
  if (*p != ']' || rank == 0) {
    return -1;
  }
  *out_rank = rank;
  return 0;
}

static const struct {
  const char *name;
  uint32_t kind;
} k_op_kinds[] = {
    {"add", PAI_OP_ADD},
    {"mul", PAI_OP_MUL},
    {"gemm", PAI_OP_GEMM},
    {"gemv", PAI_OP_GEMV},
    {"matmul", PAI_OP_MATMUL},
    {"relu", PAI_OP_RELU},
    {"softmax", PAI_OP_SOFTMAX},
    {"layernorm", PAI_OP_LAYERNORM},
    {"rmsnorm", PAI_OP_RMSNORM},
    {"rope", PAI_OP_ROPE},
    {"attention", PAI_OP_ATTENTION},
    {"reshape", PAI_OP_RESHAPE},
    {"concat", PAI_OP_CONCAT},
    {"convert", PAI_OP_CONVERT},
    {"copy", PAI_OP_COPY},
};

static int
op_kind_from_name(const char *name, uint32_t *out) {
  for (size_t i = 0; i < sizeof(k_op_kinds) / sizeof(k_op_kinds[0]); i++) {
    if (strcmp(name, k_op_kinds[i].name) == 0) {
      *out = k_op_kinds[i].kind;
      return 0;
    }
  }
  return -1;
}

static const char *
basename_of(const char *path) {
  const char *b = strrchr(path, '/');
  const char *b2 = strrchr(path, '\\');
  if (b2 != NULL && (b == NULL || b2 > b)) {
    b = b2;
  }
  return b != NULL ? b + 1 : path;
}

/* ------------------------------------------------------------------ */
/* Parser                                                              */
/* ------------------------------------------------------------------ */

static int
parse_value_line(char **pp, imp_model_t *m) {
  char tok[IMP_MAX_TEXT];
  imp_value_t *v;
  uint32_t expected = m->num_values + 1;
  uint32_t id;

  if (!read_tok(pp, tok, sizeof(tok)) || parse_u32(tok, &id) != 0) {
    return -1;
  }
  if (id != expected) {
    return -1; /* ids must be 1..N ascending */
  }
  v = &m->values[m->num_values];
  memset(v, 0, sizeof(*v));
  v->id = id;

  if (!read_tok(pp, tok, sizeof(tok))) {
    return -1; /* dtype */
  }
  if (strcmp(tok, "f32") != 0) {
    return -1; /* v0: f32 only */
  }
  v->dtype = PAI_DTYPE_F32;

  if (!read_tok(pp, tok, sizeof(tok)) || parse_shape(tok, v->shape, &v->rank) != 0) {
    return -1; /* shape */
  }

  if (!read_tok(pp, tok, sizeof(tok))) {
    return -1; /* kind */
  }
  if (strcmp(tok, "input") == 0) {
    v->kind = 0;
  } else if (strcmp(tok, "param") == 0) {
    v->kind = 1;
  } else if (strcmp(tok, "activation") == 0) {
    v->kind = 2;
  } else if (strcmp(tok, "output") == 0) {
    v->kind = 3;
  } else {
    return -1;
  }

  if (read_tok(pp, tok, sizeof(tok)) && strncmp(tok, "weights=", 8) == 0) {
    if (v->kind != 1) {
      return -1;
    }
    if (strlen(tok + 8) + 1 > sizeof(v->weight_file)) {
      return -1;
    }
    strcpy(v->weight_file, tok + 8);
    v->has_weight = 1;
  }

  m->num_values++;
  return 0;
}

/* Append comma-separated ids from tok to the current in/out list. The
 * list may span multiple tokens (e.g. "1, 2" tokenizes as "1," "2"), so
 * ids accumulate at the current offset. */
static int
op_append_tok(imp_op_t *op, int outputs, const char *tok) {
  uint32_t *dst = outputs ? op->out : op->in;
  uint32_t *count = outputs ? &op->num_out : &op->num_in;
  uint32_t max = PAI_GRAPH_MAX_ARITY - *count;
  uint32_t n;

  if (parse_id_list(tok, dst + *count, max, &n) != 0) {
    return -1;
  }
  *count += n;
  return 0;
}

static int
parse_op_line(char **pp, imp_model_t *m) {
  char tok[IMP_MAX_TEXT];
  imp_op_t *op = &m->ops[m->num_ops];
  uint32_t expected = m->num_ops + 1;
  int outputs = 0;

  memset(op, 0, sizeof(*op));
  op->id = 0;

  if (!read_tok(pp, tok, sizeof(tok)) || parse_u32(tok, &op->id) != 0 ||
      op->id != expected) {
    return -1; /* op ids must be 1..M ascending */
  }
  if (!read_tok(pp, tok, sizeof(tok)) ||
      op_kind_from_name(tok, &op->kind) != 0) {
    return -1;
  }

  while (read_tok(pp, tok, sizeof(tok))) {
    char *arrow = strstr(tok, "->");
    if (arrow != NULL) {
      *arrow = '\0';
      if (tok[0] != '\0') {
        if (op_append_tok(op, outputs, tok) != 0) {
          return -1;
        }
      }
      outputs = 1;
      if (arrow[2] != '\0') {
        if (op_append_tok(op, 1, arrow + 2) != 0) {
          return -1;
        }
      }
      continue;
    }
    if (op_append_tok(op, outputs, tok) != 0) {
      return -1;
    }
  }

  if (op->num_in == 0 || op->num_out == 0 || op->num_in > PAI_GRAPH_MAX_ARITY ||
      op->num_out > PAI_GRAPH_MAX_ARITY) {
    return -1;
  }
  m->num_ops++;
  return 0;
}

static int
parse_token_line(char **pp, imp_model_t *m) {
  char tok[IMP_MAX_TEXT];
  imp_tok_t *t;
  char *text;

  if (!read_tok(pp, tok, sizeof(tok))) {
    return -1;
  }
  t = &m->tokens[m->num_tokens];
  t->id = 0;
  if (parse_u32(tok, &t->id) != 0) {
    return -1;
  }
  text = skip_ws(*pp);
  {
    char *eol = text;
    size_t len = 0;
    while (*eol && *eol != '\r' && *eol != '\n') {
      eol++;
      len++;
    }
    if (len == 0 || len >= sizeof(t->text)) {
      return -1;
    }
    memcpy(t->text, text, len);
    t->text[len] = '\0';
    *pp = eol;
  }
  m->num_tokens++;
  return 0;
}

static int
parse_model(const char *buf, imp_model_t *m) {
  const char *line = buf;

  memset(m, 0, sizeof(*m));
  strcpy(m->name, "model");

  while (*line) {
    const char *eol = strchr(line, '\n');
    size_t len = eol != NULL ? (size_t)(eol - line) : strlen(line);
    char tmp[IMP_MAX_LINE];
    char *p;
    char key[IMP_MAX_TEXT];
    char val[IMP_MAX_TEXT];

    if (len >= sizeof(tmp)) {
      return -1;
    }
    memcpy(tmp, line, len);
    tmp[len] = '\0';
    p = skip_ws(tmp);
    if (*p == '\0' || *p == '#') {
      line = eol != NULL ? eol + 1 : line + len;
      continue;
    }
    if (!read_tok(&p, key, sizeof(key))) {
      line = eol != NULL ? eol + 1 : line + len;
      continue;
    }

    if (strcmp(key, "name") == 0) {
      if (!read_tok(&p, val, sizeof(val)) || strlen(val) > PAI_PAI_NAME_MAX) {
        return -1;
      }
      strcpy(m->name, val);
    } else if (strcmp(key, "family") == 0) {
      if (!read_tok(&p, val, sizeof(val))) {
        return -1;
      }
      m->family = strcmp(val, "llm") == 0 ? PAI_PAI_FAMILY_LLM : 0;
    } else if (strcmp(key, "context") == 0) {
      if (!read_tok(&p, val, sizeof(val)) || parse_u32(val, &m->context_len) != 0) {
        return -1;
      }
    } else if (strcmp(key, "layers") == 0) {
      if (!read_tok(&p, val, sizeof(val)) || parse_u32(val, &m->num_layers) != 0) {
        return -1;
      }
    } else if (strcmp(key, "kv_bytes") == 0) {
      if (!read_tok(&p, val, sizeof(val)) ||
          parse_u32(val, &m->kv_bytes_per_token) != 0) {
        return -1;
      }
    } else if (strcmp(key, "vocab") == 0) {
      if (!read_tok(&p, val, sizeof(val)) || parse_u32(val, &m->vocab_size) != 0) {
        return -1;
      }
    } else if (strcmp(key, "value") == 0) {
      if (m->num_values >= PAI_GRAPH_MAX_VALUES ||
          parse_value_line(&p, m) != 0) {
        return -1;
      }
    } else if (strcmp(key, "op") == 0) {
      if (m->num_ops >= PAI_GRAPH_MAX_OPS || parse_op_line(&p, m) != 0) {
        return -1;
      }
    } else if (strcmp(key, "token") == 0) {
      if (m->num_tokens >= PAI_TOK_MAX_TOKENS ||
          parse_token_line(&p, m) != 0) {
        return -1;
      }
    } else {
      return -1; /* unknown directive */
    }

    line = eol != NULL ? eol + 1 : line + len;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Weight loading + quantization                                       */
/* ------------------------------------------------------------------ */

static uint64_t
value_numel(const imp_value_t *v) {
  uint64_t n = 1;
  for (uint32_t i = 0; i < v->rank; i++) {
    n *= v->shape[i];
  }
  return n;
}

static pai_status_t
load_weights(const char *dir, imp_model_t *m, const pai_import_options_t *opts) {
  pai_status_t st = PAI_OK;
  char path[512];

  for (uint32_t i = 0; i < m->num_values; i++) {
    imp_value_t *v = &m->values[i];
    FILE *f;
    long nbytes;

    if (!v->has_weight) {
      continue;
    }
    v->n = value_numel(v);
    if (v->n == 0) {
      return PAI_ERR_INVALID_ARG;
    }
    if (dir != NULL && dir[0] != '\0') {
      snprintf(path, sizeof(path), "%s/%s", dir, v->weight_file);
    } else {
      snprintf(path, sizeof(path), "%s", v->weight_file);
    }
    f = fopen(path, "rb");
    if (f == NULL) {
      return PAI_ERR_IO;
    }
    fseek(f, 0, SEEK_END);
    nbytes = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (nbytes < 0 || (uint64_t)nbytes != v->n * sizeof(float)) {
      fclose(f);
      return PAI_ERR_MISMATCH; /* file size != element count * 4 */
    }
    v->data = (float *)malloc((size_t)v->n * sizeof(float));
    if (v->data == NULL) {
      fclose(f);
      return PAI_ERR_NOMEM;
    }
    if (fread(v->data, sizeof(float), (size_t)v->n, f) != (size_t)v->n) {
      fclose(f);
      return PAI_ERR_IO;
    }
    fclose(f);

    if (opts != NULL && opts->quant != NULL &&
        strcmp(opts->quant, "none") != 0) {
      pai_quant_scheme_t scheme;
      if (strcmp(opts->quant, "q8") == 0) {
        scheme.bit_width = 8;
      } else if (strcmp(opts->quant, "q4") == 0) {
        scheme.bit_width = 4;
      } else {
        return PAI_ERR_INVALID_ARG;
      }
      scheme.is_signed = 1;
      scheme.group_size = opts->quant_group;
      st = pai_quant_scheme_check(&scheme);
      if (st != PAI_OK) {
        return st;
      }
    }
  }
  return PAI_OK;
}

static void
free_weights(imp_model_t *m) {
  for (uint32_t i = 0; i < m->num_values; i++) {
    free(m->values[i].data);
    m->values[i].data = NULL;
  }
}

/* ------------------------------------------------------------------ */
/* Import                                                              */
/* ------------------------------------------------------------------ */

static pai_status_t
build_container(const imp_model_t *m, const pai_import_options_t *opts,
                uint8_t **out_blob, uint32_t *out_nbytes) {
  pai_pai_meta_t meta;
  pai_container_tensor_t *tensors = NULL;
  pai_container_op_t *ops = NULL;
  pai_container_token_t *tokens = NULL;
  pai_container_quant_t quant;
  pai_status_t st = PAI_OK;
  uint32_t i;

  tensors = (pai_container_tensor_t *)calloc(m->num_values, sizeof(*tensors));
  ops = (pai_container_op_t *)calloc(m->num_ops, sizeof(*ops));
  tokens = (pai_container_token_t *)calloc(m->num_tokens > 0 ? m->num_tokens : 1,
                                           sizeof(*tokens));
  if (tensors == NULL || ops == NULL || tokens == NULL) {
    st = PAI_ERR_NOMEM;
    goto done;
  }

  for (i = 0; i < m->num_values; i++) {
    const imp_value_t *v = &m->values[i];
    pai_container_tensor_t *t = &tensors[i];
    t->id = v->id;
    t->dtype = v->dtype;
    t->rank = v->rank;
    memcpy(t->shape, v->shape, sizeof(t->shape));
    t->kind = v->kind;
    if (v->has_weight) {
      t->name = basename_of(v->weight_file);
      t->data = v->data;
      t->n = v->n;
    }
  }
  for (i = 0; i < m->num_ops; i++) {
    const imp_op_t *o = &m->ops[i];
    pai_container_op_t *op = &ops[i];
    op->id = o->id;
    op->kind = o->kind;
    op->num_in = o->num_in;
    op->num_out = o->num_out;
    op->in = o->in;
    op->out = o->out;
  }
  for (i = 0; i < m->num_tokens; i++) {
    tokens[i].id = m->tokens[i].id;
    tokens[i].text = m->tokens[i].text;
  }

  memset(&meta, 0, sizeof(meta));
  strncpy(meta.name, m->name, sizeof(meta.name) - 1);
  meta.family = m->family;
  meta.context_len = m->context_len;
  meta.num_layers = m->num_layers;
  meta.kv_bytes_per_token = m->kv_bytes_per_token;
  meta.vocab_size = m->vocab_size;

  if (opts != NULL && opts->quant != NULL) {
    quant.quant = opts->quant;
    quant.quant_group = opts->quant_group;
  } else {
    quant.quant = "none";
    quant.quant_group = 0;
  }

  st = pai_container_build(&meta, tensors, m->num_values, ops, m->num_ops,
                           tokens, m->num_tokens, NULL, 0, &quant,
                           out_blob, out_nbytes);

done:
  free(tensors);
  free(ops);
  free(tokens);
  return st;
}

pai_status_t
pai_import_model(const char *desc_path, const pai_import_options_t *opts,
                 uint8_t **out_blob, uint32_t *out_nbytes) {
  FILE *f;
  long nbytes;
  char *buf;
  imp_model_t *m;
  pai_status_t st;
  char dir[512];
  const char *slash;

  if (desc_path == NULL || out_blob == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_blob = NULL;
  *out_nbytes = 0;

  /* The model struct scales with the raised graph/tokenizer caps
   * (1024 values/ops, 65536 tokens) — keep it off the stack. */
  m = (imp_model_t *)calloc(1, sizeof(*m));
  if (m == NULL) {
    return PAI_ERR_NOMEM;
  }

  f = fopen(desc_path, "rb");
  if (f == NULL) {
    free(m);
    return PAI_ERR_IO;
  }
  fseek(f, 0, SEEK_END);
  nbytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (nbytes < 0 || (uint64_t)nbytes > (uint64_t)(1u << 20)) {
    fclose(f);
    free(m);
    return PAI_ERR_INVALID_ARG;
  }
  buf = (char *)malloc((size_t)nbytes + 1);
  if (buf == NULL) {
    fclose(f);
    free(m);
    return PAI_ERR_NOMEM;
  }
  if (fread(buf, 1, (size_t)nbytes, f) != (size_t)nbytes) {
    free(buf);
    fclose(f);
    free(m);
    return PAI_ERR_IO;
  }
  buf[nbytes] = '\0';
  fclose(f);

  if (parse_model(buf, m) != 0) {
    free(buf);
    free(m);
    return PAI_ERR_INVALID_ARG;
  }
  free(buf);

  if (m->num_values == 0 || m->num_ops == 0) {
    free(m);
    return PAI_ERR_INVALID_ARG;
  }

  /* Weight file directory = description file directory. */
  dir[0] = '\0';
  slash = strrchr(desc_path, '/');
  {
    const char *bs = strrchr(desc_path, '\\');
    if (bs != NULL && (slash == NULL || bs > slash)) {
      slash = bs;
    }
  }
  if (slash != NULL) {
    size_t len = (size_t)(slash - desc_path);
    if (len >= sizeof(dir)) {
      len = sizeof(dir) - 1;
    }
    memcpy(dir, desc_path, len);
    dir[len] = '\0';
  }

  /* Validate the quantization option up front so a bad value is
   * rejected even when the model has no params. */
  if (opts != NULL && opts->quant != NULL &&
      strcmp(opts->quant, "none") != 0 && strcmp(opts->quant, "q8") != 0 &&
      strcmp(opts->quant, "q4") != 0) {
    free(m);
    return PAI_ERR_INVALID_ARG;
  }

  st = load_weights(dir, m, opts);
  if (st != PAI_OK) {
    free_weights(m);
    free(m);
    return st;
  }

  st = build_container(m, opts, out_blob, out_nbytes);
  free_weights(m);
  free(m);
  return st;
}

pai_status_t
pai_import_model_to_file(const char *desc_path, const char *out_path,
                         const pai_import_options_t *opts) {
  uint8_t *blob = NULL;
  uint32_t nbytes = 0;
  pai_status_t st;

  st = pai_import_model(desc_path, opts, &blob, &nbytes);
  if (st != PAI_OK) {
    return st;
  }
  st = pai_pai_write_file(out_path, blob, nbytes);
  free(blob);
  return st;
}
