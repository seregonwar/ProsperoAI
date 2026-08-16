/*
 * ProsperoAI — minimal JSON (gateway §26) — implementation
 */

#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct pai_json_node {
  uint8_t  type;     /* pai_json_type_t                               */
  uint8_t  pad[3];
  int32_t  key;      /* object member key (arena index); -1 = none    */
  int32_t  str;      /* string value (arena index); -1 = none         */
  double   num;
  int32_t  boolean;
  int32_t  first_child; /* first child node; -1 = none                */
  int32_t  next;     /* next sibling; -1 = none                       */
};

/* ------------------------------------------------------------------ */
/* Node / string pool helpers                                          */
/* ------------------------------------------------------------------ */

static int32_t
new_node(pai_json_doc_t *doc, pai_json_type_t type) {
  struct pai_json_node *n;

  if (doc->num_nodes >= PAI_JSON_MAX_NODES) {
    return -1;
  }
  if (doc->num_nodes == doc->cap_nodes) {
    uint32_t ncap = doc->cap_nodes == 0 ? 64 : doc->cap_nodes * 2;
    struct pai_json_node *nn =
        (struct pai_json_node *)realloc(doc->nodes,
                                        (size_t)ncap * sizeof(*nn));
    if (nn == NULL) {
      return -1;
    }
    doc->nodes = nn;
    doc->cap_nodes = ncap;
  }
  n = &doc->nodes[doc->num_nodes];
  memset(n, 0, sizeof(*n));
  n->type = (uint8_t)type;
  n->key = -1;
  n->str = -1;
  n->first_child = -1;
  n->next = -1;
  return (int32_t)doc->num_nodes++;
}

/* Append a NUL-terminated copy of s[0..len) to the arena. */
static int32_t
new_str(pai_json_doc_t *doc, const char *s, uint32_t len) {
  uint32_t need = len + 1;

  if (len > PAI_JSON_MAX_STRING) {
    return -1;
  }
  if (doc->str_used + need > doc->str_cap) {
    uint32_t ncap = doc->str_cap == 0 ? 1024 : doc->str_cap;
    char *ns;
    while (ncap < doc->str_used + need) {
      ncap *= 2;
    }
    ns = (char *)realloc(doc->strs, ncap);
    if (ns == NULL) {
      return -1;
    }
    doc->strs = ns;
    doc->str_cap = ncap;
  }
  memcpy(doc->strs + doc->str_used, s, len);
  doc->strs[doc->str_used + len] = '\0';
  {
    int32_t idx = (int32_t)doc->str_used;
    doc->str_used += need;
    return idx;
  }
}

/* ------------------------------------------------------------------ */
/* Parser                                                              */
/* ------------------------------------------------------------------ */

static pai_status_t
parse_value(pai_json_doc_t *doc, const char *s, uint32_t n, uint32_t *pos,
            int depth, int32_t *out);

static int
is_ws(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static void
skip_ws(const char *s, uint32_t n, uint32_t *pos) {
  while (*pos < n && is_ws(s[*pos])) {
    (*pos)++;
  }
}

/* Decode a hex digit. */
static int
hex_val(char c) {
  if (c >= '0' && c <= '9') {
    return c - '0';
  }
  if (c >= 'a' && c <= 'f') {
    return c - 'a' + 10;
  }
  if (c >= 'A' && c <= 'F') {
    return c - 'A' + 10;
  }
  return -1;
}

/* Encode code point u as UTF-8 into buf; returns bytes written. */
static uint32_t
utf8_encode(uint32_t u, char *buf) {
  if (u < 0x80) {
    buf[0] = (char)u;
    return 1;
  }
  if (u < 0x800) {
    buf[0] = (char)(0xC0 | (u >> 6));
    buf[1] = (char)(0x80 | (u & 0x3F));
    return 2;
  }
  if (u < 0x10000) {
    buf[0] = (char)(0xE0 | (u >> 12));
    buf[1] = (char)(0x80 | ((u >> 6) & 0x3F));
    buf[2] = (char)(0x80 | (u & 0x3F));
    return 3;
  }
  buf[0] = (char)(0xF0 | (u >> 18));
  buf[1] = (char)(0x80 | ((u >> 12) & 0x3F));
  buf[2] = (char)(0x80 | ((u >> 6) & 0x3F));
  buf[3] = (char)(0x80 | (u & 0x3F));
  return 4;
}

/* Parse a double from s[pos..n) following the JSON grammar; advances
 * pos past the token. PAI_ERR_PROTOCOL on malformed numbers. */
static pai_status_t
parse_number(const char *s, uint32_t n, uint32_t *pos, double *out) {
  uint32_t i = *pos;
  double v = 0.0;
  int neg = 0;
  double frac = 0.1;
  int exp_neg = 0;
  double exp = 0.0;
  int any = 0;

  if (i < n && s[i] == '-') {
    neg = 1;
    i++;
  }
  if (i < n && s[i] == '0') {
    i++;
    any = 1;
  } else if (i < n && s[i] >= '1' && s[i] <= '9') {
    while (i < n && s[i] >= '0' && s[i] <= '9') {
      v = v * 10.0 + (double)(s[i] - '0');
      i++;
      any = 1;
    }
  }
  if (!any) {
    return PAI_ERR_PROTOCOL;
  }
  /* Fraction. */
  if (i < n && s[i] == '.') {
    i++;
    if (i >= n || s[i] < '0' || s[i] > '9') {
      return PAI_ERR_PROTOCOL;
    }
    while (i < n && s[i] >= '0' && s[i] <= '9') {
      v += (double)(s[i] - '0') * frac;
      frac *= 0.1;
      i++;
    }
  }
  /* Exponent. */
  if (i < n && (s[i] == 'e' || s[i] == 'E')) {
    i++;
    if (i < n && (s[i] == '+' || s[i] == '-')) {
      exp_neg = s[i] == '-';
      i++;
    }
    if (i >= n || s[i] < '0' || s[i] > '9') {
      return PAI_ERR_PROTOCOL;
    }
    while (i < n && s[i] >= '0' && s[i] <= '9') {
      exp = exp * 10.0 + (double)(s[i] - '0');
      i++;
      if (exp > 300.0) {
        return PAI_ERR_PROTOCOL; /* out of double range */
      }
    }
  }
  if (exp != 0.0) {
    v *= pow(10.0, exp_neg ? -exp : exp);
  }
  *pos = i;
  *out = neg ? -v : v;
  return PAI_OK;
}

/* Parse a JSON string literal (the opening quote already consumed).
 * Returns the decoded text as an arena string. */
static pai_status_t
parse_string(pai_json_doc_t *doc, const char *s, uint32_t n, uint32_t *pos,
             int32_t *out_idx) {
  uint32_t cap = 0;
  uint32_t used = 0;
  char *tmp = NULL;

  for (;;) {
    char c;
    if (*pos >= n) {
      free(tmp);
      return PAI_ERR_PROTOCOL;
    }
    c = s[*pos];
    if (c == '"') {
      (*pos)++;
      break;
    }
    if (c == '\\') {
      (*pos)++;
      if (*pos >= n) {
        free(tmp);
        return PAI_ERR_PROTOCOL;
      }
      c = s[*pos];
      (*pos)++;
      switch (c) {
      case '"':
      case '\\':
      case '/':
        break;
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case 'u': {
        /* \uXXXX with optional low-surrogate continuation. */
        uint32_t cp = 0;
        int h, k;
        char enc[4];
        uint32_t elen;
        for (k = 0; k < 4; k++) {
          if (*pos >= n || (h = hex_val(s[*pos])) < 0) {
            free(tmp);
            return PAI_ERR_PROTOCOL;
          }
          cp = (cp << 4) | (uint32_t)h;
          (*pos)++;
        }
        if (cp >= 0xD800 && cp <= 0xDBFF) {
          /* Expect a low surrogate immediately after. */
          if (*pos + 6 <= n && s[*pos] == '\\' && s[*pos + 1] == 'u') {
            uint32_t lo = 0;
            int ok = 1;
            for (k = 0; k < 4; k++) {
              h = hex_val(s[*pos + 2 + k]);
              if (h < 0) {
                ok = 0;
                break;
              }
              lo = (lo << 4) | (uint32_t)h;
            }
            if (ok && lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              *pos += 6;
            } else {
              cp = 0xFFFD; /* lone high surrogate */
            }
          } else {
            cp = 0xFFFD;
          }
        } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
          cp = 0xFFFD; /* lone low surrogate */
        }
        elen = utf8_encode(cp, enc);
        if (used + elen > cap) {
          uint32_t ncap = cap == 0 ? 64 : cap * 2;
          char *nt = (char *)realloc(tmp, ncap);
          if (nt == NULL) {
            free(tmp);
            return PAI_ERR_NOMEM;
          }
          tmp = nt;
          cap = ncap;
        }
        memcpy(tmp + used, enc, elen);
        used += elen;
        continue;
      }
      default:
        free(tmp);
        return PAI_ERR_PROTOCOL; /* invalid escape */
      }
      /* Simple escape: write the mapped character. The position was
       * already advanced past both escape characters above, so the
       * plain-character path below must not consume anything else. */
      if (used + 1 > cap) {
        uint32_t ncap = cap == 0 ? 64 : cap * 2;
        char *nt = (char *)realloc(tmp, ncap);
        if (nt == NULL) {
          free(tmp);
          return PAI_ERR_NOMEM;
        }
        tmp = nt;
        cap = ncap;
      }
      tmp[used++] = c;
      continue;
    }
    if ((uint8_t)c < 0x20) {
      free(tmp);
      return PAI_ERR_PROTOCOL; /* raw control character */
    }

    if (used + 1 > cap) {
      uint32_t ncap = cap == 0 ? 64 : cap * 2;
      char *nt = (char *)realloc(tmp, ncap);
      if (nt == NULL) {
        free(tmp);
        return PAI_ERR_NOMEM;
      }
      tmp = nt;
      cap = ncap;
    }
    tmp[used++] = c;
    (*pos)++; /* consume the plain character just copied */
  }

  if (used > PAI_JSON_MAX_STRING) {
    free(tmp);
    return PAI_ERR_NOMEM;
  }
  *out_idx = new_str(doc, tmp != NULL ? tmp : "", used);
  free(tmp);
  return *out_idx >= 0 ? PAI_OK : PAI_ERR_NOMEM;
}

/* Append `child` to `parent`'s child list. */
static void
append_child(pai_json_doc_t *doc, int32_t parent, int32_t child) {
  struct pai_json_node *p = &doc->nodes[parent];
  if (p->first_child < 0) {
    p->first_child = child;
    return;
  }
  {
    int32_t last = p->first_child;
    while (doc->nodes[last].next >= 0) {
      last = doc->nodes[last].next;
    }
    doc->nodes[last].next = child;
  }
}

static pai_status_t
parse_value(pai_json_doc_t *doc, const char *s, uint32_t n, uint32_t *pos,
            int depth, int32_t *out) {
  int32_t node;
  pai_status_t st;

  if (depth > PAI_JSON_MAX_DEPTH) {
    return PAI_ERR_NOMEM;
  }
  skip_ws(s, n, pos);
  if (*pos >= n) {
    return PAI_ERR_PROTOCOL;
  }

  switch (s[*pos]) {
  case '{': {
    node = new_node(doc, PAI_JSON_OBJECT);
    if (node < 0) {
      return PAI_ERR_NOMEM;
    }
    (*pos)++;
    skip_ws(s, n, pos);
    if (*pos < n && s[*pos] == '}') {
      (*pos)++;
      *out = node;
      return PAI_OK;
    }
    for (;;) {
      int32_t key_idx;
      int32_t val_idx;
      if (*pos >= n || s[*pos] != '"') {
        return PAI_ERR_PROTOCOL; /* member keys must be strings */
      }
      (*pos)++;
      st = parse_string(doc, s, n, pos, &key_idx);
      if (st != PAI_OK) {
        return st;
      }
      skip_ws(s, n, pos);
      if (*pos >= n || s[*pos] != ':') {
        return PAI_ERR_PROTOCOL;
      }
      (*pos)++;
      st = parse_value(doc, s, n, pos, depth + 1, &val_idx);
      if (st != PAI_OK) {
        return st;
      }
      doc->nodes[val_idx].key = key_idx;
      append_child(doc, node, val_idx);
      skip_ws(s, n, pos);
      if (*pos < n && s[*pos] == ',') {
        (*pos)++;
        continue;
      }
      if (*pos < n && s[*pos] == '}') {
        (*pos)++;
        *out = node;
        return PAI_OK;
      }
      return PAI_ERR_PROTOCOL;
    }
  }
  case '[': {
    node = new_node(doc, PAI_JSON_ARRAY);
    if (node < 0) {
      return PAI_ERR_NOMEM;
    }
    (*pos)++;
    skip_ws(s, n, pos);
    if (*pos < n && s[*pos] == ']') {
      (*pos)++;
      *out = node;
      return PAI_OK;
    }
    for (;;) {
      int32_t val_idx;
      st = parse_value(doc, s, n, pos, depth + 1, &val_idx);
      if (st != PAI_OK) {
        return st;
      }
      append_child(doc, node, val_idx);
      skip_ws(s, n, pos);
      if (*pos < n && s[*pos] == ',') {
        (*pos)++;
        continue;
      }
      if (*pos < n && s[*pos] == ']') {
        (*pos)++;
        *out = node;
        return PAI_OK;
      }
      return PAI_ERR_PROTOCOL;
    }
  }
  case '"': {
    node = new_node(doc, PAI_JSON_STRING);
    if (node < 0) {
      return PAI_ERR_NOMEM;
    }
    (*pos)++;
    st = parse_string(doc, s, n, pos, &doc->nodes[node].str);
    if (st == PAI_OK) {
      *out = node;
    }
    return st;
  }
  case 't':
    if (*pos + 4 <= n && memcmp(s + *pos, "true", 4) == 0) {
      node = new_node(doc, PAI_JSON_BOOL);
      if (node < 0) {
        return PAI_ERR_NOMEM;
      }
      doc->nodes[node].boolean = 1;
      *pos += 4;
      *out = node;
      return PAI_OK;
    }
    return PAI_ERR_PROTOCOL;
  case 'f':
    if (*pos + 5 <= n && memcmp(s + *pos, "false", 5) == 0) {
      node = new_node(doc, PAI_JSON_BOOL);
      if (node < 0) {
        return PAI_ERR_NOMEM;
      }
      doc->nodes[node].boolean = 0;
      *pos += 5;
      *out = node;
      return PAI_OK;
    }
    return PAI_ERR_PROTOCOL;
  case 'n':
    if (*pos + 4 <= n && memcmp(s + *pos, "null", 4) == 0) {
      node = new_node(doc, PAI_JSON_NULL);
      if (node < 0) {
        return PAI_ERR_NOMEM;
      }
      *pos += 4;
      *out = node;
      return PAI_OK;
    }
    return PAI_ERR_PROTOCOL;
  default: {
    double v;
    st = parse_number(s, n, pos, &v);
    if (st != PAI_OK) {
      return st;
    }
    node = new_node(doc, PAI_JSON_NUMBER);
    if (node < 0) {
      return PAI_ERR_NOMEM;
    }
    doc->nodes[node].num = v;
    *out = node;
    return PAI_OK;
  }
  }
}

pai_status_t
pai_json_parse(pai_json_doc_t *doc, const char *text, uint32_t nbytes) {
  uint32_t pos = 0;
  int32_t root;
  pai_status_t st;

  if (doc == NULL || (text == NULL && nbytes != 0)) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(doc, 0, sizeof(*doc));
  doc->root = -1;

  st = parse_value(doc, text, nbytes, &pos, 0, &root);
  if (st != PAI_OK) {
    return st;
  }
  skip_ws(text, nbytes, &pos);
  if (pos != nbytes) {
    return PAI_ERR_PROTOCOL; /* trailing garbage */
  }
  doc->root = root;
  return PAI_OK;
}

void
pai_json_destroy(pai_json_doc_t *doc) {
  if (doc == NULL) {
    return;
  }
  free(doc->nodes);
  free(doc->strs);
  memset(doc, 0, sizeof(*doc));
  doc->root = -1;
}

int32_t
pai_json_root(const pai_json_doc_t *doc) {
  return doc != NULL ? doc->root : -1;
}

pai_json_type_t
pai_json_type(const pai_json_doc_t *doc, int32_t node) {
  if (doc == NULL || node < 0 || (uint32_t)node >= doc->num_nodes) {
    return PAI_JSON_NULL;
  }
  return (pai_json_type_t)doc->nodes[node].type;
}

const char *
pai_json_str(const pai_json_doc_t *doc, int32_t node) {
  if (doc == NULL || node < 0 || (uint32_t)node >= doc->num_nodes ||
      doc->nodes[node].type != PAI_JSON_STRING ||
      doc->nodes[node].str < 0) {
    return NULL;
  }
  return doc->strs + doc->nodes[node].str;
}

double
pai_json_num(const pai_json_doc_t *doc, int32_t node) {
  if (doc == NULL || node < 0 || (uint32_t)node >= doc->num_nodes ||
      doc->nodes[node].type != PAI_JSON_NUMBER) {
    return 0.0;
  }
  return doc->nodes[node].num;
}

int
pai_json_bool(const pai_json_doc_t *doc, int32_t node) {
  if (doc == NULL || node < 0 || (uint32_t)node >= doc->num_nodes ||
      doc->nodes[node].type != PAI_JSON_BOOL) {
    return 0;
  }
  return doc->nodes[node].boolean;
}

int32_t
pai_json_member(const pai_json_doc_t *doc, int32_t obj, const char *key) {
  int32_t child;

  if (doc == NULL || obj < 0 || (uint32_t)obj >= doc->num_nodes ||
      doc->nodes[obj].type != PAI_JSON_OBJECT || key == NULL) {
    return -1;
  }
  child = doc->nodes[obj].first_child;
  while (child >= 0) {
    if (doc->nodes[child].key >= 0 &&
        strcmp(doc->strs + doc->nodes[child].key, key) == 0) {
      return child;
    }
    child = doc->nodes[child].next;
  }
  return -1;
}

int32_t
pai_json_array_len(const pai_json_doc_t *doc, int32_t arr) {
  int32_t child;
  int32_t n = 0;

  if (doc == NULL || arr < 0 || (uint32_t)arr >= doc->num_nodes ||
      doc->nodes[arr].type != PAI_JSON_ARRAY) {
    return -1;
  }
  child = doc->nodes[arr].first_child;
  while (child >= 0) {
    n++;
    child = doc->nodes[child].next;
  }
  return n;
}

int32_t
pai_json_array_at(const pai_json_doc_t *doc, int32_t arr, uint32_t i) {
  int32_t child;
  uint32_t n = 0;

  if (doc == NULL || arr < 0 || (uint32_t)arr >= doc->num_nodes ||
      doc->nodes[arr].type != PAI_JSON_ARRAY) {
    return -1;
  }
  child = doc->nodes[arr].first_child;
  while (child >= 0) {
    if (n == i) {
      return child;
    }
    n++;
    child = doc->nodes[child].next;
  }
  return -1;
}

const char *
pai_json_str_member(const pai_json_doc_t *doc, int32_t obj, const char *key) {
  int32_t m = pai_json_member(doc, obj, key);
  if (m < 0) {
    return NULL;
  }
  return pai_json_str(doc, m);
}

/* ------------------------------------------------------------------ */
/* Writer                                                              */
/* ------------------------------------------------------------------ */

void
pai_json_wb_init(pai_json_wb_t *wb) {
  wb->buf = NULL;
  wb->len = 0;
  wb->cap = 0;
}

void
pai_json_wb_destroy(pai_json_wb_t *wb) {
  free(wb->buf);
  wb->buf = NULL;
  wb->len = 0;
  wb->cap = 0;
}

pai_status_t
pai_json_wb_reserve(pai_json_wb_t *wb, uint32_t extra) {
  uint32_t need = wb->len + extra;

  if (need <= wb->cap) {
    return PAI_OK;
  }
  {
    uint32_t ncap = wb->cap == 0 ? 256 : wb->cap;
    char *nb;
    while (ncap < need) {
      ncap *= 2;
    }
    nb = (char *)realloc(wb->buf, ncap);
    if (nb == NULL) {
      return PAI_ERR_NOMEM;
    }
    wb->buf = nb;
    wb->cap = ncap;
  }
  return PAI_OK;
}

pai_status_t
pai_json_wb_putc(pai_json_wb_t *wb, char c) {
  pai_status_t st = pai_json_wb_reserve(wb, 1);
  if (st != PAI_OK) {
    return st;
  }
  wb->buf[wb->len++] = c;
  return PAI_OK;
}

pai_status_t
pai_json_wb_putn(pai_json_wb_t *wb, const char *s, uint32_t n) {
  pai_status_t st;

  if (s == NULL && n != 0) {
    return PAI_ERR_INVALID_ARG;
  }
  st = pai_json_wb_reserve(wb, n);
  if (st != PAI_OK) {
    return st;
  }
  if (n != 0) {
    memcpy(wb->buf + wb->len, s, n);
    wb->len += n;
  }
  return PAI_OK;
}

pai_status_t
pai_json_wb_puts(pai_json_wb_t *wb, const char *s) {
  return pai_json_wb_putn(wb, s, (uint32_t)strlen(s));
}

pai_status_t
pai_json_wb_putf(pai_json_wb_t *wb, const char *fmt, ...) {
  char stack[512];
  char *heap = NULL;
  char *buf = stack;
  va_list ap;
  int n;
  pai_status_t st;

  va_start(ap, fmt);
  n = vsnprintf(stack, sizeof(stack), fmt, ap);
  va_end(ap);
  if (n < 0) {
    return PAI_ERR_INVALID_ARG;
  }
  if ((uint32_t)n >= sizeof(stack)) {
    heap = (char *)malloc((size_t)n + 1);
    if (heap == NULL) {
      return PAI_ERR_NOMEM;
    }
    buf = heap;
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
  }
  st = pai_json_wb_putn(wb, buf, (uint32_t)n);
  free(heap);
  return st;
}

pai_status_t
pai_json_quote(pai_json_wb_t *wb, const char *s, uint32_t n) {
  uint32_t i;
  pai_status_t st;

  st = pai_json_wb_putc(wb, '"');
  if (st != PAI_OK) {
    return st;
  }
  for (i = 0; i < n; i++) {
    unsigned char c = (unsigned char)s[i];
    switch (c) {
    case '"':
      st = pai_json_wb_puts(wb, "\\\"");
      break;
    case '\\':
      st = pai_json_wb_puts(wb, "\\\\");
      break;
    case '\b':
      st = pai_json_wb_puts(wb, "\\b");
      break;
    case '\f':
      st = pai_json_wb_puts(wb, "\\f");
      break;
    case '\n':
      st = pai_json_wb_puts(wb, "\\n");
      break;
    case '\r':
      st = pai_json_wb_puts(wb, "\\r");
      break;
    case '\t':
      st = pai_json_wb_puts(wb, "\\t");
      break;
    default:
      if (c < 0x20) {
        char esc[8];
        snprintf(esc, sizeof(esc), "\\u%04x", c);
        st = pai_json_wb_puts(wb, esc);
      } else {
        st = pai_json_wb_putc(wb, (char)c);
      }
      break;
    }
    if (st != PAI_OK) {
      return st;
    }
  }
  return pai_json_wb_putc(wb, '"');
}

pai_status_t
pai_json_put_number(pai_json_wb_t *wb, double v) {
  return pai_json_wb_putf(wb, "%.7g", v);
}

pai_status_t
pai_json_serialize(const pai_json_doc_t *doc, int32_t node,
                   pai_json_wb_t *wb) {
  pai_json_type_t t;
  pai_status_t st;

  if (doc == NULL || node < 0 || (uint32_t)node >= doc->num_nodes) {
    return PAI_ERR_INVALID_ARG;
  }
  t = pai_json_type(doc, node);
  switch (t) {
  case PAI_JSON_NULL:
    return pai_json_wb_puts(wb, "null");
  case PAI_JSON_BOOL:
    return pai_json_wb_puts(wb, pai_json_bool(doc, node) ? "true" : "false");
  case PAI_JSON_NUMBER:
    return pai_json_put_number(wb, pai_json_num(doc, node));
  case PAI_JSON_STRING: {
    const char *s = pai_json_str(doc, node);
    return pai_json_quote(wb, s != NULL ? s : "", s != NULL ? (uint32_t)strlen(s) : 0);
  }
  case PAI_JSON_ARRAY: {
    int32_t child = doc->nodes[node].first_child;
    int32_t n = 0;
    st = pai_json_wb_putc(wb, '[');
    if (st != PAI_OK) {
      return st;
    }
    while (child >= 0) {
      if (n++ != 0) {
        st = pai_json_wb_putc(wb, ',');
        if (st != PAI_OK) {
          return st;
        }
      }
      st = pai_json_serialize(doc, child, wb);
      if (st != PAI_OK) {
        return st;
      }
      child = doc->nodes[child].next;
    }
    return pai_json_wb_putc(wb, ']');
  }
  case PAI_JSON_OBJECT: {
    int32_t child = doc->nodes[node].first_child;
    int32_t n = 0;
    st = pai_json_wb_putc(wb, '{');
    if (st != PAI_OK) {
      return st;
    }
    while (child >= 0) {
      if (n++ != 0) {
        st = pai_json_wb_putc(wb, ',');
        if (st != PAI_OK) {
          return st;
        }
      }
      st = pai_json_quote(wb, doc->strs + doc->nodes[child].key,
                          (uint32_t)strlen(doc->strs + doc->nodes[child].key));
      if (st != PAI_OK) {
        return st;
      }
      st = pai_json_wb_putc(wb, ':');
      if (st != PAI_OK) {
        return st;
      }
      st = pai_json_serialize(doc, child, wb);
      if (st != PAI_OK) {
        return st;
      }
      child = doc->nodes[child].next;
    }
    return pai_json_wb_putc(wb, '}');
  }
  }
  return PAI_ERR_INVALID_ARG;
}
