#include "tokenizer.h"

#include <stdlib.h>
#include <string.h>

void
pai_tok_init(pai_tok_t *tok) {
  memset(tok, 0, sizeof(*tok));
  tok->byte_fallback_base = 256u;
}

void
pai_tok_destroy(pai_tok_t *tok) {
  if (tok == NULL) {
    return;
  }
  free(tok->tokens);
  free(tok->token_ids);
  free(tok->merges);
  memset(tok, 0, sizeof(*tok));
}

pai_status_t
pai_tok_add_token(pai_tok_t *tok, uint32_t id, const char *text) {
  uint32_t len;
  uint32_t i;

  if (tok == NULL || text == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  len = (uint32_t)strlen(text);
  if (len == 0 || len > PAI_TOK_MAX_TOKEN_LEN) {
    return PAI_ERR_INVALID_ARG;
  }
  /* A duplicate id would make encode/decode ambiguous (decode picks the
   * first match); reject it so the mapping stays a bijection. */
  for (i = 0; i < tok->num_tokens; i++) {
    if (tok->token_ids[i] == id) {
      return PAI_ERR_INVALID_ARG;
    }
  }
  if (tok->num_tokens >= PAI_TOK_MAX_TOKENS) {
    return PAI_ERR_NOMEM;
  }

  if (tok->tokens == NULL) {
    tok->tokens = (char(*)[PAI_TOK_MAX_TOKEN_LEN + 1])malloc(
        (size_t)PAI_TOK_MAX_TOKENS * (PAI_TOK_MAX_TOKEN_LEN + 1));
    tok->token_ids = (uint32_t *)malloc((size_t)PAI_TOK_MAX_TOKENS *
                                        sizeof(uint32_t));
    if (tok->tokens == NULL || tok->token_ids == NULL) {
      free(tok->tokens);
      free(tok->token_ids);
      tok->tokens = NULL;
      tok->token_ids = NULL;
      return PAI_ERR_NOMEM;
    }
  }

  memcpy(tok->tokens[tok->num_tokens], text, len);
  tok->tokens[tok->num_tokens][len] = '\0';
  tok->token_ids[tok->num_tokens] = id;
  if (id > tok->max_id) {
    tok->max_id = id;
  }
  tok->num_tokens++;
  return PAI_OK;
}

pai_status_t
pai_tok_add_merge(pai_tok_t *tok, uint32_t left, uint32_t right,
                  uint32_t result) {
  uint32_t i;
  int have_left = 0;
  int have_right = 0;

  if (tok == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  for (i = 0; i < tok->num_tokens; i++) {
    if (tok->token_ids[i] == left) {
      have_left = 1;
    }
    if (tok->token_ids[i] == right) {
      have_right = 1;
    }
  }
  if (!have_left || !have_right) {
    return PAI_ERR_INVALID_ARG;
  }
  if (tok->num_merges >= PAI_TOK_MAX_MERGES) {
    return PAI_ERR_NOMEM;
  }

  if (tok->merges == NULL) {
    tok->merges = (struct pai_tok_merge *)malloc(
        (size_t)PAI_TOK_MAX_MERGES * sizeof(*tok->merges));
    if (tok->merges == NULL) {
      return PAI_ERR_NOMEM;
    }
  }
  tok->merges[tok->num_merges].left = left;
  tok->merges[tok->num_merges].right = right;
  tok->merges[tok->num_merges].result = result;
  if (result > tok->max_id) {
    tok->max_id = result;
  }
  tok->num_merges++;
  return PAI_OK;
}

void
pai_tok_finalize(pai_tok_t *tok) {
  uint32_t base = 256u;

  if (tok == NULL) {
    return;
  }
  if (tok->max_id >= base) {
    base = tok->max_id + 1;
  }
  tok->byte_fallback_base = base;
}

int
pai_tok_id_valid(const pai_tok_t *tok, uint32_t id) {
  uint32_t i;

  if (tok == NULL) {
    return 0;
  }
  for (i = 0; i < tok->num_tokens; i++) {
    if (tok->token_ids[i] == id) {
      return 1;
    }
  }
  return (id >= tok->byte_fallback_base && id < tok->byte_fallback_base + 256u)
             ? 1
             : 0;
}

/* Longest vocab token that is a prefix of text[nbytes] starting at 0. */
static uint32_t
match_prefix(const pai_tok_t *tok, const char *text, uint32_t nbytes) {
  uint32_t best = 0;
  uint32_t best_len = 0;
  uint32_t i;

  for (i = 0; i < tok->num_tokens; i++) {
    uint32_t len = (uint32_t)strlen(tok->tokens[i]);
    if (len > nbytes || len <= best_len) {
      continue;
    }
    if (memcmp(tok->tokens[i], text, len) == 0) {
      best = i;
      best_len = len;
    }
  }
  return best_len > 0 ? best : UINT32_MAX; /* token index or none */
}

pai_status_t
pai_tok_encode(const pai_tok_t *tok, const char *text, uint32_t nbytes,
               uint32_t *out, uint32_t cap, uint32_t *out_n) {
  uint32_t *ids;
  uint32_t n = 0;
  uint32_t pos = 0;
  uint32_t i;
  uint32_t rank;

  if (tok == NULL || (text == NULL && nbytes > 0) || out == NULL ||
      out_n == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (nbytes > PAI_TOK_MAX_INPUT) {
    return PAI_ERR_INVALID_ARG;
  }

  ids = (uint32_t *)malloc(nbytes > 0 ? (size_t)nbytes * sizeof(uint32_t)
                                      : sizeof(uint32_t));
  if (ids == NULL) {
    return PAI_ERR_NOMEM;
  }

  /* Greedy longest-prefix vocabulary match with byte fallback. */
  while (pos < nbytes) {
    uint32_t idx = match_prefix(tok, text + pos, nbytes - pos);
    if (idx != UINT32_MAX) {
      ids[n++] = tok->token_ids[idx];
      pos += (uint32_t)strlen(tok->tokens[idx]);
    } else {
      ids[n++] = tok->byte_fallback_base + (uint8_t)text[pos];
      pos += 1;
    }
  }

  /* BPE: apply merges in rank order, left to right. */
  for (rank = 0; rank < tok->num_merges; rank++) {
    uint32_t left = tok->merges[rank].left;
    uint32_t right = tok->merges[rank].right;
    uint32_t result = tok->merges[rank].result;
    int applied;

    do {
      applied = 0;
      for (i = 0; i + 1 < n; i++) {
        if (ids[i] == left && ids[i + 1] == right) {
          ids[i] = result;
          memmove(ids + i + 1, ids + i + 2, (size_t)(n - i - 2) * sizeof(uint32_t));
          n--;
          applied = 1;
          break;
        }
      }
    } while (applied);
  }

  if (n > cap) {
    free(ids);
    return PAI_ERR_NOMEM;
  }
  memcpy(out, ids, (size_t)n * sizeof(uint32_t));
  *out_n = n;
  free(ids);
  return PAI_OK;
}

pai_status_t
pai_tok_decode(const pai_tok_t *tok, const uint32_t *ids, uint32_t n,
               char *out, uint32_t cap, uint32_t *out_n) {
  uint32_t written = 0;
  uint32_t i;

  if (tok == NULL || (ids == NULL && n > 0) || out == NULL || out_n == NULL ||
      cap == 0) {
    return PAI_ERR_INVALID_ARG;
  }

  for (i = 0; i < n; i++) {
    uint32_t id = ids[i];
    const char *text;
    uint32_t len;

    if (id >= tok->byte_fallback_base && id < tok->byte_fallback_base + 256u) {
      if (written + 1 >= cap) {
        return PAI_ERR_NOMEM;
      }
      out[written++] = (char)(uint8_t)(id - tok->byte_fallback_base);
      continue;
    }

    uint32_t t;
    int found = 0;
    for (t = 0; t < tok->num_tokens; t++) {
      if (tok->token_ids[t] == id) {
        text = tok->tokens[t];
        len = (uint32_t)strlen(text);
        found = 1;
        break;
      }
    }
    if (!found) {
      return PAI_ERR_MISMATCH;
    }
    if (written + len >= cap) {
      return PAI_ERR_NOMEM;
    }
    memcpy(out + written, text, len);
    written += len;
  }

  out[written] = '\0';
  *out_n = written;
  return PAI_OK;
}

static void
le_put_u32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFF);
  p[1] = (uint8_t)((v >> 8) & 0xFF);
  p[2] = (uint8_t)((v >> 16) & 0xFF);
  p[3] = (uint8_t)((v >> 24) & 0xFF);
}

static uint32_t
le_get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

uint32_t
pai_tok_blob_size(const pai_tok_t *tok) {
  uint64_t total;
  uint32_t i;

  if (tok == NULL) {
    return 0;
  }
  total = 16u; /* magic + version + num_tokens + num_merges */
  for (i = 0; i < tok->num_tokens; i++) {
    total += 4u + 1u + (uint32_t)strlen(tok->tokens[i]);
  }
  for (i = 0; i < tok->num_merges; i++) {
    total += 12u;
  }
  if (total > UINT32_MAX) {
    return 0;
  }
  return (uint32_t)total;
}

pai_status_t
pai_tok_serialize(const pai_tok_t *tok, uint8_t *out, uint32_t cap,
                  uint32_t *out_nbytes) {
  uint32_t total;
  uint8_t *p;
  uint32_t i;

  if (tok == NULL || out == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  total = pai_tok_blob_size(tok);
  if (total == 0 || total > cap) {
    return PAI_ERR_INVALID_ARG;
  }

  p = out;
  le_put_u32(p, PAI_TOK_MAGIC);
  p += 4;
  le_put_u32(p, PAI_TOK_VERSION);
  p += 4;
  le_put_u32(p, tok->num_tokens);
  p += 4;
  le_put_u32(p, tok->num_merges);
  p += 4;

  for (i = 0; i < tok->num_tokens; i++) {
    uint32_t len = (uint32_t)strlen(tok->tokens[i]);
    le_put_u32(p, tok->token_ids[i]);
    p += 4;
    *p++ = (uint8_t)len;
    memcpy(p, tok->tokens[i], len);
    p += len;
  }
  for (i = 0; i < tok->num_merges; i++) {
    le_put_u32(p, tok->merges[i].left);
    p += 4;
    le_put_u32(p, tok->merges[i].right);
    p += 4;
    le_put_u32(p, tok->merges[i].result);
    p += 4;
  }

  *out_nbytes = total;
  return PAI_OK;
}

pai_status_t
pai_tok_deserialize(pai_tok_t *tok, const uint8_t *data, uint32_t nbytes) {
  const uint8_t *p;
  uint32_t num_tokens;
  uint32_t num_merges;
  uint32_t i;
  pai_status_t st;

  if (tok == NULL || data == NULL || nbytes < 16u) {
    return PAI_ERR_INVALID_ARG;
  }
  if (le_get_u32(data) != PAI_TOK_MAGIC ||
      le_get_u32(data + 4) != PAI_TOK_VERSION) {
    return PAI_ERR_PROTOCOL;
  }
  num_tokens = le_get_u32(data + 8);
  num_merges = le_get_u32(data + 12);
  if (num_tokens > PAI_TOK_MAX_TOKENS || num_merges > PAI_TOK_MAX_MERGES) {
    return PAI_ERR_PROTOCOL;
  }

  p = data + 16;
  pai_tok_destroy(tok);
  pai_tok_init(tok);

  for (i = 0; i < num_tokens; i++) {
    char text[PAI_TOK_MAX_TOKEN_LEN + 1];
    uint32_t id;
    uint32_t len;
    if ((uint64_t)(p - data) + 5 > nbytes) {
      pai_tok_destroy(tok);
      return PAI_ERR_PROTOCOL;
    }
    id = le_get_u32(p);
    p += 4;
    len = *p++;
    if (len == 0 || len > PAI_TOK_MAX_TOKEN_LEN ||
        (uint64_t)(p - data) + len > nbytes) {
      pai_tok_destroy(tok);
      return PAI_ERR_PROTOCOL;
    }
    memcpy(text, p, len);
    text[len] = '\0';
    p += len;
    st = pai_tok_add_token(tok, id, text);
    if (st != PAI_OK) {
      pai_tok_destroy(tok);
      return st;
    }
  }

  for (i = 0; i < num_merges; i++) {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    if ((uint64_t)(p - data) + 12 > nbytes) {
      pai_tok_destroy(tok);
      return PAI_ERR_PROTOCOL;
    }
    left = le_get_u32(p);
    p += 4;
    right = le_get_u32(p);
    p += 4;
    result = le_get_u32(p);
    p += 4;
    st = pai_tok_add_merge(tok, left, right, result);
    if (st != PAI_OK) {
      pai_tok_destroy(tok);
      return st;
    }
  }

  pai_tok_finalize(tok);
  return PAI_OK;
}
