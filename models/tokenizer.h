/*
 * ProsperoAI — tokenizer (whitepaper §3.2, §20).
 * Compact byte-level BPE tokenizer shipped inside the .pai container:
 * explicit vocab, rank-ordered merge rules, and a byte-fallback range
 * so arbitrary UTF-8/binary input round-trips losslessly.
 */

#ifndef PAI_MODELS_TOKENIZER_H
#define PAI_MODELS_TOKENIZER_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 65536 covers real LLM vocabularies (LLaMA-2/3: 32k tokens); the
 * serialized blob stores each token text length in a u8, so token
 * texts are capped at 255 chars (longer GGUF tokens are skipped by
 * the adapter with a warning). */
#define PAI_TOK_MAX_TOKENS 65536u
#define PAI_TOK_MAX_MERGES 8192u
#define PAI_TOK_MAX_TOKEN_LEN 255u  /* chars (excl. NUL) per token text */
#define PAI_TOK_MAX_INPUT 65536u    /* max encoded input bytes          */

/* Serialized blob. */
#define PAI_TOK_MAGIC   0x314B5450u /* "PTK1" (LE: 50 54 4B 31)          */
#define PAI_TOK_VERSION 1u

typedef struct pai_tok {
  uint32_t num_tokens;
  uint32_t num_merges;
  char    (*tokens)[PAI_TOK_MAX_TOKEN_LEN + 1]; /* heap: token texts  */
  uint32_t *token_ids;             /* heap: id per token (may be sparse) */
  struct pai_tok_merge {
    uint32_t left;
    uint32_t right;
    uint32_t result;
  } *merges;                       /* heap; insertion order = rank       */
  uint32_t byte_fallback_base;     /* ids >= base are single bytes       */
  uint32_t max_id;                 /* highest explicit token id          */
} pai_tok_t;

/* Create an empty tokenizer; add tokens/merges, then pai_tok_finalize. */
void pai_tok_init(pai_tok_t *tok);
void pai_tok_destroy(pai_tok_t *tok);

/* Register an explicit token (text 1..PAI_TOK_MAX_TOKEN_LEN chars). */
pai_status_t pai_tok_add_token(pai_tok_t *tok, uint32_t id, const char *text);

/* Register a merge rule (left, right) -> result, rank = insertion order. */
pai_status_t pai_tok_add_merge(pai_tok_t *tok, uint32_t left, uint32_t right,
                               uint32_t result);

/* byte_fallback_base = max(256, max_id + 1); call before encode so
 * byte fallbacks are stable. */
void pai_tok_finalize(pai_tok_t *tok);

/*
 * Encode text into token ids: greedy longest-prefix vocab matches with
 * byte fallback, then BPE merges in rank order.
 */
pai_status_t pai_tok_encode(const pai_tok_t *tok, const char *text,
                            uint32_t nbytes, uint32_t *out, uint32_t cap,
                            uint32_t *out_n);

/* Decode ids to text (cap bytes incl. NUL); byte-fallback ids become
 * their single byte. */
pai_status_t pai_tok_decode(const pai_tok_t *tok, const uint32_t *ids,
                            uint32_t n, char *out, uint32_t cap,
                            uint32_t *out_n);

/* True when the id is a known token or a byte-fallback id. */
int pai_tok_id_valid(const pai_tok_t *tok, uint32_t id);

/* Size of the serialized blob. */
uint32_t pai_tok_blob_size(const pai_tok_t *tok);

pai_status_t pai_tok_serialize(const pai_tok_t *tok, uint8_t *out,
                               uint32_t cap, uint32_t *out_nbytes);

/* Parse a blob; replaces the contents of `tok`. */
pai_status_t pai_tok_deserialize(pai_tok_t *tok, const uint8_t *data,
                                 uint32_t nbytes);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_TOKENIZER_H */
