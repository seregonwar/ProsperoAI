/*
 * ProsperoAI — tokenizer (whitepaper §3.2 CPU components, §20 container)
 *
 * A compact byte-level BPE tokenizer that ships inside the .pai container
 * (Tokenizer / Preprocessor section). It supports:
 *
 *   - an explicit vocabulary: token text -> token id (ids are caller
 *     chosen; encode/decode are exact inverses over the vocab);
 *   - merge rules (left, right) -> result with rank = insertion order,
 *     applied greedily across the id sequence (classic BPE);
 *   - a byte-fallback range: any input byte not covered by the vocab
 *     encodes to (byte_fallback_base + byte), so arbitrary UTF-8 /
 *     binary input round-trips losslessly.
 *
 * The tokenizer is serializable to a standalone blob (magic + version +
 * tokens + merges) that the .pai container stores; integrity of the
 * blob on disk is covered by the container's section CRC (§20).
 *
 * Reference-quality on purpose: tokenization is CPU-side (§3.2) and not
 * the hot path; the desktop toolchain can ship a richer tokenizer blob.
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

/*
 * Create an empty tokenizer. Call pai_tok_add_token / pai_tok_add_merge,
 * then pai_tok_finalize to compute the byte-fallback base. Destroy with
 * pai_tok_destroy.
 */
void pai_tok_init(pai_tok_t *tok);
void pai_tok_destroy(pai_tok_t *tok);

/* Register an explicit token. text must be 1..63 chars; ids may repeat
 * a byte value. Returns PAI_ERR_NOMEM when full. */
pai_status_t pai_tok_add_token(pai_tok_t *tok, uint32_t id, const char *text);

/*
 * Register a merge rule (left, right) -> result with rank = insertion
 * order. Both operands must be known ids; the result may be a new id.
 */
pai_status_t pai_tok_add_merge(pai_tok_t *tok, uint32_t left, uint32_t right,
                               uint32_t result);

/* Compute byte_fallback_base (max(256, max_id + 1)). Optional but
 * recommended before encode so byte fallbacks are stable. */
void pai_tok_finalize(pai_tok_t *tok);

/*
 * Encode UTF-8/binary `text` (nbytes) into token ids. Writes at most
 * `cap` ids into `out` and sets *out_n. The emitted sequence starts as
 * greedy longest-prefix vocab matches with byte fallback, then BPE
 * merges are applied in rank order. Returns PAI_ERR_INVALID_ARG on bad
 * args and PAI_ERR_NOMEM when the result exceeds cap.
 */
pai_status_t pai_tok_encode(const pai_tok_t *tok, const char *text,
                            uint32_t nbytes, uint32_t *out, uint32_t cap,
                            uint32_t *out_n);

/*
 * Decode ids back to text into `out` (cap bytes incl. NUL). Sets
 * *out_n to the number of bytes written (excl. NUL). Ids in the byte
 * fallback range become their single byte. Returns PAI_ERR_MISMATCH on
 * unknown ids and PAI_ERR_NOMEM when the output is too small.
 */
pai_status_t pai_tok_decode(const pai_tok_t *tok, const uint32_t *ids,
                            uint32_t n, char *out, uint32_t cap,
                            uint32_t *out_n);

/* True when the id is a known token or a byte-fallback id. */
int pai_tok_id_valid(const pai_tok_t *tok, uint32_t id);

/* ------------------------------------------------------------------ */
/* Serialization                                                       */
/* ------------------------------------------------------------------ */

/* Size of the serialized blob. */
uint32_t pai_tok_blob_size(const pai_tok_t *tok);

pai_status_t pai_tok_serialize(const pai_tok_t *tok, uint8_t *out,
                               uint32_t cap, uint32_t *out_nbytes);

/* Parse a blob; replaces the contents of `tok`. PAI_ERR_PROTOCOL on
 * malformed input. */
pai_status_t pai_tok_deserialize(pai_tok_t *tok, const uint8_t *data,
                                 uint32_t nbytes);

#ifdef __cplusplus
}
#endif

#endif /* PAI_MODELS_TOKENIZER_H */
