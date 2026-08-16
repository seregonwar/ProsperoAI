#include "test.h"

#include <tokenizer.h>

#include <stdlib.h>
#include <string.h>

TEST_MAIN_BEGIN()

{
  /* Basic vocab round-trip. */
  pai_tok_t tok;
  uint32_t ids[64];
  uint32_t n = 0;
  char text[256];
  uint32_t text_n = 0;

  pai_tok_init(&tok);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 10, "hello"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 11, " "), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 12, "world"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 13, "!"), PAI_OK);
  pai_tok_finalize(&tok);

  CHECK_EQ_INT(pai_tok_encode(&tok, "hello world!", 12, ids, 64, &n), PAI_OK);
  CHECK_EQ_UINT(n, 4);
  CHECK_EQ_UINT(ids[0], 10);
  CHECK_EQ_UINT(ids[1], 11);
  CHECK_EQ_UINT(ids[2], 12);
  CHECK_EQ_UINT(ids[3], 13);

  CHECK_EQ_INT(pai_tok_decode(&tok, ids, n, text, sizeof(text), &text_n),
               PAI_OK);
  CHECK_EQ_UINT(text_n, 12);
  CHECK(strcmp(text, "hello world!") == 0);

  pai_tok_destroy(&tok);
}

{
  /* Byte fallback: unknown bytes round-trip losslessly. */
  pai_tok_t tok;
  uint32_t ids[64];
  uint32_t n = 0;
  char text[256];
  uint32_t text_n = 0;
  /* "héllo" in UTF-8: h C3 A9 l l o */
  static const char in[] = {'h', (char)0xC3, (char)0xA9, 'l', 'l', 'o'};

  pai_tok_init(&tok);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 1, "hello"), PAI_OK);
  pai_tok_finalize(&tok);

  CHECK_EQ_INT(pai_tok_encode(&tok, in, sizeof(in), ids, 64, &n), PAI_OK);
  CHECK_EQ_UINT(n, 6); /* h, C3, A9, l, l, o all fall back */
  CHECK_EQ_UINT(ids[0], tok.byte_fallback_base + 'h');
  CHECK_EQ_UINT(ids[1], tok.byte_fallback_base + 0xC3);
  CHECK_EQ_UINT(ids[2], tok.byte_fallback_base + 0xA9);

  CHECK_EQ_INT(pai_tok_decode(&tok, ids, n, text, sizeof(text), &text_n),
               PAI_OK);
  CHECK_EQ_UINT(text_n, 6);
  CHECK(memcmp(text, in, 6) == 0);

  pai_tok_destroy(&tok);
}

{
  /* Merges: "aab" = [a][a][b] -> merge [a][b] -> [a][ab]. */
  pai_tok_t tok;
  uint32_t ids[64];
  uint32_t n = 0;
  char text[256];
  uint32_t text_n = 0;

  pai_tok_init(&tok);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 0, "a"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 1, "b"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 2, "ab"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_merge(&tok, 0, 1, 2), PAI_OK);
  pai_tok_finalize(&tok);

  CHECK_EQ_INT(pai_tok_encode(&tok, "aab", 3, ids, 64, &n), PAI_OK);
  CHECK_EQ_UINT(n, 2);
  CHECK_EQ_UINT(ids[0], 0);
  CHECK_EQ_UINT(ids[1], 2);

  CHECK_EQ_INT(pai_tok_decode(&tok, ids, n, text, sizeof(text), &text_n),
               PAI_OK);
  CHECK(strcmp(text, "aab") == 0);

  /* Invalid merge operands are rejected. */
  CHECK_EQ_INT(pai_tok_add_merge(&tok, 0, 99, 2), PAI_ERR_INVALID_ARG);

  pai_tok_destroy(&tok);
}

{
  /* Serialization round-trip preserves everything. */
  pai_tok_t a;
  pai_tok_t b;
  uint8_t *blob;
  uint32_t blob_size;
  uint32_t nbytes = 0;
  uint32_t ids[64];
  uint32_t n = 0;
  char text[256];
  uint32_t text_n = 0;

  pai_tok_init(&a);
  CHECK_EQ_INT(pai_tok_add_token(&a, 0, "a"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&a, 1, "b"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&a, 2, "ab"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_token(&a, 200, "merge-pair"), PAI_OK);
  CHECK_EQ_INT(pai_tok_add_merge(&a, 0, 1, 2), PAI_OK);
  pai_tok_finalize(&a);

  blob_size = pai_tok_blob_size(&a);
  CHECK(blob_size > 0);
  blob = (uint8_t *)malloc(blob_size);
  CHECK(blob != NULL);
  CHECK_EQ_INT(pai_tok_serialize(&a, blob, blob_size, &nbytes), PAI_OK);
  CHECK_EQ_UINT(nbytes, blob_size);

  pai_tok_init(&b);
  CHECK_EQ_INT(pai_tok_deserialize(&b, blob, nbytes), PAI_OK);
  CHECK_EQ_UINT(b.num_tokens, a.num_tokens);
  CHECK_EQ_UINT(b.num_merges, a.num_merges);
  CHECK_EQ_UINT(b.byte_fallback_base, a.byte_fallback_base);

  CHECK_EQ_INT(pai_tok_encode(&b, "ab", 2, ids, 64, &n), PAI_OK);
  CHECK_EQ_UINT(n, 1);
  CHECK_EQ_UINT(ids[0], 2);
  CHECK_EQ_INT(pai_tok_decode(&b, ids, n, text, sizeof(text), &text_n),
               PAI_OK);
  CHECK(strcmp(text, "ab") == 0);

  /* Corrupted magic / truncation are rejected. */
  blob[0] ^= 0xFF;
  CHECK_EQ_INT(pai_tok_deserialize(&b, blob, nbytes), PAI_ERR_PROTOCOL);
  blob[0] ^= 0xFF;
  CHECK_EQ_INT(pai_tok_deserialize(&b, blob, 8), PAI_ERR_INVALID_ARG);

  pai_tok_destroy(&a);
  pai_tok_destroy(&b);
  free(blob);
}

{
  /* Validation. */
  pai_tok_t tok;
  pai_tok_init(&tok);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 1, ""), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_tok_add_token(&tok, 1, NULL), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_tok_encode(&tok, "x", 1, NULL, 4, NULL), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_tok_id_valid(&tok, 0), 0);
  pai_tok_destroy(&tok);
}

TEST_MAIN_END()
