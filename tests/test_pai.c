#include "test.h"

#include <pai/pai.h>

#include <stdlib.h>
#include <string.h>

static const uint64_t k_shape[] = {4, 8};

TEST_MAIN_BEGIN()

{
  /* Build a container with all portable sections and reopen it. */
  pai_pai_meta_t meta;
  pai_pai_tensor_t tensors[2];
  pai_pai_builder_t builder;
  static const uint8_t weights[64] = {0};
  static const uint8_t ir_blob[32] = {1, 2, 3};
  uint8_t meta_blob[PAI_PAI_META_SIZE];
  uint8_t manifest_blob[512];
  uint32_t meta_n = 0;
  uint32_t manifest_n = 0;
  uint32_t total = 0;
  uint8_t *blob;
  pai_pai_container_t container;
  const uint8_t *sec;
  uint32_t sec_size;
  pai_pai_meta_t meta_out;
  pai_pai_tensor_t tensors_out[2];
  uint32_t count = 0;
  uint32_t i;

  memset(&meta, 0, sizeof(meta));
  strcpy(meta.name, "toy-llm");
  meta.family = PAI_PAI_FAMILY_LLM;
  meta.context_len = 1024;
  meta.num_layers = 2;
  meta.kv_bytes_per_token = 8;
  meta.vocab_size = 32000;

  memset(tensors, 0, sizeof(tensors));
  strcpy(tensors[0].name, "w_embed");
  tensors[0].value_id = 2;
  tensors[0].dtype = PAI_DTYPE_F32;
  tensors[0].rank = 2;
  memcpy(tensors[0].shape, k_shape, sizeof(k_shape));
  tensors[0].offset = 0;
  tensors[0].size_bytes = 32;
  strcpy(tensors[1].name, "w_out");
  tensors[1].value_id = 5;
  tensors[1].dtype = PAI_DTYPE_F32;
  tensors[1].rank = 2;
  memcpy(tensors[1].shape, k_shape, sizeof(k_shape));
  tensors[1].offset = 32;
  tensors[1].size_bytes = 32;

  CHECK_EQ_INT(pai_pai_meta_encode(&meta, meta_blob, sizeof(meta_blob),
                                   &meta_n),
               PAI_OK);
  CHECK_EQ_UINT(meta_n, PAI_PAI_META_SIZE);
  CHECK_EQ_INT(pai_pai_manifest_encode(tensors, 2, manifest_blob,
                                       sizeof(manifest_blob), &manifest_n),
               PAI_OK);

  pai_pai_builder_init(&builder);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_META, meta_blob,
                                   meta_n),
               PAI_OK);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_MANIFEST,
                                   manifest_blob, manifest_n),
               PAI_OK);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_WEIGHTS, weights,
                                   sizeof(weights)),
               PAI_OK);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, PAI_PAI_SEC_IR, ir_blob,
                                   sizeof(ir_blob)),
               PAI_OK);

  total = pai_pai_encoded_size(&builder);
  CHECK(total > 0);
  blob = (uint8_t *)malloc(total);
  CHECK(blob != NULL);
  CHECK_EQ_INT(pai_pai_build(&builder, blob, total, &total), PAI_OK);

  /* Open + read back every section. */
  CHECK_EQ_INT(pai_pai_open(blob, total, &container), PAI_OK);
  CHECK_EQ_UINT(container.num_sections, 4);

  sec = pai_pai_section(&container, PAI_PAI_SEC_META, &sec_size);
  CHECK(sec != NULL);
  CHECK_EQ_UINT(sec_size, PAI_PAI_META_SIZE);
  CHECK_EQ_INT(pai_pai_meta_decode(sec, sec_size, &meta_out), PAI_OK);
  CHECK(strcmp(meta_out.name, "toy-llm") == 0);
  CHECK_EQ_UINT(meta_out.family, PAI_PAI_FAMILY_LLM);
  CHECK_EQ_UINT(meta_out.context_len, 1024);
  CHECK_EQ_UINT(meta_out.num_layers, 2);
  CHECK_EQ_UINT(meta_out.vocab_size, 32000);

  sec = pai_pai_section(&container, PAI_PAI_SEC_MANIFEST, &sec_size);
  CHECK(sec != NULL);
  CHECK_EQ_INT(pai_pai_manifest_decode(sec, sec_size, tensors_out, 2, &count),
               PAI_OK);
  CHECK_EQ_UINT(count, 2);
  for (i = 0; i < 2; i++) {
    CHECK(strcmp(tensors_out[i].name, tensors[i].name) == 0);
    CHECK_EQ_UINT(tensors_out[i].value_id, tensors[i].value_id);
    CHECK_EQ_UINT(tensors_out[i].offset, tensors[i].offset);
    CHECK_EQ_UINT(tensors_out[i].size_bytes, tensors[i].size_bytes);
    CHECK_EQ_UINT(tensors_out[i].shape[0], k_shape[0]);
    CHECK_EQ_UINT(tensors_out[i].shape[1], k_shape[1]);
  }

  sec = pai_pai_section(&container, PAI_PAI_SEC_WEIGHTS, &sec_size);
  CHECK(sec != NULL);
  CHECK_EQ_UINT(sec_size, 64);
  sec = pai_pai_section(&container, PAI_PAI_SEC_IR, &sec_size);
  CHECK(sec != NULL);
  CHECK_EQ_UINT(sec_size, 32);
  CHECK(memcmp(sec, ir_blob, 3) == 0);

  /* Absent sections return NULL. */
  CHECK(pai_pai_section(&container, PAI_PAI_SEC_PROFILE, &sec_size) == NULL);
  CHECK_EQ_UINT(sec_size, 0);

  /* Corruption is detected. */
  blob[20] ^= 0x40; /* inside the first section payload */
  CHECK_EQ_INT(pai_pai_open(blob, total, &container), PAI_ERR_PROTOCOL);
  blob[20] ^= 0x40;

  /* Truncation is detected. */
  CHECK_EQ_INT(pai_pai_open(blob, total - 1, &container), PAI_ERR_PROTOCOL);

  /* Bad magic is detected. */
  blob[0] ^= 0xFF;
  CHECK_EQ_INT(pai_pai_open(blob, total, &container), PAI_ERR_PROTOCOL);
  blob[0] ^= 0xFF;

  /* Empty builder is rejected. */
  {
    pai_pai_builder_t empty;
    pai_pai_builder_init(&empty);
    CHECK_EQ_INT(pai_pai_build(&empty, blob, total, &total),
                 PAI_ERR_INVALID_ARG);
  }

  free(blob);
}

{
  /* File write + read round-trip. */
  pai_pai_meta_t meta;
  pai_pai_tensor_t tensors[1];
  pai_pai_builder_t builder;
  uint8_t meta_blob[PAI_PAI_META_SIZE];
  uint8_t manifest_blob[512];
  uint32_t meta_n = 0;
  uint32_t manifest_n = 0;
  uint32_t total = 0;
  uint8_t *blob;
  uint8_t *loaded = NULL;
  uint32_t loaded_n = 0;
  pai_pai_container_t container;

  memset(&meta, 0, sizeof(meta));
  strcpy(meta.name, "disk-model");
  meta.family = PAI_PAI_FAMILY_LLM;
  meta.context_len = 128;
  meta.num_layers = 1;
  meta.kv_bytes_per_token = 4;
  meta.vocab_size = 100;

  memset(tensors, 0, sizeof(tensors));
  strcpy(tensors[0].name, "w");
  tensors[0].value_id = 1;
  tensors[0].dtype = PAI_DTYPE_F32;
  tensors[0].rank = 1;
  tensors[0].shape[0] = 16;
  tensors[0].size_bytes = 64;

  pai_pai_meta_encode(&meta, meta_blob, sizeof(meta_blob), &meta_n);
  pai_pai_manifest_encode(tensors, 1, manifest_blob, sizeof(manifest_blob),
                          &manifest_n);
  pai_pai_builder_init(&builder);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_META, meta_blob, meta_n);
  pai_pai_builder_add(&builder, PAI_PAI_SEC_MANIFEST, manifest_blob,
                      manifest_n);
  {
    static uint8_t w[64];
    memset(w, 0xAB, sizeof(w));
    pai_pai_builder_add(&builder, PAI_PAI_SEC_WEIGHTS, w, sizeof(w));
  }

  total = pai_pai_encoded_size(&builder);
  blob = (uint8_t *)malloc(total);
  CHECK(blob != NULL);
  CHECK_EQ_INT(pai_pai_build(&builder, blob, total, &total), PAI_OK);
  CHECK_EQ_INT(pai_pai_write_file("test_tmp.pai", blob, total), PAI_OK);

  CHECK_EQ_INT(pai_pai_read_file("test_tmp.pai", &loaded, &loaded_n), PAI_OK);
  CHECK_EQ_UINT(loaded_n, total);
  CHECK(memcmp(loaded, blob, total) == 0);
  CHECK_EQ_INT(pai_pai_open(loaded, loaded_n, &container), PAI_OK);
  CHECK(strcmp(container.data ? "x" : "x", "x") == 0);

  /* Missing file -> PAI_ERR_IO. */
  CHECK_EQ_INT(pai_pai_read_file("no_such_file.pai", &loaded, &loaded_n),
               PAI_ERR_IO);

  free(loaded);
  free(blob);
}

{
  /* Validation. */
  pai_pai_builder_t builder;
  uint8_t buf[256];
  uint32_t n = 0;

  pai_pai_builder_init(&builder);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, 0, buf, 4), PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_pai_builder_add(&builder, 99, buf, 4),
               PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_pai_meta_encode(NULL, buf, sizeof(buf), &n),
               PAI_ERR_INVALID_ARG);
  CHECK_EQ_INT(pai_pai_open(buf, 4, NULL), PAI_ERR_INVALID_ARG);
}

TEST_MAIN_END()
