/*
 * ProsperoAI — GGUF reader (whitepaper §9.3).
 * Minimal, bounds-checked reader for the llama.cpp GGUF format
 * (versions 1-3): header, metadata KV, tensor infos and quantized
 * data. Dequantization covers the types real LLaMA models ship with
 * (F32/F16/BF16, Q4_0..Q8_1, Q2_K..Q8_K); the K-quant formulas are
 * ported verbatim from llama.cpp's ggml-quants.c. Host-side tooling.
 */

#ifndef PAI_ADAPTERS_LLAMA_GGUF_H
#define PAI_ADAPTERS_LLAMA_GGUF_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_GGUF_MAGIC 0x46554747u /* "GGUF" little-endian               */
#define PAI_GGUF_MAX_VERSION 3u
#define PAI_GGUF_ALIGN_DEFAULT 32u
#define PAI_GGUF_MAX_TENSORS 1024u

/*
 * Maximum per-dimension extent accepted from a tensor-info record.
 * Real GGUF models stay far below this (LLaMA vocab ~320k, hidden
 * ~64k); the cap keeps dims * numel products from wrapping u64 and
 * bounds downstream allocation math against hostile files.
 */
#define PAI_GGUF_MAX_DIM (1u << 26)

/* ggml_type values used by GGUF (stable across versions). */
typedef enum pai_gguf_type {
  PAI_GGUF_F32 = 0,
  PAI_GGUF_F16 = 1,
  PAI_GGUF_Q4_0 = 2,
  PAI_GGUF_Q4_1 = 3,
  PAI_GGUF_Q5_0 = 6,
  PAI_GGUF_Q5_1 = 7,
  PAI_GGUF_Q8_0 = 8,
  PAI_GGUF_Q8_1 = 9,
  PAI_GGUF_Q2_K = 10,
  PAI_GGUF_Q3_K = 11,
  PAI_GGUF_Q4_K = 12,
  PAI_GGUF_Q5_K = 13,
  PAI_GGUF_Q6_K = 14,
  PAI_GGUF_Q8_K = 15,
  PAI_GGUF_BF16 = 20,
  PAI_GGUF_TYPE_COUNT,
} pai_gguf_type_t;

typedef struct pai_gguf_tensor {
  char name[128];
  uint32_t n_dims;
  uint64_t dims[4];
  uint32_t type;   /* pai_gguf_type_t */
  uint64_t offset; /* byte offset into the (aligned) data region */
} pai_gguf_tensor_t;

/* Read-only view over a GGUF file buffer (does not own it). */
typedef struct pai_gguf {
  const uint8_t *data;
  uint64_t size;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t kv_count;         /* metadata KV entries                    */
  uint64_t alignment;
  const uint8_t *kv_start;   /* start of the metadata KV section      */
  const uint8_t *data_start; /* aligned start of tensor data          */
  pai_gguf_tensor_t *tensors; /* parsed tensor infos (owned)          */
} pai_gguf_t;

/* Parse + validate a GGUF buffer. Fails with PAI_ERR_PROTOCOL on
 * malformed input and PAI_ERR_UNSUPPORTED for unknown versions. */
pai_status_t pai_gguf_open(const uint8_t *data, uint64_t size,
                           pai_gguf_t *out);

void pai_gguf_close(pai_gguf_t *f);

/* Metadata KV getters (key is NUL-terminated; the file stores lengths
 * internally). Returns PAI_ERR_MISMATCH when the key is absent. */
pai_status_t pai_gguf_get_u32(const pai_gguf_t *f, const char *key,
                              uint32_t *out);
pai_status_t pai_gguf_get_f32(const pai_gguf_t *f, const char *key,
                              float *out);
pai_status_t pai_gguf_get_string(const pai_gguf_t *f, const char *key,
                                 char *out, uint64_t cap);
pai_status_t pai_gguf_get_string_array(const pai_gguf_t *f, const char *key,
                                       char ***out, uint64_t *out_n);
pai_status_t pai_gguf_get_f32_array(const pai_gguf_t *f, const char *key,
                                    float **out, uint64_t *out_n);

/* Find a tensor by exact name (NULL when absent). */
const pai_gguf_tensor_t *pai_gguf_find_tensor(const pai_gguf_t *f,
                                              const char *name);

uint64_t pai_gguf_tensor_numel(const pai_gguf_tensor_t *t);

/* Bytes of stored data for `numel` elements of `type` (0 when the type
 * is unsupported or the layout is irregular). */
uint64_t pai_gguf_type_bytes(uint32_t type, uint64_t numel);

/* Dequantize a tensor to f32 (malloc'd, caller frees). Returns
 * PAI_ERR_UNSUPPORTED for non-covered types and PAI_ERR_PROTOCOL when
 * the tensor's bytes fall outside the file. */
pai_status_t pai_gguf_dequant(const pai_gguf_t *f, const pai_gguf_tensor_t *t,
                              float **out_f32, uint64_t *out_n);

#ifdef __cplusplus
}
#endif

#endif /* PAI_ADAPTERS_LLAMA_GGUF_H */
