/*
 * ProsperoAI — `.pai` Model Container (whitepaper §20)
 *
 * The native ProsperoAI model package: a portable model + a vehicle
 * for hardware-specific compiled assets. v0 implements the portable
 * sections; target slices and execution profiles are reserved placeholders.
 *
 * Layout (all integers little-endian):
 *
 *   [0]    u32 magic           0x43494150 ("PAIC")
 *   [4]    u16 version         1
 *   [6]    u16 flags           0
 *   [8]    u32 num_sections
 *   [12]   u32 crc32           over bytes [0,12) + section table + all
 *                              section payloads
 *   [16]   section table       num_sections x 16 bytes:
 *                                u32 type
 *                                u32 flags
 *                                u32 offset   (from file start)
 *                                u32 size     (payload bytes)
 *   [16 + 16*num_sections]     section payloads (each 8-aligned)
 *
 * Sections (pai_pai_section_type):
 *   META       structured model metadata (name, family, context, ...)
 *   MANIFEST   tensor manifest: name -> value id, dtype, shape, offset
 *              into the WEIGHTS section (the canonical weights blob)
 *   WEIGHTS    canonical weights, raw bytes, referenced by the manifest
 *   IR         encoded Prospero IR blob (§10.1, pai_ir_encode format)
 *   TOKENIZER  serialized tokenizer blob (models/tokenizer.h)
 *   TARGET     PS5 target slice (packed tensors / kernel candidates) —
 *              reserved in v0, must be empty or opaque
 *   PROFILE    execution profiles (autotuning results) — reserved in v0
 *
 * The container is a portable model package (canonical sections) that
 * may later carry hardware slices, matching §21 (universal container
 * model / fat binary). Integrity: the file CRC plus per-section access
 * is validated on open; readers must never trust section payloads.
 *
 * The manifest payload format (v0):
 *   u32 count
 *   per entry:
 *     u16 name_len; u8 name[name_len]
 *     u32 value_id       (IR value id this tensor belongs to)
 *     u8  dtype          (pai_dtype_t)
 *     u8  rank
 *     u16 reserved
 *     u64 shape[6]
 *     u64 offset         (byte offset into WEIGHTS)
 *     u64 size_bytes
 */

#ifndef PAI_PAI_H
#define PAI_PAI_H

#include <pai/dtype.h>
#include <pai/error.h>
#include <pai/tensor.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_PAI_MAGIC   0x43494150u /* "PAIC"                              */
#define PAI_PAI_VERSION 1u
#define PAI_PAI_HEADER_SIZE 16u
#define PAI_PAI_MAX_SECTIONS 8u
#define PAI_PAI_ALIGN 8u

typedef enum pai_pai_section_type {
  PAI_PAI_SEC_META = 1,
  PAI_PAI_SEC_MANIFEST,
  PAI_PAI_SEC_WEIGHTS,
  PAI_PAI_SEC_IR,
  PAI_PAI_SEC_TOKENIZER,
  PAI_PAI_SEC_TARGET,   /* PS5 target slice (reserved v0)                */
  PAI_PAI_SEC_PROFILE,  /* execution profiles (reserved v0)              */
} pai_pai_section_type_t;

/* Model families (metadata). */
#define PAI_PAI_FAMILY_LLM 1u

/* Meta payload layout (v0), fixed size PAI_PAI_META_SIZE. */
#define PAI_PAI_META_SIZE 64u
#define PAI_PAI_NAME_MAX 31u

typedef struct pai_pai_meta {
  char     name[PAI_PAI_NAME_MAX + 1];
  uint32_t family;      /* PAI_PAI_FAMILY_*                              */
  uint32_t context_len; /* max tokens in one sequence                    */
  uint32_t num_layers;  /* transformer layers (KV cache sizing)          */
  uint32_t kv_bytes_per_token; /* bytes per layer per token position     */
  uint32_t vocab_size;
  uint32_t reserved;
} pai_pai_meta_t;

/* Manifest entry (see layout comment above). */
typedef struct pai_pai_tensor {
  char     name[64];
  uint32_t value_id;
  pai_dtype_t dtype;
  uint32_t rank;
  uint64_t shape[PAI_TENSOR_MAX_RANK];
  uint64_t offset;      /* into the WEIGHTS section                      */
  uint64_t size_bytes;
} pai_pai_tensor_t;

#define PAI_PAI_MAX_TENSORS 512u

/* Parsed container view (does not own the file buffer). */
typedef struct pai_pai_container {
  uint32_t version;
  uint32_t flags;
  uint32_t num_sections;
  struct {
    uint32_t type;
    uint32_t flags;
    uint32_t offset;
    uint32_t size;
  } sections[PAI_PAI_MAX_SECTIONS];
  const uint8_t *data;  /* whole-file buffer                              */
  uint32_t nbytes;
} pai_pai_container_t;

/* ------------------------------------------------------------------ */
/* Writer (builder)                                                    */
/* ------------------------------------------------------------------ */

typedef struct pai_pai_builder {
  uint32_t num_sections;
  struct {
    uint32_t type;
    const uint8_t *data;
    uint32_t size;
  } sections[PAI_PAI_MAX_SECTIONS];
} pai_pai_builder_t;

void pai_pai_builder_init(pai_pai_builder_t *builder);

/* Attach a section payload (copied into the encoded blob). Returns
 * PAI_ERR_NOMEM when the section table is full. */
pai_status_t pai_pai_builder_add(pai_pai_builder_t *builder, uint32_t type,
                                 const void *data, uint32_t size);

/* Encode the meta structure into its fixed payload. */
pai_status_t pai_pai_meta_encode(const pai_pai_meta_t *meta, uint8_t *out,
                                 uint32_t cap, uint32_t *out_nbytes);

/* Encode a tensor manifest. */
pai_status_t pai_pai_manifest_encode(const pai_pai_tensor_t *tensors,
                                     uint32_t count, uint8_t *out,
                                     uint32_t cap, uint32_t *out_nbytes);

/* Size of the encoded container. */
uint32_t pai_pai_encoded_size(const pai_pai_builder_t *builder);

/* Build the container blob. */
pai_status_t pai_pai_build(const pai_pai_builder_t *builder, uint8_t *out,
                           uint32_t cap, uint32_t *out_nbytes);

/* Convenience: write a built blob to a file. */
pai_status_t pai_pai_write_file(const char *path, const uint8_t *blob,
                                uint32_t nbytes);

/* ------------------------------------------------------------------ */
/* Reader                                                              */
/* ------------------------------------------------------------------ */

/* Open + validate a container over an in-memory buffer (not owned). */
pai_status_t pai_pai_open(const uint8_t *data, uint32_t nbytes,
                          pai_pai_container_t *out);

/* Payload of a section (NULL when absent). */
const uint8_t *pai_pai_section(const pai_pai_container_t *container,
                               uint32_t type, uint32_t *out_size);

pai_status_t pai_pai_meta_decode(const uint8_t *data, uint32_t size,
                                 pai_pai_meta_t *out);
pai_status_t pai_pai_manifest_decode(const uint8_t *data, uint32_t size,
                                     pai_pai_tensor_t *out_tensors,
                                     uint32_t max_tensors,
                                     uint32_t *out_count);

/* Convenience: read a whole file into a malloc'd buffer (caller frees).
 * Returns PAI_ERR_IO on I/O failures. */
pai_status_t pai_pai_read_file(const char *path, uint8_t **out,
                               uint32_t *out_nbytes);

/* Stable section name for diagnostics. */
const char *pai_pai_section_name(uint32_t type);

#ifdef __cplusplus
}
#endif

#endif /* PAI_PAI_H */
