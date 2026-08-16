#include "pai.h"

#include <protocol/protocol.h> /* pai_proto_crc32 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const k_section_names[] = {
    "?",        "meta",   "manifest", "weights", "ir",
    "tokenizer", "target", "profile",
};

const char *
pai_pai_section_name(uint32_t type) {
  if (type == 0 || type > PAI_PAI_SEC_PROFILE) {
    return "?";
  }
  return k_section_names[type];
}

void
pai_pai_builder_init(pai_pai_builder_t *builder) {
  memset(builder, 0, sizeof(*builder));
}

pai_status_t
pai_pai_builder_add(pai_pai_builder_t *builder, uint32_t type,
                    const void *data, uint32_t size) {
  if (builder == NULL || type == 0 || type > PAI_PAI_SEC_PROFILE ||
      (data == NULL && size > 0) || builder->num_sections >= PAI_PAI_MAX_SECTIONS) {
    return PAI_ERR_INVALID_ARG;
  }
  builder->sections[builder->num_sections].type = type;
  builder->sections[builder->num_sections].data = (const uint8_t *)data;
  builder->sections[builder->num_sections].size = size;
  builder->num_sections++;
  return PAI_OK;
}

uint32_t
pai_pai_encoded_size(const pai_pai_builder_t *builder) {
  uint64_t total;
  uint32_t i;

  if (builder == NULL) {
    return 0;
  }
  total = PAI_PAI_HEADER_SIZE + (uint64_t)builder->num_sections * 16u;
  for (i = 0; i < builder->num_sections; i++) {
    total += builder->sections[i].size;
  }
  if (total > UINT32_MAX) {
    return 0;
  }
  return (uint32_t)total;
}

pai_status_t
pai_pai_build(const pai_pai_builder_t *builder, uint8_t *out, uint32_t cap,
              uint32_t *out_nbytes) {
  uint32_t total;
  uint32_t cursor;
  uint32_t i;
  uint32_t crc;

  if (builder == NULL || out == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (builder->num_sections == 0) {
    return PAI_ERR_INVALID_ARG; /* an empty container is never valid */
  }
  total = pai_pai_encoded_size(builder);
  if (total == 0 || total > cap) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(out, 0, PAI_PAI_HEADER_SIZE);
  /* magic, version, flags, num_sections */
  out[0] = (uint8_t)(PAI_PAI_MAGIC & 0xFF);
  out[1] = (uint8_t)((PAI_PAI_MAGIC >> 8) & 0xFF);
  out[2] = (uint8_t)((PAI_PAI_MAGIC >> 16) & 0xFF);
  out[3] = (uint8_t)((PAI_PAI_MAGIC >> 24) & 0xFF);
  out[4] = (uint8_t)PAI_PAI_VERSION;
  out[5] = (uint8_t)(PAI_PAI_VERSION >> 8);
  out[6] = 0; /* flags */
  out[7] = 0;
  out[8] = (uint8_t)builder->num_sections;
  out[9] = (uint8_t)(builder->num_sections >> 8);
  out[10] = (uint8_t)(builder->num_sections >> 16);
  out[11] = (uint8_t)(builder->num_sections >> 24);
  /* crc at [12..16) filled at the end */

  /* Section table. */
  cursor = PAI_PAI_HEADER_SIZE + builder->num_sections * 16u;
  for (i = 0; i < builder->num_sections; i++) {
    uint8_t *t = out + PAI_PAI_HEADER_SIZE + i * 16u;
    uint32_t type = builder->sections[i].type;
    uint32_t size = builder->sections[i].size;

    t[0] = (uint8_t)(type & 0xFF);
    t[1] = (uint8_t)((type >> 8) & 0xFF);
    t[2] = (uint8_t)((type >> 16) & 0xFF);
    t[3] = (uint8_t)((type >> 24) & 0xFF);
    t[4] = 0; /* flags */
    t[5] = 0;
    t[6] = 0;
    t[7] = 0;
    t[8] = (uint8_t)(cursor & 0xFF);
    t[9] = (uint8_t)((cursor >> 8) & 0xFF);
    t[10] = (uint8_t)((cursor >> 16) & 0xFF);
    t[11] = (uint8_t)((cursor >> 24) & 0xFF);
    t[12] = (uint8_t)(size & 0xFF);
    t[13] = (uint8_t)((size >> 8) & 0xFF);
    t[14] = (uint8_t)((size >> 16) & 0xFF);
    t[15] = (uint8_t)((size >> 24) & 0xFF);

    if (size > 0) {
      memcpy(out + cursor, builder->sections[i].data, size);
    }
    cursor += size;
  }

  /* CRC over [0,12) + table + payloads. */
  crc = pai_proto_crc32_init();
  crc = pai_proto_crc32_upd(crc, out, 12);
  crc = pai_proto_crc32_upd(crc, out + PAI_PAI_HEADER_SIZE,
                            total - PAI_PAI_HEADER_SIZE);
  crc = pai_proto_crc32_fin(crc);
  out[12] = (uint8_t)(crc & 0xFF);
  out[13] = (uint8_t)((crc >> 8) & 0xFF);
  out[14] = (uint8_t)((crc >> 16) & 0xFF);
  out[15] = (uint8_t)((crc >> 24) & 0xFF);

  *out_nbytes = total;
  return PAI_OK;
}

pai_status_t
pai_pai_open(const uint8_t *data, uint32_t nbytes, pai_pai_container_t *out) {
  uint32_t magic;
  uint32_t version;
  uint32_t num_sections;
  uint32_t expected_crc;
  uint32_t crc;
  uint32_t table_bytes;
  uint32_t i;

  if (data == NULL || out == NULL || nbytes < PAI_PAI_HEADER_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }

  magic = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
          ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
  version = (uint32_t)data[4] | ((uint32_t)data[5] << 8);
  if (magic != PAI_PAI_MAGIC) {
    return PAI_ERR_PROTOCOL;
  }
  if (version != PAI_PAI_VERSION || data[6] != 0 || data[7] != 0) {
    return PAI_ERR_PROTOCOL;
  }
  num_sections = (uint32_t)data[8] | ((uint32_t)data[9] << 8) |
                 ((uint32_t)data[10] << 16) | ((uint32_t)data[11] << 24);
  if (num_sections == 0 || num_sections > PAI_PAI_MAX_SECTIONS) {
    return PAI_ERR_PROTOCOL;
  }
  expected_crc = (uint32_t)data[12] | ((uint32_t)data[13] << 8) |
                 ((uint32_t)data[14] << 16) | ((uint32_t)data[15] << 24);

  table_bytes = num_sections * 16u;
  if ((uint64_t)PAI_PAI_HEADER_SIZE + table_bytes > nbytes) {
    return PAI_ERR_PROTOCOL;
  }

  /* Validate table bounds + flags before trusting anything. */
  for (i = 0; i < num_sections; i++) {
    const uint8_t *t = data + PAI_PAI_HEADER_SIZE + i * 16u;
    uint32_t offset = (uint32_t)t[8] | ((uint32_t)t[9] << 8) |
                      ((uint32_t)t[10] << 16) | ((uint32_t)t[11] << 24);
    uint32_t size = (uint32_t)t[12] | ((uint32_t)t[13] << 8) |
                    ((uint32_t)t[14] << 16) | ((uint32_t)t[15] << 24);
    uint32_t type = (uint32_t)t[0] | ((uint32_t)t[1] << 8) |
                    ((uint32_t)t[2] << 16) | ((uint32_t)t[3] << 24);

    if (t[4] != 0 || t[5] != 0 || t[6] != 0 || t[7] != 0) {
      return PAI_ERR_PROTOCOL;
    }
    if (type == 0 || type > PAI_PAI_SEC_PROFILE) {
      return PAI_ERR_PROTOCOL;
    }
    /* Section payloads must not overlap the table or each other, and a
     * section type may appear at most once (pai_pai_section returns the
     * first match, so duplicates would be ambiguous). */
    if (offset < PAI_PAI_HEADER_SIZE + table_bytes) {
      return PAI_ERR_PROTOCOL;
    }
    if ((uint64_t)offset + size > nbytes) {
      return PAI_ERR_PROTOCOL;
    }
    for (uint32_t k = 0; k < i; k++) {
      const uint8_t *prev = data + PAI_PAI_HEADER_SIZE + k * 16u;
      uint32_t prev_type = (uint32_t)prev[0] | ((uint32_t)prev[1] << 8) |
                           ((uint32_t)prev[2] << 16) |
                           ((uint32_t)prev[3] << 24);
      uint32_t prev_off = (uint32_t)prev[8] | ((uint32_t)prev[9] << 8) |
                          ((uint32_t)prev[10] << 16) |
                          ((uint32_t)prev[11] << 24);
      uint32_t prev_size = (uint32_t)prev[12] | ((uint32_t)prev[13] << 8) |
                           ((uint32_t)prev[14] << 16) |
                           ((uint32_t)prev[15] << 24);

      if (prev_type == type) {
        return PAI_ERR_PROTOCOL;
      }
      if (offset < prev_off + prev_size && prev_off < offset + size) {
        return PAI_ERR_PROTOCOL;
      }
    }
  }

  /* Whole-file CRC. */
  crc = pai_proto_crc32_init();
  crc = pai_proto_crc32_upd(crc, data, 12);
  crc = pai_proto_crc32_upd(crc, data + PAI_PAI_HEADER_SIZE,
                            nbytes - PAI_PAI_HEADER_SIZE);
  crc = pai_proto_crc32_fin(crc);
  if (crc != expected_crc) {
    return PAI_ERR_PROTOCOL;
  }

  memset(out, 0, sizeof(*out));
  out->version = version;
  out->nbytes = nbytes;
  out->data = data;
  out->num_sections = num_sections;
  for (i = 0; i < num_sections; i++) {
    const uint8_t *t = data + PAI_PAI_HEADER_SIZE + i * 16u;
    out->sections[i].type = (uint32_t)t[0] | ((uint32_t)t[1] << 8) |
                            ((uint32_t)t[2] << 16) | ((uint32_t)t[3] << 24);
    out->sections[i].flags = 0;
    out->sections[i].offset = (uint32_t)t[8] | ((uint32_t)t[9] << 8) |
                              ((uint32_t)t[10] << 16) |
                              ((uint32_t)t[11] << 24);
    out->sections[i].size = (uint32_t)t[12] | ((uint32_t)t[13] << 8) |
                            ((uint32_t)t[14] << 16) |
                            ((uint32_t)t[15] << 24);
  }
  return PAI_OK;
}

const uint8_t *
pai_pai_section(const pai_pai_container_t *container, uint32_t type,
                uint32_t *out_size) {
  uint32_t i;

  if (container == NULL) {
    return NULL;
  }
  for (i = 0; i < container->num_sections; i++) {
    if (container->sections[i].type == type) {
      if (out_size != NULL) {
        *out_size = container->sections[i].size;
      }
      return container->data + container->sections[i].offset;
    }
  }
  if (out_size != NULL) {
    *out_size = 0;
  }
  return NULL;
}

/* ------------------------------------------------------------------ */
/* Meta codec                                                          */
/* ------------------------------------------------------------------ */

pai_status_t
pai_pai_meta_encode(const pai_pai_meta_t *meta, uint8_t *out, uint32_t cap,
                    uint32_t *out_nbytes) {
  uint32_t name_len;
  uint32_t p;

  if (meta == NULL || out == NULL || out_nbytes == NULL || cap < PAI_PAI_META_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }
  name_len = (uint32_t)strlen(meta->name);
  if (name_len > PAI_PAI_NAME_MAX) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(out, 0, PAI_PAI_META_SIZE);
  p = 0;
  out[p++] = (uint8_t)(name_len & 0xFF);
  out[p++] = (uint8_t)(name_len >> 8);
  memcpy(out + p, meta->name, name_len);
  p += name_len;

  /* Pad name field to PAI_PAI_NAME_MAX+1 chars (0-indexed NUL-filled),
   * then fixed u32 fields. */
  {
    uint32_t vals[6];
    uint32_t i;
    vals[0] = meta->family;
    vals[1] = meta->context_len;
    vals[2] = meta->num_layers;
    vals[3] = meta->kv_bytes_per_token;
    vals[4] = meta->vocab_size;
    vals[5] = meta->reserved;
    p = 2 + PAI_PAI_NAME_MAX + 1;
    for (i = 0; i < 6; i++) {
      uint32_t v = vals[i];
      out[p++] = (uint8_t)(v & 0xFF);
      out[p++] = (uint8_t)((v >> 8) & 0xFF);
      out[p++] = (uint8_t)((v >> 16) & 0xFF);
      out[p++] = (uint8_t)((v >> 24) & 0xFF);
    }
  }

  *out_nbytes = PAI_PAI_META_SIZE;
  return PAI_OK;
}

pai_status_t
pai_pai_meta_decode(const uint8_t *data, uint32_t size, pai_pai_meta_t *out) {
  uint32_t name_len;
  uint32_t p;

  if (data == NULL || out == NULL || size < PAI_PAI_META_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));
  name_len = (uint32_t)data[0] | ((uint32_t)data[1] << 8);
  if (name_len > PAI_PAI_NAME_MAX) {
    return PAI_ERR_PROTOCOL;
  }
  memcpy(out->name, data + 2, name_len);
  out->name[name_len] = '\0';

  p = 2 + PAI_PAI_NAME_MAX + 1;
  out->family = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                ((uint32_t)data[p + 2] << 16) | ((uint32_t)data[p + 3] << 24);
  p += 4;
  out->context_len = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                     ((uint32_t)data[p + 2] << 16) |
                     ((uint32_t)data[p + 3] << 24);
  p += 4;
  out->num_layers = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                    ((uint32_t)data[p + 2] << 16) |
                    ((uint32_t)data[p + 3] << 24);
  p += 4;
  out->kv_bytes_per_token = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                            ((uint32_t)data[p + 2] << 16) |
                            ((uint32_t)data[p + 3] << 24);
  p += 4;
  out->vocab_size = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                    ((uint32_t)data[p + 2] << 16) |
                    ((uint32_t)data[p + 3] << 24);
  p += 4;
  out->reserved = (uint32_t)data[p] | ((uint32_t)data[p + 1] << 8) |
                  ((uint32_t)data[p + 2] << 16) |
                  ((uint32_t)data[p + 3] << 24);

  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Manifest codec                                                      */
/* ------------------------------------------------------------------ */

static void
put_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    p[i] = (uint8_t)(v >> (8 * i));
  }
}

static uint64_t
get_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; i--) {
    v = (v << 8) | p[i];
  }
  return v;
}

pai_status_t
pai_pai_manifest_encode(const pai_pai_tensor_t *tensors, uint32_t count,
                        uint8_t *out, uint32_t cap, uint32_t *out_nbytes) {
  uint64_t total = 4u;
  uint32_t i;

  if (tensors == NULL || count == 0 || out == NULL || out_nbytes == NULL ||
      count > PAI_PAI_MAX_TENSORS) {
    return PAI_ERR_INVALID_ARG;
  }
  /* Per entry: name_len(2) + name + value_id(4) + dtype/rank(4) +
   * shape(6*8) + offset(8) + size(8). */
  for (i = 0; i < count; i++) {
    total += 2u + (uint32_t)strlen(tensors[i].name) + 4u + 4u + 6u * 8u +
             8u + 8u;
  }
  if (total > cap) {
    return PAI_ERR_INVALID_ARG;
  }

  out[0] = (uint8_t)(count & 0xFF);
  out[1] = (uint8_t)((count >> 8) & 0xFF);
  out[2] = (uint8_t)((count >> 16) & 0xFF);
  out[3] = (uint8_t)((count >> 24) & 0xFF);
  {
    uint8_t *p = out + 4;
    for (i = 0; i < count; i++) {
      const pai_pai_tensor_t *t = &tensors[i];
      uint32_t name_len = (uint32_t)strlen(t->name);
      uint32_t d;

      p[0] = (uint8_t)(name_len & 0xFF);
      p[1] = (uint8_t)(name_len >> 8);
      p += 2;
      memcpy(p, t->name, name_len);
      p += name_len;
      p[0] = (uint8_t)(t->value_id & 0xFF);
      p[1] = (uint8_t)((t->value_id >> 8) & 0xFF);
      p[2] = (uint8_t)((t->value_id >> 16) & 0xFF);
      p[3] = (uint8_t)((t->value_id >> 24) & 0xFF);
      p += 4;
      p[0] = (uint8_t)t->dtype;
      p[1] = (uint8_t)t->rank;
      p[2] = 0;
      p[3] = 0;
      p += 4;
      for (d = 0; d < PAI_TENSOR_MAX_RANK; d++) {
        put_le64(p, t->shape[d]);
        p += 8;
      }
      put_le64(p, t->offset);
      p += 8;
      put_le64(p, t->size_bytes);
      p += 8;
    }
  }

  *out_nbytes = (uint32_t)total;
  return PAI_OK;
}

pai_status_t
pai_pai_manifest_decode(const uint8_t *data, uint32_t size,
                        pai_pai_tensor_t *out_tensors, uint32_t max_tensors,
                        uint32_t *out_count) {
  uint32_t count;
  const uint8_t *p;
  uint32_t i;

  if (data == NULL || size < 4 || out_tensors == NULL ||
      max_tensors == 0 || out_count == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  count = (uint32_t)data[0] | ((uint32_t)data[1] << 8) |
          ((uint32_t)data[2] << 16) | ((uint32_t)data[3] << 24);
  if (count == 0 || count > PAI_PAI_MAX_TENSORS || count > max_tensors) {
    return PAI_ERR_PROTOCOL;
  }

  p = data + 4;
  for (i = 0; i < count; i++) {
    pai_pai_tensor_t *t = &out_tensors[i];
    uint32_t name_len;
    uint32_t d;

    if ((uint64_t)(p - data) + 2 > size) {
      return PAI_ERR_PROTOCOL;
    }
    name_len = (uint32_t)p[0] | ((uint32_t)p[1] << 8);
    p += 2;
    /* The guard below must cover name + value_id(4) + dtype/rank(4) +
     * shape(6*8) + offset(8) + size(8). */
    if (name_len == 0 || name_len >= sizeof(t->name) ||
        (uint64_t)(p - data) + name_len + 72 > size) {
      return PAI_ERR_PROTOCOL;
    }
    if (name_len == 0 || name_len >= sizeof(t->name) ||
        (uint64_t)(p - data) + name_len + 4u + 4u + 6u * 8u + 8u + 8u >
            size) {
      return PAI_ERR_PROTOCOL;
    }
    memset(t, 0, sizeof(*t));
    memcpy(t->name, p, name_len);
    p += name_len;
    t->value_id = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                  ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    p += 4;
    t->dtype = (pai_dtype_t)p[0];
    t->rank = p[1];
    p += 4;
    if (t->rank == 0 || t->rank > PAI_TENSOR_MAX_RANK ||
        t->dtype >= PAI_DTYPE_COUNT) {
      return PAI_ERR_PROTOCOL;
    }
    for (d = 0; d < PAI_TENSOR_MAX_RANK; d++) {
      t->shape[d] = get_le64(p);
      p += 8;
    }
    t->offset = get_le64(p);
    p += 8;
    t->size_bytes = get_le64(p);
    p += 8;
    if (t->size_bytes == 0) {
      return PAI_ERR_PROTOCOL;
    }
  }

  *out_count = count;
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* File helpers                                                        */
/* ------------------------------------------------------------------ */

pai_status_t
pai_pai_write_file(const char *path, const uint8_t *blob, uint32_t nbytes) {
  FILE *f;

  if (path == NULL || blob == NULL || nbytes == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  f = fopen(path, "wb");
  if (f == NULL) {
    return PAI_ERR_IO;
  }
  if (fwrite(blob, 1, nbytes, f) != nbytes) {
    fclose(f);
    return PAI_ERR_IO;
  }
  if (fclose(f) != 0) {
    return PAI_ERR_IO;
  }
  return PAI_OK;
}

pai_status_t
pai_pai_read_file(const char *path, uint8_t **out, uint32_t *out_nbytes) {
  FILE *f;
  long sz;
  uint8_t *buf;

  if (path == NULL || out == NULL || out_nbytes == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  f = fopen(path, "rb");
  if (f == NULL) {
    return PAI_ERR_IO;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return PAI_ERR_IO;
  }
  sz = ftell(f);
  if (sz <= 0 || (unsigned long)sz > UINT32_MAX) {
    fclose(f);
    return PAI_ERR_IO;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return PAI_ERR_IO;
  }
  buf = (uint8_t *)malloc((size_t)sz);
  if (buf == NULL) {
    fclose(f);
    return PAI_ERR_NOMEM;
  }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return PAI_ERR_IO;
  }
  if (fclose(f) != 0) {
    free(buf);
    return PAI_ERR_IO;
  }
  *out = buf;
  *out_nbytes = (uint32_t)sz;
  return PAI_OK;
}
