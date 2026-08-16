/* ProsperoAI — GGUF reader (see gguf.h). Dequantization formulas for
 * Q4_0..Q6_K are ported verbatim from llama.cpp ggml-quants.c (scalar
 * reference dequantizers); block layouts match ggml-common.h. */

#include "gguf.h"

#include <stdlib.h>
#include <string.h>

typedef struct gguf_cursor {
  const uint8_t *p;
  const uint8_t *end;
} gguf_cursor_t;

static int
cur_ok(const gguf_cursor_t *c, uint64_t need) {
  return c->p != NULL && (uint64_t)(uintptr_t)(c->end - c->p) >= need;
}

static uint64_t
cur_u64(gguf_cursor_t *c) {
  const uint8_t *p = c->p;
  uint64_t v = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) |
               ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) |
               ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) |
               ((uint64_t)p[7] << 56);
  c->p += 8;
  return v;
}

static uint32_t
cur_u32(gguf_cursor_t *c) {
  const uint8_t *p = c->p;
  uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
               ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  c->p += 4;
  return v;
}

/* Point at a string's bytes (not NUL-terminated); NULL on overrun. */
static const char *
read_string(gguf_cursor_t *c, uint64_t *out_len) {
  uint64_t len;
  const uint8_t *s;

  if (!cur_ok(c, 8)) {
    return NULL;
  }
  len = cur_u64(c);
  if (!cur_ok(c, len)) {
    return NULL;
  }
  s = c->p;
  c->p += len;
  *out_len = len;
  return (const char *)s;
}

/* Skip one KV value of the given GGUF type. 0 on overrun/unknown. */
static int
skip_value(gguf_cursor_t *c, uint32_t type) {
  switch (type) {
    case 0:  /* u8   */
    case 1:  /* i8   */
    case 7:  /* bool */
      if (!cur_ok(c, 1)) {
        return 0;
      }
      c->p += 1;
      return 1;
    case 2:  /* u16  */
    case 3:  /* i16  */
      if (!cur_ok(c, 2)) {
        return 0;
      }
      c->p += 2;
      return 1;
    case 4:  /* u32  */
    case 5:  /* i32  */
    case 6:  /* f32  */
      if (!cur_ok(c, 4)) {
        return 0;
      }
      c->p += 4;
      return 1;
    case 10: /* u64  */
    case 11: /* i64  */
    case 12: /* f64  */
      if (!cur_ok(c, 8)) {
        return 0;
      }
      c->p += 8;
      return 1;
    case 8: { /* string */
      uint64_t len;
      if (!cur_ok(c, 8)) {
        return 0;
      }
      len = cur_u64(c);
      if (!cur_ok(c, len)) {
        return 0;
      }
      c->p += len;
      return 1;
    }
    case 9: { /* array */
      uint32_t et;
      uint64_t count;
      uint64_t i;
      if (!cur_ok(c, 12)) {
        return 0;
      }
      et = cur_u32(c);
      count = cur_u64(c);
      for (i = 0; i < count; i++) {
        if (!skip_value(c, et)) {
          return 0;
        }
      }
      return 1;
    }
    default:
      return 0;
  }
}

pai_status_t
pai_gguf_open(const uint8_t *data, uint64_t size, pai_gguf_t *out) {
  gguf_cursor_t c;
  uint64_t kv_count;
  uint64_t after;
  uint64_t i;
  uint32_t alignment;

  if (data == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));
  if (size < 24) {
    return PAI_ERR_PROTOCOL;
  }

  c.p = data;
  c.end = data + size;
  if (cur_u32(&c) != PAI_GGUF_MAGIC) {
    return PAI_ERR_PROTOCOL;
  }
  out->version = cur_u32(&c);
  if (out->version < 1 || out->version > PAI_GGUF_MAX_VERSION) {
    return PAI_ERR_UNSUPPORTED;
  }
  out->tensor_count = cur_u64(&c);
  kv_count = cur_u64(&c);
  if (out->tensor_count > PAI_GGUF_MAX_TENSORS) {
    return PAI_ERR_UNSUPPORTED;
  }

  /* Metadata KV section. */
  out->kv_count = kv_count;
  out->kv_start = c.p;
  for (i = 0; i < kv_count; i++) {
    uint64_t key_len;
    uint32_t vtype;
    if (read_string(&c, &key_len) == NULL || !cur_ok(&c, 4)) {
      return PAI_ERR_PROTOCOL;
    }
    vtype = cur_u32(&c);
    if (!skip_value(&c, vtype)) {
      return PAI_ERR_PROTOCOL;
    }
  }

  /* Tensor infos. */
  out->tensors =
      (pai_gguf_tensor_t *)calloc(out->tensor_count > 0 ? out->tensor_count : 1,
                                  sizeof(pai_gguf_tensor_t));
  if (out->tensors == NULL) {
    return PAI_ERR_NOMEM;
  }
  for (i = 0; i < out->tensor_count; i++) {
    pai_gguf_tensor_t *t = &out->tensors[i];
    uint64_t len;
    uint32_t n_dims;
    uint32_t d;

    if (read_string(&c, &len) == NULL || len == 0 ||
        len >= sizeof(t->name)) {
      goto proto;
    }
    memcpy(t->name, c.p - len, (size_t)len);
    t->name[len] = '\0';
    if (!cur_ok(&c, 4)) {
      goto proto;
    }
    n_dims = cur_u32(&c);
    if (n_dims == 0 || n_dims > 4) {
      goto proto;
    }
    t->n_dims = n_dims;
    for (d = 0; d < n_dims; d++) {
      if (!cur_ok(&c, 8)) {
        goto proto;
      }
      t->dims[d] = cur_u64(&c);
      if (t->dims[d] > PAI_GGUF_MAX_DIM) {
        goto proto; /* hostile extent: would wrap numel/size math */
      }
    }
    if (!cur_ok(&c, 4)) {
      goto proto;
    }
    t->type = cur_u32(&c);
    if (!cur_ok(&c, 8)) {
      goto proto;
    }
    t->offset = cur_u64(&c);
  }

  out->data = data;
  out->size = size;

  /* Alignment may be overridden by metadata. */
  alignment = PAI_GGUF_ALIGN_DEFAULT;
  (void)pai_gguf_get_u32(out, "general.alignment", &alignment);
  if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      alignment > 4096) {
    goto proto;
  }
  out->alignment = alignment;

  after = (uint64_t)(uintptr_t)(c.p - data);
  {
    uint64_t aligned = (after + alignment - 1) / alignment * alignment;
    if (aligned > size) {
      goto proto;
    }
    out->data_start = data + aligned;
  }
  return PAI_OK;

proto:
  free(out->tensors);
  out->tensors = NULL;
  return PAI_ERR_PROTOCOL;
}

void
pai_gguf_close(pai_gguf_t *f) {
  if (f == NULL) {
    return;
  }
  free(f->tensors);
  f->tensors = NULL;
}

/* Locate a KV value by exact key; returns the value bytes and type. */
static const uint8_t *
find_kv(const pai_gguf_t *f, const char *key, uint32_t *out_type) {
  gguf_cursor_t c;
  uint64_t i;
  size_t key_len = strlen(key);

  if (f == NULL || key == NULL || f->kv_start == NULL) {
    return NULL;
  }
  c.p = f->kv_start;
  c.end = f->data + f->size;
  for (i = 0; i < f->kv_count; i++) {
    uint64_t len;
    const char *k = read_string(&c, &len);
    uint32_t vtype;
    const uint8_t *v;
    if (k == NULL || !cur_ok(&c, 4)) {
      return NULL;
    }
    vtype = cur_u32(&c);
    v = c.p;
    if (len == key_len && memcmp(k, key, len) == 0) {
      *out_type = vtype;
      return v;
    }
    if (!skip_value(&c, vtype)) {
      return NULL;
    }
  }
  return NULL;
}

/* Read a little-endian u64 from a value buffer of the given size. */
static int
le_bytes_to_u64(const uint8_t *p, uint32_t n, uint64_t *out) {
  uint64_t v = 0;
  uint32_t i;
  if (n == 0 || n > 8) {
    return 0;
  }
  for (i = 0; i < n; i++) {
    v |= (uint64_t)p[i] << (8 * i);
  }
  return 1;
}

pai_status_t
pai_gguf_get_u32(const pai_gguf_t *f, const char *key, uint32_t *out) {
  const uint8_t *v;
  uint32_t type;
  uint64_t raw;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  v = find_kv(f, key, &type);
  if (v == NULL) {
    return PAI_ERR_MISMATCH;
  }
  switch (type) {
    case 0: /* u8  */
    case 1: /* i8  */
    case 7: /* bool */
      raw = v[0];
      break;
    case 2: /* u16 */
    case 3: /* i16 */
      raw = (uint64_t)v[0] | ((uint64_t)v[1] << 8);
      break;
    case 4: /* u32 */
    case 5: /* i32 */
    case 6: /* f32 bits */
      raw = (uint64_t)v[0] | ((uint64_t)v[1] << 8) | ((uint64_t)v[2] << 16) |
            ((uint64_t)v[3] << 24);
      break;
    case 10: /* u64 */
    case 11: /* i64 */
    case 12: /* f64 bits */
      if (!le_bytes_to_u64(v, 8, &raw)) {
        return PAI_ERR_PROTOCOL;
      }
      break;
    default:
      return PAI_ERR_MISMATCH;
  }
  *out = (uint32_t)raw;
  return PAI_OK;
}

pai_status_t
pai_gguf_get_f32(const pai_gguf_t *f, const char *key, float *out) {
  const uint8_t *v;
  uint32_t type;
  uint32_t u;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  v = find_kv(f, key, &type);
  if (v == NULL) {
    return PAI_ERR_MISMATCH;
  }
  switch (type) {
    case 6: { /* f32 */
      uint32_t bits = (uint32_t)v[0] | ((uint32_t)v[1] << 8) |
                      ((uint32_t)v[2] << 16) | ((uint32_t)v[3] << 24);
      memcpy(out, &bits, 4);
      return PAI_OK;
    }
    case 12: { /* f64 */
      double d;
      uint64_t bits;
      if (!le_bytes_to_u64(v, 8, &bits)) {
        return PAI_ERR_PROTOCOL;
      }
      memcpy(&d, &bits, 8);
      *out = (float)d;
      return PAI_OK;
    }
    default: {
      pai_status_t st = pai_gguf_get_u32(f, key, &u);
      if (st == PAI_OK) {
        *out = (float)u;
      }
      return st;
    }
  }
}

pai_status_t
pai_gguf_get_string(const pai_gguf_t *f, const char *key, char *out,
                    uint64_t cap) {
  const uint8_t *v;
  uint32_t type;
  uint64_t len;

  if (out == NULL || cap == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  v = find_kv(f, key, &type);
  if (v == NULL) {
    return PAI_ERR_MISMATCH;
  }
  if (type != 8) {
    return PAI_ERR_MISMATCH;
  }
  len = (uint64_t)v[0] | ((uint64_t)v[1] << 8) | ((uint64_t)v[2] << 16) |
        ((uint64_t)v[3] << 24) | ((uint64_t)v[4] << 32) |
        ((uint64_t)v[5] << 40) | ((uint64_t)v[6] << 48) |
        ((uint64_t)v[7] << 56);
  if (len >= cap) {
    return PAI_ERR_NOMEM;
  }
  memcpy(out, v + 8, (size_t)len);
  out[len] = '\0';
  return PAI_OK;
}

/* Parse an array value: returns element type, count and the element
 * data start. */
static const uint8_t *
find_array(const pai_gguf_t *f, const char *key, uint32_t *elem_type,
           uint64_t *count) {
  const uint8_t *v;
  uint32_t type;

  v = find_kv(f, key, &type);
  if (v == NULL || type != 9) {
    return NULL;
  }
  *elem_type = (uint32_t)v[0] | ((uint32_t)v[1] << 8) |
               ((uint32_t)v[2] << 16) | ((uint32_t)v[3] << 24);
  *count = (uint64_t)v[4] | ((uint64_t)v[5] << 8) | ((uint64_t)v[6] << 16) |
           ((uint64_t)v[7] << 24) | ((uint64_t)v[8] << 32) |
           ((uint64_t)v[9] << 40) | ((uint64_t)v[10] << 48) |
           ((uint64_t)v[11] << 56);
  return v + 12;
}

pai_status_t
pai_gguf_get_string_array(const pai_gguf_t *f, const char *key, char ***out,
                          uint64_t *out_n) {
  const uint8_t *p;
  uint32_t elem_type;
  uint64_t count;
  uint64_t i;
  char **arr;
  size_t blob = 0;
  char *blobp;

  if (out == NULL || out_n == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out = NULL;
  *out_n = 0;
  p = find_array(f, key, &elem_type, &count);
  if (p == NULL || elem_type != 8) {
    return PAI_ERR_MISMATCH;
  }
  if (count > (1u << 20)) {
    return PAI_ERR_UNSUPPORTED;
  }

  /* Layout: [count x char*][payload]; strings live in the payload, so
   * the caller frees the returned block once. */
  for (i = 0; i < count; i++) {
    uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                   ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                   ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                   ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
    if (len > (uint64_t)(f->data + f->size - p - 8)) {
      return PAI_ERR_PROTOCOL;
    }
    blob += (size_t)len + 1;
    p += 8 + len;
  }

  arr = (char **)malloc((size_t)count * sizeof(char *) + (blob > 0 ? blob : 1));
  if (arr == NULL) {
    return PAI_ERR_NOMEM;
  }
  blobp = (char *)(arr + count);
  p = find_array(f, key, &elem_type, &count);
  for (i = 0; i < count; i++) {
    uint64_t len = (uint64_t)p[0] | ((uint64_t)p[1] << 8) |
                   ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) |
                   ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) |
                   ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
    memcpy(blobp, p + 8, (size_t)len);
    blobp[len] = '\0';
    arr[i] = blobp;
    blobp += len + 1;
    p += 8 + len;
  }
  *out = arr;
  *out_n = count;
  return PAI_OK;
}

pai_status_t
pai_gguf_get_f32_array(const pai_gguf_t *f, const char *key, float **out,
                       uint64_t *out_n) {
  const uint8_t *p;
  uint32_t elem_type;
  uint64_t count;
  uint64_t i;
  float *arr;

  if (out == NULL || out_n == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out = NULL;
  *out_n = 0;
  p = find_array(f, key, &elem_type, &count);
  if (p == NULL || elem_type != 6) {
    return PAI_ERR_MISMATCH;
  }
  if (count > (1u << 20) ||
      count * 4 > (uint64_t)(f->data + f->size - p)) {
    return PAI_ERR_PROTOCOL;
  }
  arr = (float *)malloc((size_t)count * sizeof(float));
  if (arr == NULL) {
    return PAI_ERR_NOMEM;
  }
  for (i = 0; i < count; i++) {
    uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    memcpy(&arr[i], &bits, 4);
    p += 4;
  }
  *out = arr;
  *out_n = count;
  return PAI_OK;
}

const pai_gguf_tensor_t *
pai_gguf_find_tensor(const pai_gguf_t *f, const char *name) {
  uint64_t i;

  if (f == NULL || name == NULL || f->tensors == NULL) {
    return NULL; /* never deref a failed-open file view */
  }
  for (i = 0; i < f->tensor_count; i++) {
    if (strcmp(f->tensors[i].name, name) == 0) {
      return &f->tensors[i];
    }
  }
  return NULL;
}

uint64_t
pai_gguf_tensor_numel(const pai_gguf_tensor_t *t) {
  uint64_t n = 1;
  uint32_t i;

  if (t == NULL || t->n_dims == 0) {
    return 0;
  }
  for (i = 0; i < t->n_dims; i++) {
    n *= t->dims[i];
  }
  return n;
}

uint64_t
pai_gguf_type_bytes(uint32_t type, uint64_t numel) {
  switch (type) {
    case PAI_GGUF_F32:
      return numel * 4;
    case PAI_GGUF_F16:
    case PAI_GGUF_BF16:
      return numel * 2;
    case PAI_GGUF_Q4_0:
      return numel % 32 == 0 ? (numel / 32) * 18 : 0;
    case PAI_GGUF_Q4_1:
      return numel % 32 == 0 ? (numel / 32) * 20 : 0;
    case PAI_GGUF_Q5_0:
      return numel % 32 == 0 ? (numel / 32) * 22 : 0;
    case PAI_GGUF_Q5_1:
      return numel % 32 == 0 ? (numel / 32) * 24 : 0;
    case PAI_GGUF_Q8_0:
      return numel % 32 == 0 ? (numel / 32) * 34 : 0;
    case PAI_GGUF_Q8_1:
      return numel % 32 == 0 ? (numel / 32) * 36 : 0;
    case PAI_GGUF_Q2_K:
      return numel % 256 == 0 ? (numel / 256) * 84 : 0;
    case PAI_GGUF_Q3_K:
      return numel % 256 == 0 ? (numel / 256) * 110 : 0;
    case PAI_GGUF_Q4_K:
      return numel % 256 == 0 ? (numel / 256) * 144 : 0;
    case PAI_GGUF_Q5_K:
      return numel % 256 == 0 ? (numel / 256) * 176 : 0;
    case PAI_GGUF_Q6_K:
      return numel % 256 == 0 ? (numel / 256) * 210 : 0;
    case PAI_GGUF_Q8_K:
      return numel % 256 == 0 ? (numel / 256) * 292 : 0;
    default:
      return 0;
  }
}

/* Dequantization */

static float
f16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t v;

  if (exp == 0) {
    if (mant == 0) {
      v = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3FFu;
      v = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    v = sign | 0x7F800000u | (mant << 13);
  } else {
    v = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  {
    float f;
    memcpy(&f, &v, 4);
    return f;
  }
}

static float
le_f16(const uint8_t *p) {
  return f16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

/* Bit-unpacking helper for the K-quant scale fields (verbatim port of
 * get_scale_min_k4 from llama.cpp). */
static void
get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
    *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

static pai_status_t
dequant_block(uint32_t type, const uint8_t *src, float *dst,
              uint64_t numel) {
  uint64_t nb;

  switch (type) {
    case PAI_GGUF_F32: {
      const uint8_t *p = src;
      for (uint64_t i = 0; i < numel; i++, p += 4) {
        uint32_t bits = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        memcpy(&dst[i], &bits, 4);
      }
      return PAI_OK;
    }
    case PAI_GGUF_F16: {
      const uint8_t *p = src;
      for (uint64_t i = 0; i < numel; i++, p += 2) {
        dst[i] = le_f16(p);
      }
      return PAI_OK;
    }
    case PAI_GGUF_BF16: {
      const uint8_t *p = src;
      for (uint64_t i = 0; i < numel; i++, p += 2) {
        uint32_t bits = (uint32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8)) << 16;
        memcpy(&dst[i], &bits, 4);
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q4_0: /* 32 elems: half d, int8 qs[16] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 16; j++) {
          int x0 = (qs[j] & 0x0F) - 8;
          int x1 = (qs[j] >> 4) - 8;
          dst[j] = (float)x0 * d;
          dst[j + 16] = (float)x1 * d;
        }
        src += 18;
        dst += 32;
      }
      return PAI_OK;
    case PAI_GGUF_Q4_1: /* half d, half m, int8 qs[16] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        float m = le_f16(src + 2);
        const uint8_t *qs = src + 4;
        for (int j = 0; j < 16; j++) {
          dst[j] = (float)(qs[j] & 0x0F) * d + m;
          dst[j + 16] = (float)(qs[j] >> 4) * d + m;
        }
        src += 20;
        dst += 32;
      }
      return PAI_OK;
    case PAI_GGUF_Q5_0: { /* half d, u32 qh, int8 qs[16] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        uint32_t qh = (uint32_t)src[2] | ((uint32_t)src[3] << 8) |
                      ((uint32_t)src[4] << 16) | ((uint32_t)src[5] << 24);
        const uint8_t *qs = src + 6;
        for (int j = 0; j < 16; j++) {
          uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
          uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
          int32_t x0 = (int32_t)((qs[j] & 0x0F) | xh_0) - 16;
          int32_t x1 = (int32_t)((qs[j] >> 4) | xh_1) - 16;
          dst[j] = (float)x0 * d;
          dst[j + 16] = (float)x1 * d;
        }
        src += 22;
        dst += 32;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q5_1: { /* half d, half m, u32 qh, int8 qs[16] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        float m = le_f16(src + 2);
        uint32_t qh = (uint32_t)src[4] | ((uint32_t)src[5] << 8) |
                      ((uint32_t)src[6] << 16) | ((uint32_t)src[7] << 24);
        const uint8_t *qs = src + 8;
        for (int j = 0; j < 16; j++) {
          uint8_t xh_0 = (uint8_t)(((qh >> (j + 0)) << 4) & 0x10);
          uint8_t xh_1 = (uint8_t)((qh >> (j + 12)) & 0x10);
          int x0 = (qs[j] & 0x0F) | xh_0;
          int x1 = (qs[j] >> 4) | xh_1;
          dst[j] = (float)x0 * d + m;
          dst[j + 16] = (float)x1 * d + m;
        }
        src += 24;
        dst += 32;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q8_0: /* half d, i8 qs[32] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        const int8_t *qs = (const int8_t *)(src + 2);
        for (int j = 0; j < 32; j++) {
          dst[j] = (float)qs[j] * d;
        }
        src += 34;
        dst += 32;
      }
      return PAI_OK;
    case PAI_GGUF_Q8_1: /* half d, half s, i8 qs[32] */
      nb = numel / 32;
      for (uint64_t b = 0; b < nb; b++) {
        float d = le_f16(src);
        const int8_t *qs = (const int8_t *)(src + 4);
        for (int j = 0; j < 32; j++) {
          dst[j] = (float)qs[j] * d;
        }
        src += 36;
        dst += 32;
      }
      return PAI_OK;
    case PAI_GGUF_Q2_K: { /* u8 scales[16], u8 qs[64], half d, half dmin */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *sc = src;
        const uint8_t *q = src + 16;
        float d = le_f16(src + 80);
        float min = le_f16(src + 82);
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; j++) {
            uint8_t s = sc[is++];
            float dl = d * (float)(s & 0x0F);
            float ml = min * (float)(s >> 4);
            for (int l = 0; l < 16; l++) {
              *dst++ = dl * (float)(int8_t)((q[l] >> shift) & 3) - ml;
            }
            s = sc[is++];
            dl = d * (float)(s & 0x0F);
            ml = min * (float)(s >> 4);
            for (int l = 0; l < 16; l++) {
              *dst++ = dl * (float)(int8_t)((q[l + 16] >> shift) & 3) - ml;
            }
            shift += 2;
          }
          q += 32;
        }
        src += 84;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q3_K: { /* u8 hmask[32], u8 qs[64], u8 scales[12], half d */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *hm = src;
        const uint8_t *q = src + 32;
        const uint8_t *scs = src + 96;
        float d_all = le_f16(src + 108);
        uint32_t aux[4];
        uint32_t tmp;
        const int8_t *scales;
        uint8_t m = 1;
        int is = 0;

        memcpy(aux, scs, 12);
        tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & 0x0F0F0F0Fu) |
                 (((tmp >> 4) & 0x03030303u) << 4);
        aux[3] = ((aux[1] >> 4) & 0x0F0F0F0Fu) |
                 (((tmp >> 6) & 0x03030303u) << 4);
        aux[0] = (aux[0] & 0x0F0F0F0Fu) | (((tmp >> 0) & 0x03030303u) << 4);
        aux[1] = (aux[1] & 0x0F0F0F0Fu) | (((tmp >> 2) & 0x03030303u) << 4);
        scales = (const int8_t *)aux;

        for (int n = 0; n < 256; n += 128) {
          int shift = 0;
          for (int j = 0; j < 4; j++) {
            float dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
              *dst++ = dl *
                       ((float)(int8_t)((q[l] >> shift) & 3) -
                        (hm[l] & m ? 0.0f : 4.0f));
            }
            dl = d_all * (float)(scales[is++] - 32);
            for (int l = 0; l < 16; l++) {
              *dst++ = dl *
                       ((float)(int8_t)((q[l + 16] >> shift) & 3) -
                        (hm[l + 16] & m ? 0.0f : 4.0f));
            }
            shift += 2;
            m <<= 1;
          }
          q += 32;
        }
        src += 110;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q4_K: { /* half d, half dmin, u8 scales[12], u8 qs[128] */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *q = src + 16;
        const uint8_t *sc = src + 4;
        float d = le_f16(src);
        float min = le_f16(src + 2);
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
          uint8_t s, m;
          float d1, m1, d2, m2;
          get_scale_min_k4(is + 0, sc, &s, &m);
          d1 = d * (float)s;
          m1 = min * (float)m;
          get_scale_min_k4(is + 1, sc, &s, &m);
          d2 = d * (float)s;
          m2 = min * (float)m;
          for (int l = 0; l < 32; l++) {
            dst[l] = d1 * (float)(q[l] & 0x0F) - m1;
          }
          for (int l = 0; l < 32; l++) {
            dst[l + 32] = d2 * (float)(q[l] >> 4) - m2;
          }
          q += 32;
          is += 2;
          dst += 64;
        }
        src += 144;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q5_K: { /* half d, half dmin, scales[12], qh[32], qs[128] */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *ql = src + 48; /* qs[128]: low nibbles          */
        const uint8_t *qh = src + 16; /* qh[32]: high bit              */
        const uint8_t *sc = src + 4;
        float d = le_f16(src);
        float min = le_f16(src + 2);
        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;
        for (int j = 0; j < 256; j += 64) {
          uint8_t s, m;
          float d1, m1, d2, m2;
          get_scale_min_k4(is + 0, sc, &s, &m);
          d1 = d * (float)s;
          m1 = min * (float)m;
          get_scale_min_k4(is + 1, sc, &s, &m);
          d2 = d * (float)s;
          m2 = min * (float)m;
          for (int l = 0; l < 32; l++) {
            dst[l] = d1 * (float)((ql[l] & 0x0F) + (qh[l] & u1 ? 16 : 0)) - m1;
          }
          for (int l = 0; l < 32; l++) {
            dst[l + 32] =
                d2 * (float)((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
          }
          ql += 32;
          is += 2;
          u1 = (uint8_t)(u1 << 2);
          u2 = (uint8_t)(u2 << 2);
          dst += 64;
        }
        src += 176;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q6_K: { /* ql[128], qh[64], i8 scales[16], half d */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        const uint8_t *ql = src;
        const uint8_t *qh = src + 128;
        const int8_t *sc = (const int8_t *)(src + 192);
        float d = le_f16(src + 208);
        for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; l++) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            dst[l] = d * (float)sc[is + 0] * (float)q1;
            dst[l + 32] = d * (float)sc[is + 2] * (float)q2;
            dst[l + 64] = d * (float)sc[is + 4] * (float)q3;
            dst[l + 96] = d * (float)sc[is + 6] * (float)q4;
          }
          dst += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
        src += 210;
      }
      return PAI_OK;
    }
    case PAI_GGUF_Q8_K: { /* float d, i8 qs[256] */
      nb = numel / 256;
      for (uint64_t b = 0; b < nb; b++) {
        uint32_t bits;
        float d;
        const int8_t *qs = (const int8_t *)(src + 4);
        memcpy(&bits, src, 4);
        memcpy(&d, &bits, 4);
        for (int j = 0; j < 256; j++) {
          dst[j] = (float)qs[j] * d;
        }
        src += 292;
        dst += 256;
      }
      return PAI_OK;
    }
    default:
      return PAI_ERR_UNSUPPORTED;
  }
}

pai_status_t
pai_gguf_dequant(const pai_gguf_t *f, const pai_gguf_tensor_t *t,
                 float **out_f32, uint64_t *out_n) {
  uint64_t numel;
  uint64_t bytes;
  uint64_t region;
  uint64_t rel;
  float *dst;

  if (f == NULL || t == NULL || out_f32 == NULL || out_n == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_f32 = NULL;
  *out_n = 0;

  numel = pai_gguf_tensor_numel(t);
  if (numel == 0) {
    return PAI_ERR_PROTOCOL;
  }
  bytes = pai_gguf_type_bytes(t->type, numel);
  if (bytes == 0) {
    return PAI_ERR_UNSUPPORTED;
  }
  region = f->size - (uint64_t)(uintptr_t)(f->data_start - f->data);
  rel = t->offset;
  if (rel > region || bytes > region - rel) {
    return PAI_ERR_PROTOCOL; /* tensor bytes fall outside the file */
  }

  dst = (float *)malloc((size_t)numel * sizeof(float));
  if (dst == NULL) {
    return PAI_ERR_NOMEM;
  }
  {
    pai_status_t st = dequant_block(t->type, f->data_start + rel, dst, numel);
    if (st != PAI_OK) {
      free(dst);
      return st;
    }
  }
  *out_f32 = dst;
  *out_n = numel;
  return PAI_OK;
}
