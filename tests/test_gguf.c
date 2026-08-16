#include "test.h"

#include <gguf.h>
#include <llama.h>

#include <model.h>
#include <pai/pai.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal GGUF writer for test fixtures */

typedef struct gb {
  uint8_t *d;
  size_t n;
  size_t cap;
} gb_t;

static int
gb_push(gb_t *b, const void *p, size_t n) {
  if (b->n + n > b->cap) {
    size_t ncap = b->cap == 0 ? 1024 : b->cap * 2;
    while (ncap < b->n + n) {
      ncap *= 2;
    }
    {
      uint8_t *nd = (uint8_t *)realloc(b->d, ncap);
      if (nd == NULL) {
        return -1;
      }
      b->d = nd;
      b->cap = ncap;
    }
  }
  memcpy(b->d + b->n, p, n);
  b->n += n;
  return 0;
}

static void
gb_u64(gb_t *b, uint64_t v) {
  uint8_t t[8];
  for (int i = 0; i < 8; i++) {
    t[i] = (uint8_t)(v >> (8 * i));
  }
  gb_push(b, t, 8);
}

static void
gb_u32(gb_t *b, uint32_t v) {
  uint8_t t[4];
  for (int i = 0; i < 4; i++) {
    t[i] = (uint8_t)(v >> (8 * i));
  }
  gb_push(b, t, 4);
}

static void
gb_str(gb_t *b, const char *s) {
  gb_u64(b, strlen(s));
  gb_push(b, s, strlen(s));
}

static void
gb_f16(gb_t *b, float v) {
  uint32_t bits;
  uint32_t sign;
  uint32_t exp;
  uint32_t mant;
  uint16_t h;
  memcpy(&bits, &v, 4);
  sign = (bits >> 16) & 0x8000u;
  exp = (bits >> 23) & 0xFFu;
  mant = bits & 0x7FFFFFu;
  if (exp == 0xFFu) {
    h = (uint16_t)(sign | 0x7C00u | (mant != 0 ? 0x200u : 0));
  } else if (exp == 0) {
    h = (uint16_t)sign;
  } else {
    int e = (int)exp - 127 + 15;
    if (e >= 31) {
      h = (uint16_t)(sign | 0x7C00u);
    } else if (e <= 0) {
      h = (uint16_t)sign;
    } else {
      h = (uint16_t)(sign | ((uint32_t)e << 10) | (mant >> 13));
    }
  }
  gb_push(b, &h, 2);
}

/* Metadata KV entry: key + type + raw value bytes. */
static void
gb_kv_raw(gb_t *b, const char *key, uint32_t type, const void *val, size_t n) {
  gb_str(b, key);
  gb_u32(b, type);
  gb_push(b, val, n);
}

/* KV conveniences. */
static void
gb_kv_u32(gb_t *b, const char *key, uint32_t v) {
  gb_kv_raw(b, key, 4, &v, 4);
}

static void
gb_kv_str(gb_t *b, const char *key, const char *v) {
  /* GGUF string values are [u64 len][bytes], like keys. */
  gb_str(b, key);
  gb_u32(b, 8);
  gb_str(b, v);
}

/* One tensor to serialize. Data must be laid out row-major with the
 * contiguous (innermost) dim first, per GGUF. */
typedef struct ggt {
  const char *name;
  uint32_t n_dims;
  uint64_t dims[4];
  uint32_t type;
  const uint8_t *data;
  uint64_t nbytes;
} ggt_t;

/* Build a GGUF v3 file: caller-supplied KV section (raw entries, may be
 * empty) + tensor list. Offsets are patched relative to the aligned
 * data start. Returns malloc'd buffer. */
static int
build_gguf(const uint8_t *kvs, size_t kv_n, uint32_t kv_count,
           const ggt_t *tensors, uint32_t n_tensors, uint8_t **out,
           size_t *out_n) {
  gb_t b;
  size_t *off_pos;
  size_t data_start;
  uint32_t i;

  memset(&b, 0, sizeof(b));
  gb_push(&b, "GGUF", 4);
  gb_u32(&b, 3);               /* version */
  gb_u64(&b, n_tensors);
  gb_u64(&b, kv_count);
  if (kv_n > 0) {
    gb_push(&b, kvs, kv_n);
  }
  off_pos = (size_t *)malloc(n_tensors > 0 ? n_tensors * sizeof(size_t) : 1);
  if (off_pos == NULL) {
    free(b.d);
    return -1;
  }
  for (i = 0; i < n_tensors; i++) {
    gb_str(&b, tensors[i].name);
    gb_u32(&b, tensors[i].n_dims);
    for (uint32_t d = 0; d < tensors[i].n_dims; d++) {
      gb_u64(&b, tensors[i].dims[d]);
    }
    gb_u32(&b, tensors[i].type);
    off_pos[i] = b.n;
    gb_u64(&b, 0);
  }
  while (b.n % 32 != 0) {
    uint8_t z = 0;
    gb_push(&b, &z, 1);
  }
  data_start = b.n;
  for (i = 0; i < n_tensors; i++) {
    size_t rel = b.n - data_start;
    while (rel % 32 != 0) {
      uint8_t z = 0;
      gb_push(&b, &z, 1);
      rel = b.n - data_start;
    }
    for (int k = 0; k < 8; k++) {
      b.d[off_pos[i] + (size_t)k] = (uint8_t)(rel >> (8 * k));
    }
    gb_push(&b, tensors[i].data, (size_t)tensors[i].nbytes);
  }
  free(off_pos);
  *out = b.d;
  *out_n = b.n;
  return 0;
}

/* Dequant unit helpers */

static void
free_gguf(uint8_t *buf) {
  free(buf);
}

static pai_status_t
dequant_one(uint32_t type, uint64_t numel, const uint8_t *raw,
            uint64_t raw_n, float *dst) {
  ggt_t t;
  uint8_t *buf = NULL;
  size_t n = 0;
  pai_gguf_t f;
  const pai_gguf_tensor_t *gt;
  float *out = NULL;
  uint64_t out_n = 0;
  pai_status_t st;

  memset(&t, 0, sizeof(t));
  t.name = "t";
  t.n_dims = 1;
  t.dims[0] = numel;
  t.type = type;
  t.data = raw;
  t.nbytes = raw_n;
  if (build_gguf(NULL, 0, 0, &t, 1, &buf, &n) != 0) {
    return PAI_ERR_NOMEM;
  }
  st = pai_gguf_open(buf, n, &f);
  if (st == PAI_OK) {
    gt = pai_gguf_find_tensor(&f, "t");
    st = gt != NULL ? pai_gguf_dequant(&f, gt, &out, &out_n) : PAI_ERR_PROTOCOL;
    if (st == PAI_OK) {
      memcpy(dst, out, (size_t)out_n * sizeof(float));
      free(out);
    }
    pai_gguf_close(&f);
  }
  free_gguf(buf);
  return st;
}

static void
check_dequant(const char *name, uint32_t type, uint64_t numel,
              const uint8_t *raw, uint64_t raw_n, const float *expect) {
  float *got = (float *)malloc((size_t)numel * sizeof(float));
  pai_status_t st;

  CHECK(got != NULL);
  st = dequant_one(type, numel, raw, raw_n, got);
  CHECK_EQ_INT((int)st, PAI_OK);
  if (st == PAI_OK) {
    for (uint64_t i = 0; i < numel; i++) {
      if (fabsf(got[i] - expect[i]) > 1e-3f) {
        printf("  FAIL %s[%llu]: got %f want %f\n", name,
               (unsigned long long)i, got[i], expect[i]);
        g_pai_test_failures++;
        break;
      }
    }
  }
  free(got);
}

/* Dequant unit tests (hand-computed against llama.cpp formulas) */

static void
test_dequant_types(void) {
  uint8_t raw[4096];
  float expect[512];

  /* F32: identity. */
  {
    float v[4] = {1.5f, -2.25f, 0.0f, 3.0f};
    uint8_t *p = raw;
    for (int i = 0; i < 4; i++) {
      memcpy(p, &v[i], 4);
      p += 4;
      expect[i] = v[i];
    }
    check_dequant("f32", PAI_GGUF_F32, 4, raw, 16, expect);
  }

  /* F16 / BF16: known bit patterns. */
  {
    uint16_t h[4] = {0x3C00, 0xC000, 0x3800, 0x0000}; /* 1, -2, .5, 0 */
    float v[4] = {1.0f, -2.0f, 0.5f, 0.0f};
    memcpy(raw, h, 8);
    check_dequant("f16", PAI_GGUF_F16, 4, raw, 8, v);
  }
  {
    uint16_t h[4] = {0x3F80, 0xC000, 0x3F00, 0x0000}; /* bf16 of 1,-2,.5,0 */
    float v[4] = {1.0f, -2.0f, 0.5f, 0.0f};
    memcpy(raw, h, 8);
    check_dequant("bf16", PAI_GGUF_BF16, 4, raw, 8, v);
  }

  /* Q8_0: d=1, qs[i] = i-16. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    for (int i = 0; i < 32; i++) {
      int8_t q = (int8_t)(i - 16);
      gb_push(&b, &q, 1);
    }
    for (int i = 0; i < 32; i++) {
      expect[i] = (float)(i - 16);
    }
    check_dequant("q8_0", PAI_GGUF_Q8_0, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q8_1: d=1, s ignored by dequant, qs[i] = i-16. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 0.0f);
    for (int i = 0; i < 32; i++) {
      int8_t q = (int8_t)(i - 16);
      gb_push(&b, &q, 1);
    }
    for (int i = 0; i < 32; i++) {
      expect[i] = (float)(i - 16);
    }
    check_dequant("q8_1", PAI_GGUF_Q8_1, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q4_0: d=1, signed 4-bit: x0=(q&0xF)-8, x1=(q>>4)-8. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    for (int j = 0; j < 16; j++) {
      uint8_t byte = (uint8_t)(j | ((j ^ 8) << 4)); /* x0=j-8, x1=(j^8)-8 */
      gb_push(&b, &byte, 1);
    }
    for (int j = 0; j < 16; j++) {
      expect[j] = (float)(j - 8);
      expect[j + 16] = (float)((j ^ 8) - 8);
    }
    check_dequant("q4_0", PAI_GGUF_Q4_0, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q4_1: d=1, m=2, unsigned: x0=q&0xF, x1=q>>4. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 2.0f);
    for (int j = 0; j < 16; j++) {
      uint8_t byte = (uint8_t)(j | ((15 - j) << 4)); /* x0=j, x1=15-j */
      gb_push(&b, &byte, 1);
    }
    for (int j = 0; j < 16; j++) {
      expect[j] = (float)(j + 2);
      expect[j + 16] = (float)(17 - j);
    }
    check_dequant("q4_1", PAI_GGUF_Q4_1, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q5_0: d=1, qh=0, signed 5-bit: x=(q&0xF)-16 or (q>>4)-16. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_u32(&b, 0); /* qh */
    for (int j = 0; j < 16; j++) {
      uint8_t byte = (uint8_t)(j | ((j ^ 8) << 4)); /* x0=j-16, x1=(j^8)-16 */
      gb_push(&b, &byte, 1);
    }
    for (int j = 0; j < 16; j++) {
      expect[j] = (float)(j - 16);
      expect[j + 16] = (float)((j ^ 8) - 16);
    }
    check_dequant("q5_0", PAI_GGUF_Q5_0, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q5_1: d=1, m=1, qh=0. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 1.0f);
    gb_u32(&b, 0); /* qh */
    for (int j = 0; j < 16; j++) {
      uint8_t byte = (uint8_t)(j | ((15 - j) << 4)); /* x0=j, x1=15-j */
      gb_push(&b, &byte, 1);
    }
    for (int j = 0; j < 16; j++) {
      expect[j] = (float)(j + 1);
      expect[j + 16] = (float)(16 - j);
    }
    check_dequant("q5_1", PAI_GGUF_Q5_1, 32, b.d, b.n, expect);
    free(b.d);
  }

  /* Q2_K: d=1, min=0, all scales 0x01 (dl=1, ml=0); every q byte 0x1B
   * holds 2-bit fields 3,2,1,0 at shifts 0,2,4,6. Streaming order:
   * element e in the 128-half gets field (e>>5)&3 (j = e>>5), so
   * value = dl * (3 - j) = 3 - ((e>>5)&3). */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    for (int i = 0; i < 16; i++) {
      gb_push(&b, "\x01", 1);
    }
    for (int i = 0; i < 64; i++) {
      gb_push(&b, "\x1B", 1);
    }
    gb_f16(&b, 1.0f);
    gb_f16(&b, 0.0f);
    for (int e = 0; e < 256; e++) {
      expect[e] = (float)(3 - ((e >> 5) & 3));
    }
    check_dequant("q2_k", PAI_GGUF_Q2_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q3_K: d=1, q bytes 0x1B. The 12-byte scales field spreads to 16
   * bytes: first 8 read 0x21 (33) -> dl=1, last 8 read 0x22 (34) ->
   * dl=2, so the second 128-half is scaled by 2. hmask: all 0xFF
   * except hmask[1] = 0x00, so element 1 (hmask[1]&m=0 -> subtract 4)
   * = dl*(3-4) = -1. Value = dl*(3-j) for the rest, j=(e>>5)&3. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    for (int i = 0; i < 32; i++) {
      gb_push(&b, "\xFF", 1); /* hmask */
    }
    b.d[1] = 0x00; /* element 1: hmask bit clear -> subtract 4 */
    for (int i = 0; i < 64; i++) {
      gb_push(&b, "\x1B", 1);
    }
    for (int i = 0; i < 8; i++) {
      gb_push(&b, "\x21", 1);
    }
    for (int i = 0; i < 4; i++) {
      gb_push(&b, "\xAA", 1);
    }
    gb_f16(&b, 1.0f);
    for (int e = 0; e < 256; e++) {
      float dl = e < 128 ? 1.0f : 2.0f;
      expect[e] = dl * (float)(3 - ((e >> 5) & 3));
    }
    /* hm[1]=0 clears all 8 m bits, so elements at l=1 across every
     * j and both 128-halves subtract 4: (3-j)*dl - 4*dl. */
    expect[1] = -1.0f;   /* j=0 dl=1: 1*(3-4)   */
    expect[33] = -2.0f;  /* j=1 dl=1: 1*(2-4)   */
    expect[65] = -3.0f;  /* j=2 dl=1: 1*(1-4)   */
    expect[97] = -4.0f;  /* j=3 dl=1: 1*(0-4)   */
    expect[129] = -2.0f; /* j=0 dl=2: 2*(3-4)   */
    expect[161] = -4.0f; /* j=1 dl=2: 2*(2-4)   */
    expect[193] = -6.0f; /* j=2 dl=2: 2*(1-4)   */
    expect[225] = -8.0f; /* j=3 dl=2: 2*(0-4)   */
    check_dequant("q3_k", PAI_GGUF_Q3_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q4_K: d=1, min=0 (so the m term vanishes), q bytes 0x12 (low
   * nibble 2, high 1), scales all 0x01 (s=1 everywhere). Value =
   * d*s*q - min*m = 2 for low nibbles, 1 for high. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 0.0f);
    for (int i = 0; i < 12; i++) {
      gb_push(&b, "\x01", 1);
    }
    for (int i = 0; i < 128; i++) {
      gb_push(&b, "\x12", 1);
    }
    for (int e = 0; e < 256; e++) {
      expect[e] = (e & 32) ? 1.0f : 2.0f;
    }
    check_dequant("q4_k", PAI_GGUF_Q4_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q4_K with dmin=0.5: the m term must fire (m=1 for the first two
   * 64-groups, m=0 after). Value = d*q - min*m. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 0.5f);
    for (int i = 0; i < 12; i++) {
      gb_push(&b, "\x01", 1);
    }
    for (int i = 0; i < 128; i++) {
      gb_push(&b, "\x12", 1);
    }
    for (int e = 0; e < 256; e++) {
      float m = (e < 128) ? 0.5f : 0.0f;
      expect[e] = ((e & 32) ? 1.0f : 2.0f) - m;
    }
    check_dequant("q4_k_min", PAI_GGUF_Q4_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q5_K: qh bit 0 set -> element 0 gets +16 (2+16=18, then -m1=1
   * since dmin=0 and m1 = min*m = 0... m1=0 here). Others: same as
   * Q4_K with qh=0. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_f16(&b, 1.0f);
    gb_f16(&b, 0.0f);
    for (int i = 0; i < 12; i++) {
      gb_push(&b, "\x01", 1);
    }
    for (int i = 0; i < 32; i++) {
      gb_push(&b, "\x00", 1); /* qh */
    }
    b.d[16] = 0x01; /* qh[0]: bit 0 -> element 0 += 16 */
    for (int i = 0; i < 128; i++) {
      gb_push(&b, "\x12", 1); /* qs */
    }
    for (int e = 0; e < 256; e++) {
      expect[e] = (e & 32) ? 1.0f : 2.0f;
    }
    expect[0] = 18.0f; /* low nibble 2 + qh bit 16, minus m1 (=0) */
    check_dequant("q5_k_qh", PAI_GGUF_Q5_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q6_K: d=1, scales all 1, ql=qh=0 -> every value -32. */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    for (int i = 0; i < 128; i++) {
      gb_push(&b, "\x00", 1);
    }
    for (int i = 0; i < 64; i++) {
      gb_push(&b, "\x00", 1);
    }
    for (int i = 0; i < 16; i++) {
      gb_push(&b, "\x01", 1);
    }
    gb_f16(&b, 1.0f);
    for (int e = 0; e < 256; e++) {
      expect[e] = -32.0f;
    }
    check_dequant("q6_k", PAI_GGUF_Q6_K, 256, b.d, b.n, expect);
    free(b.d);
  }

  /* Q8_K: float d=1, qs[i] = i-128, plus the 32-byte bsums tail that
   * real files carry (block is 292 bytes; dequant ignores bsums). */
  {
    gb_t b;
    memset(&b, 0, sizeof(b));
    gb_push(&b, "\x00\x00\x80\x3F", 4);
    for (int i = 0; i < 256; i++) {
      int8_t q = (int8_t)(i - 128);
      gb_push(&b, &q, 1);
    }
    for (int i = 0; i < 32; i++) {
      gb_push(&b, "\x00", 1); /* bsums */
    }
    for (int i = 0; i < 256; i++) {
      expect[i] = (float)(i - 128);
    }
    check_dequant("q8_k", PAI_GGUF_Q8_K, 256, b.d, b.n, expect);
    free(b.d);
  }
}

/* Reader edge cases */

static void
test_reader_errors(void) {
  uint8_t buf[64];
  uint8_t raw[8];
  pai_gguf_t f;

  /* Too short. */
  CHECK_EQ_INT((int)pai_gguf_open(buf, 4, &f), PAI_ERR_PROTOCOL);
  /* Wrong magic. */
  memset(buf, 0xAA, sizeof(buf));
  CHECK_EQ_INT((int)pai_gguf_open(buf, sizeof(buf), &f), PAI_ERR_PROTOCOL);
  /* Version out of range. */
  memcpy(buf, "GGUF", 4);
  buf[4] = 99;
  buf[5] = 0;
  buf[6] = 0;
  buf[7] = 0;
  CHECK_EQ_INT((int)pai_gguf_open(buf, sizeof(buf), &f), PAI_ERR_UNSUPPORTED);

  /* A tensor whose data extends past EOF must fail dequant. */
  {
    uint8_t *file = NULL;
    size_t n = 0;
    ggt_t t;
    pai_gguf_t ff;
    const pai_gguf_tensor_t *gt;
    float *out = NULL;
    uint64_t out_n = 0;

    memset(&t, 0, sizeof(t));
    t.name = "t";
    t.n_dims = 1;
    t.dims[0] = 32; /* declares 32 f32 (128 B) */
    t.type = PAI_GGUF_F32;
    t.data = (const uint8_t *)raw; /* but only 4 bytes present */
    t.nbytes = 4;
    build_gguf(NULL, 0, 0, &t, 1, &file, &n);
    CHECK_EQ_INT((int)pai_gguf_open(file, n, &ff), PAI_OK);
    gt = pai_gguf_find_tensor(&ff, "t");
    CHECK(gt != NULL);
    if (gt != NULL) {
      CHECK_EQ_INT((int)pai_gguf_dequant(&ff, gt, &out, &out_n),
                   PAI_ERR_PROTOCOL);
    }
    pai_gguf_close(&ff);
    free_gguf(file);
  }

  /* A tensor dim beyond PAI_GGUF_MAX_DIM is rejected at open (would
   * wrap the numel/size math used by dequant and the importer). */
  {
    uint8_t *file = NULL;
    size_t n = 0;
    ggt_t t;
    pai_gguf_t ff;

    memset(&t, 0, sizeof(t));
    t.name = "t";
    t.n_dims = 1;
    t.dims[0] = (uint64_t)PAI_GGUF_MAX_DIM + 1;
    t.type = PAI_GGUF_F32;
    t.data = (const uint8_t *)raw;
    t.nbytes = sizeof(raw);
    build_gguf(NULL, 0, 0, &t, 1, &file, &n);
    CHECK_EQ_INT((int)pai_gguf_open(file, n, &ff), PAI_ERR_PROTOCOL);
    free_gguf(file);
  }
}

/* Tiny LLaMA end-to-end */

#define T_VOCAB 16
#define T_EMBD 8
#define T_FF 16
#define T_CTX 8

static float g_ones8[8];

static void
fill_deterministic(float *w, uint64_t n, uint32_t seed) {
  uint32_t s = seed;
  for (uint64_t i = 0; i < n; i++) {
    s = s * 1664525u + 1013904223u;
    w[i] = ((float)(s >> 8) / 16777216.0f) - 0.5f;
  }
}

/* Build a tiny LLaMA GGUF with F32 weights: 1 layer, 8 hidden, 16 ff,
 * 2 heads, ctx 8, vocab 16 ("a".."p"). */
static int
write_tiny_gguf(const char *path) {
  gb_t kv;
  ggt_t tensors[12];
  uint32_t n_tensors = 0;
  float *token_embd;
  float *q;
  float *k;
  float *v;
  float *o;
  float *gate;
  float *up;
  float *down;
  float *out_w;
  uint8_t *file = NULL;
  size_t n = 0;
  int rc = -1;
  ggt_t *t;

  memset(&kv, 0, sizeof(kv));
  memset(tensors, 0, sizeof(tensors));

  /* Metadata KV section. */
  gb_kv_str(&kv, "general.architecture", "llama");
  gb_kv_u32(&kv, "llama.block_count", 1);
  gb_kv_u32(&kv, "llama.embedding_length", T_EMBD);
  gb_kv_u32(&kv, "llama.feed_forward_length", T_FF);
  gb_kv_u32(&kv, "llama.attention.head_count", 2);
  gb_kv_u32(&kv, "llama.attention.head_count_kv", 2);
  gb_kv_u32(&kv, "llama.context_length", T_CTX);
  gb_kv_u32(&kv, "llama.rope.dimension_count", 4);
  /* tokenizer.ggml.tokens: array of 16 single-char strings. */
  {
    char text[2] = {'a', 0};
    gb_str(&kv, "tokenizer.ggml.tokens");
    gb_u32(&kv, 9); /* array */
    gb_u32(&kv, 8); /* elem type: string */
    gb_u64(&kv, T_VOCAB);
    for (int i = 0; i < T_VOCAB; i++) {
      text[0] = (char)('a' + i);
      gb_str(&kv, text);
    }
  }

  /* Weights. */
  token_embd = (float *)malloc((size_t)T_VOCAB * T_EMBD * sizeof(float));
  q = (float *)malloc((size_t)T_EMBD * T_EMBD * sizeof(float));
  k = (float *)malloc((size_t)T_EMBD * T_EMBD * sizeof(float));
  v = (float *)malloc((size_t)T_EMBD * T_EMBD * sizeof(float));
  o = (float *)malloc((size_t)T_EMBD * T_EMBD * sizeof(float));
  gate = (float *)malloc((size_t)T_EMBD * T_FF * sizeof(float));
  up = (float *)malloc((size_t)T_EMBD * T_FF * sizeof(float));
  down = (float *)malloc((size_t)T_FF * T_EMBD * sizeof(float));
  out_w = (float *)malloc((size_t)T_EMBD * T_VOCAB * sizeof(float));
  if (!token_embd || !q || !k || !v || !o || !gate || !up || !down || !out_w) {
    goto done;
  }
  fill_deterministic(token_embd, (uint64_t)T_VOCAB * T_EMBD, 1);
  fill_deterministic(q, (uint64_t)T_EMBD * T_EMBD, 2);
  fill_deterministic(k, (uint64_t)T_EMBD * T_EMBD, 3);
  fill_deterministic(v, (uint64_t)T_EMBD * T_EMBD, 4);
  fill_deterministic(o, (uint64_t)T_EMBD * T_EMBD, 5);
  fill_deterministic(gate, (uint64_t)T_EMBD * T_FF, 6);
  fill_deterministic(up, (uint64_t)T_EMBD * T_FF, 7);
  fill_deterministic(down, (uint64_t)T_FF * T_EMBD, 8);
  fill_deterministic(out_w, (uint64_t)T_EMBD * T_VOCAB, 9);
  for (int i = 0; i < T_EMBD; i++) {
    g_ones8[i] = 1.0f;
  }

  /* GGUF weight dims are [in, out]; data is row-major [out rows][in]. */
  t = &tensors[n_tensors++];
  t->name = "token_embd.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_VOCAB;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)token_embd;
  t->nbytes = (uint64_t)T_VOCAB * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.attn_norm.weight";
  t->n_dims = 1;
  t->dims[0] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)g_ones8;
  t->nbytes = T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.attn_q.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)q;
  t->nbytes = (uint64_t)T_EMBD * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.attn_k.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)k;
  t->nbytes = (uint64_t)T_EMBD * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.attn_v.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)v;
  t->nbytes = (uint64_t)T_EMBD * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.attn_output.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)o;
  t->nbytes = (uint64_t)T_EMBD * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.ffn_norm.weight";
  t->n_dims = 1;
  t->dims[0] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)g_ones8;
  t->nbytes = T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.ffn_gate.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_FF;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)gate;
  t->nbytes = (uint64_t)T_EMBD * T_FF * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.ffn_up.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_FF;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)up;
  t->nbytes = (uint64_t)T_EMBD * T_FF * 4;

  t = &tensors[n_tensors++];
  t->name = "blk.0.ffn_down.weight";
  t->n_dims = 2;
  t->dims[0] = T_FF;
  t->dims[1] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)down;
  t->nbytes = (uint64_t)T_FF * T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "output_norm.weight";
  t->n_dims = 1;
  t->dims[0] = T_EMBD;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)g_ones8;
  t->nbytes = T_EMBD * 4;

  t = &tensors[n_tensors++];
  t->name = "output.weight";
  t->n_dims = 2;
  t->dims[0] = T_EMBD;
  t->dims[1] = T_VOCAB;
  t->type = PAI_GGUF_F32;
  t->data = (const uint8_t *)out_w;
  t->nbytes = (uint64_t)T_EMBD * T_VOCAB * 4;

  if (build_gguf(kv.d, kv.n, 9, tensors, n_tensors, &file, &n) != 0) {
    goto done;
  }
  {
    FILE *f = fopen(path, "wb");
    if (f != NULL) {
      rc = fwrite(file, 1, n, f) == n ? 0 : -1;
      fclose(f);
    }
  }

done:
  free(token_embd);
  free(q);
  free(k);
  free(v);
  free(o);
  free(gate);
  free(up);
  free(down);
  free(out_w);
  free(file);
  free(kv.d);
  return rc;
}

static char gen_text[512];
static uint32_t gen_text_n;

static void
on_token(const char *token, void *user) {
  (void)user;
  if (gen_text_n + (uint32_t)strlen(token) < sizeof(gen_text)) {
    gen_text_n += (uint32_t)strlen(token);
    strcat(gen_text, token);
  }
}

static int
e2e_run(const char *out_path, const pai_container_quant_t *quant) {
  pai_status_t st;
  pai_model_t *model = NULL;
  pai_session_t *session = NULL;

  st = pai_llama_import("tiny_llama.gguf", out_path, quant);
  if (st != PAI_OK) {
    printf("  FAIL import %s: %s\n", out_path, pai_status_str(st));
    g_pai_test_failures++;
    return -1;
  }
  st = pai_model_open(NULL, out_path, &model);
  if (st != PAI_OK) {
    printf("  FAIL open %s: %s\n", out_path, pai_status_str(st));
    g_pai_test_failures++;
    return -1;
  }
  CHECK_EQ_INT((int)model->vocab_size, T_VOCAB);
  CHECK_EQ_INT((int)model->context_len, T_CTX);
  CHECK_EQ_INT((int)model->num_layers, 1);

  st = pai_session_init(model, &session);
  if (st != PAI_OK) {
    printf("  FAIL session %s: %s\n", out_path, pai_status_str(st));
    g_pai_test_failures++;
    pai_model_close(model);
    return -1;
  }
  CHECK_EQ_INT((int)pai_session_set_generation(session, 4, 0), PAI_OK);

  memset(gen_text, 0, sizeof(gen_text));
  gen_text_n = 0;
  st = pai_session_generate(session, "a", on_token, NULL);
  if (st != PAI_OK) {
    printf("  FAIL generate %s: %s\n", out_path, pai_status_str(st));
    g_pai_test_failures++;
  } else {
    CHECK_EQ_UINT(gen_text_n, 4);
    CHECK(gen_text[0] != '\0');
  }

  /* Determinism: same seed -> same text. */
  if (st == PAI_OK) {
    char first[512];
    memcpy(first, gen_text, sizeof(first));
    memset(gen_text, 0, sizeof(gen_text));
    gen_text_n = 0;
    st = pai_session_generate(session, "a", on_token, NULL);
    CHECK_EQ_INT((int)st, PAI_OK);
    if (st == PAI_OK) {
      CHECK(strcmp(first, gen_text) == 0);
    }
  }

  pai_session_destroy(session);
  pai_model_close(model);
  return 0;
}

TEST_MAIN_BEGIN()

test_dequant_types();
test_reader_errors();

/* Magic detection. */
{
  uint8_t head[4] = {'G', 'G', 'U', 'F'};
  uint8_t nohead[4] = {'N', 'O', 'P', 'E'};
  CHECK_EQ_INT(pai_llama_is_gguf(head), 1);
  CHECK_EQ_INT(pai_llama_is_gguf(nohead), 0);
}

/* End-to-end: tiny LLaMA GGUF -> .pai, f32 and q8. */
CHECK_EQ_INT(write_tiny_gguf("tiny_llama.gguf"), 0);
{
  pai_container_quant_t q;
  memset(&q, 0, sizeof(q));
  e2e_run("tiny_llama.pai", NULL);
  q.quant = "q8";
  e2e_run("tiny_llama_q8.pai", &q);
}

TEST_MAIN_END()
