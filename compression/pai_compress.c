/*
 * ProsperoAI — weight compression frontend (see pai_compress.h)
 */

#include "pai_compress.h"

#include <stdlib.h>
#include <string.h>

int kk_decode_stream(const uint8_t *src, uint64_t len, uint8_t *dst,
                     uint64_t dst_len);
int kk_encode_stream(const uint8_t *src, uint64_t len, void *out);

typedef struct {
  uint8_t *p;
  uint32_t n;
  uint32_t cap;
} pc_buf_t;

/* envelope: "PAKC" | u32 method | u64 decoded size | stream */
#define PC_ENV_HDR (4u + 4u + 8u)

uint64_t
pai_compress_bound(uint64_t len) {
  /* worst case: every quantum stored raw + container overhead */
  return len + len / 0x40000u * 8u + 0x10000u;
}

pai_status_t
pai_compress_encode(const uint8_t *src, uint64_t len, uint8_t *dst,
                    uint64_t cap, uint64_t *out_len) {
  pc_buf_t b;
  uint64_t bound = pai_compress_bound(len);
  int r;

  if (src == NULL || dst == NULL || out_len == NULL || len == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  if (cap < PC_ENV_HDR || bound + PC_ENV_HDR > cap) {
    return PAI_ERR_INVALID_ARG;
  }

  b.p = dst;
  b.n = 0;
  b.cap = (uint32_t)cap;

  memcpy(b.p + b.n, "PAKC", 4);
  b.n += 4;
  {
    uint32_t m = PAI_COMP_METHOD_KRAKEN;
    memcpy(b.p + b.n, &m, 4);
    b.n += 4;
  }
  memcpy(b.p + b.n, &len, 8);
  b.n += 8;

  {
    pc_buf_t stream = {b.p + b.n, 0, b.cap - b.n};
    r = kk_encode_stream(src, len, &stream);
    if (r != 0) {
      return PAI_ERR_INTERNAL;
    }
    b.n += stream.n;
  }

  *out_len = b.n;
  return PAI_OK;
}

pai_status_t
pai_compress_decode(const uint8_t *src, uint64_t len, uint8_t *dst,
                    uint64_t cap, uint64_t *out_len) {
  uint32_t method;
  uint64_t dec_len;
  const uint8_t *stream;
  uint64_t stream_len;
  int r;

  if (src == NULL || dst == NULL || out_len == NULL) {
    return PAI_ERR_INVALID_ARG;
  }

  if (len >= PC_ENV_HDR && memcmp(src, "PAKC", 4) == 0) {
    memcpy(&method, src + 4, 4);
    memcpy(&dec_len, src + 8, 8);
    stream = src + PC_ENV_HDR;
    stream_len = len - PC_ENV_HDR;
    if (method != PAI_COMP_METHOD_KRAKEN) {
      return PAI_ERR_UNSUPPORTED;
    }
    if (dec_len > cap || dec_len == 0) {
      return PAI_ERR_INVALID_ARG;
    }
    r = kk_decode_stream(stream, stream_len, dst, dec_len);
    if (r != 0) {
      return PAI_ERR_PROTOCOL;
    }
    *out_len = dec_len;
    return PAI_OK;
  }

  return PAI_ERR_UNSUPPORTED;
}

pai_status_t
pai_compress_probe(const uint8_t *src, uint64_t len, uint32_t *out_method) {
  uint32_t method;

  if (src == NULL || out_method == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (len >= PC_ENV_HDR && memcmp(src, "PAKC", 4) == 0) {
    memcpy(&method, src + 4, 4);
    *out_method = method;
    return PAI_OK;
  }
  if (len >= 2 && (src[0] & 0xFu) == 0xCu) {
    *out_method = PAI_COMP_METHOD_AMPR; /* raw Kraken-family stream */
    return PAI_OK;
  }
  *out_method = PAI_COMP_METHOD_NONE;
  return PAI_OK;
}
