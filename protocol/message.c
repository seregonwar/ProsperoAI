/*
 * ProsperoAI — Prospero Protocol
 *
 * Message payload codecs (v0): small, length-prefixed, fully validated;
 * the frame header carries request/session correlation.
 */

#include <protocol/protocol.h>

#include <string.h>

#define PAI_PROTO_ERROR_MSG_MAX 255u

const char *
pai_proto_msg_name(uint32_t msg_type) {
  switch (msg_type) {
  case PAI_PROTO_MSG_HELLO:        return "hello";
  case PAI_PROTO_MSG_HELLO_ACK:    return "hello_ack";
  case PAI_PROTO_MSG_HELLO_NACK:   return "hello_nack";
  case PAI_PROTO_MSG_PING:         return "ping";
  case PAI_PROTO_MSG_PONG:         return "pong";
  case PAI_PROTO_MSG_ERROR:        return "error";
  case PAI_PROTO_MSG_CLOSE:        return "close";
  case PAI_PROTO_MSG_SESSION_CREATE:  return "session_create";
  case PAI_PROTO_MSG_SESSION_CREATED: return "session_created";
  case PAI_PROTO_MSG_SESSION_CLOSE:   return "session_close";
  case PAI_PROTO_MSG_SESSION_CLOSED:  return "session_closed";
  case PAI_PROTO_MSG_GENERATE:     return "generate";
  case PAI_PROTO_MSG_ACCEPTED:     return "accepted";
  case PAI_PROTO_MSG_TOKEN:        return "token";
  case PAI_PROTO_MSG_COMPLETE:     return "complete";
  case PAI_PROTO_MSG_EMBED:        return "embed";
  case PAI_PROTO_MSG_EMBEDDING:    return "embedding";
  case PAI_PROTO_MSG_STREAM_DATA:  return "stream_data";
  case PAI_PROTO_MSG_STREAM_CLOSE: return "stream_close";
  default:                         return "unknown";
  }
}

static void
put_le16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)(v >> 8);
}

static void
put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)((v >> 8) & 0xFFu);
  p[2] = (uint8_t)((v >> 16) & 0xFFu);
  p[3] = (uint8_t)(v >> 24);
}

static uint16_t
get_le16(const uint8_t *p) {
  return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t
get_le32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static void
put_le64(uint8_t *p, uint64_t v) {
  uint32_t lo = (uint32_t)(v & 0xFFFFFFFFu);
  uint32_t hi = (uint32_t)(v >> 32);
  put_le32(p, lo);
  put_le32(p + 4, hi);
}

static uint64_t
get_le64(const uint8_t *p) {
  return (uint64_t)get_le32(p) | ((uint64_t)get_le32(p + 4) << 32);
}

/* ------------------------------------------------------------------ */
/* caps / u32 pairs                                                    */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_msg_encode_caps(uint8_t *out, uint32_t cap, uint32_t caps,
                          uint32_t *out_len) {
  if (!out || !out_len || cap < PAI_PROTO_U32_PAIR_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }
  put_le32(out + 0, caps);
  put_le32(out + 4, 0);
  *out_len = PAI_PROTO_U32_PAIR_SIZE;
  return PAI_OK;
}

pai_status_t
pai_proto_msg_decode_caps(const uint8_t *payload, uint32_t len,
                          uint32_t *out_caps) {
  if (!payload || !out_caps || len < PAI_PROTO_U32_PAIR_SIZE) {
    return PAI_ERR_PROTOCOL;
  }
  *out_caps = get_le32(payload + 0);
  return PAI_OK;
}

pai_status_t
pai_proto_msg_encode_u32(uint8_t *out, uint32_t cap, uint32_t value,
                         uint32_t *out_len) {
  if (!out || !out_len || cap < PAI_PROTO_U32_PAIR_SIZE) {
    return PAI_ERR_INVALID_ARG;
  }
  put_le32(out + 0, value);
  put_le32(out + 4, 0);
  *out_len = PAI_PROTO_U32_PAIR_SIZE;
  return PAI_OK;
}

pai_status_t
pai_proto_msg_decode_u32(const uint8_t *payload, uint32_t len,
                         uint32_t *out_value) {
  if (!payload || !out_value || len < PAI_PROTO_U32_PAIR_SIZE) {
    return PAI_ERR_PROTOCOL;
  }
  *out_value = get_le32(payload + 0);
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* error                                                               */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_msg_encode_error(uint8_t *out, uint32_t cap, pai_status_t status,
                           const char *msg, uint32_t *out_len) {
  uint32_t msg_len = 0;
  uint32_t total;

  if (!out || !out_len) {
    return PAI_ERR_INVALID_ARG;
  }
  if (msg) {
    msg_len = (uint32_t)strlen(msg);
    if (msg_len > PAI_PROTO_ERROR_MSG_MAX) {
      msg_len = PAI_PROTO_ERROR_MSG_MAX;
    }
  }

  total = 8u + msg_len;
  if (cap < total) {
    return PAI_ERR_INVALID_ARG;
  }

  put_le32(out + 0, (uint32_t)status);
  put_le32(out + 4, msg_len);
  if (msg_len > 0) {
    memcpy(out + 8, msg, msg_len);
  }
  *out_len = total;
  return PAI_OK;
}

pai_status_t
pai_proto_msg_decode_error(const uint8_t *payload, uint32_t len,
                           pai_status_t *out_status, char *out_msg,
                           uint32_t msg_cap) {
  uint32_t msg_len;

  if (!payload || !out_status || len < 8u) {
    return PAI_ERR_PROTOCOL;
  }
  *out_status = (pai_status_t)get_le32(payload + 0);
  msg_len = get_le32(payload + 4);
  if (msg_len > len - 8u) {
    return PAI_ERR_PROTOCOL;
  }
  if (out_msg && msg_cap > 0) {
    uint32_t n = msg_len < msg_cap - 1u ? msg_len : msg_cap - 1u;
    memcpy(out_msg, payload + 8, n);
    out_msg[n] = '\0';
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* generate / token (u16 length + bytes)                               */
/* ------------------------------------------------------------------ */

static pai_status_t
encode_u16_blob(uint8_t *out, uint32_t cap, const void *blob,
                uint32_t blob_len, uint32_t *out_len) {
  uint32_t total;

  if (!out || !out_len) {
    return PAI_ERR_INVALID_ARG;
  }
  if (blob_len > 0xFFFFu) {
    return PAI_ERR_INVALID_ARG;
  }
  if (blob_len > 0 && !blob) {
    return PAI_ERR_INVALID_ARG;
  }

  total = 2u + blob_len;
  if (cap < total) {
    return PAI_ERR_INVALID_ARG;
  }

  put_le16(out + 0, (uint16_t)blob_len);
  if (blob_len > 0) {
    memcpy(out + 2, blob, blob_len);
  }
  *out_len = total;
  return PAI_OK;
}

static pai_status_t
decode_u16_blob(const uint8_t *payload, uint32_t len, const uint8_t **out_blob,
                uint32_t *out_blob_len) {
  uint32_t blob_len;

  if (!payload || !out_blob || !out_blob_len || len < 2u) {
    return PAI_ERR_PROTOCOL;
  }
  blob_len = get_le16(payload + 0);
  if (blob_len > len - 2u) {
    return PAI_ERR_PROTOCOL;
  }
  *out_blob = payload + 2;
  *out_blob_len = blob_len;
  return PAI_OK;
}

/* Copy the sampler trailer (magic + fixed-size block) after the
 * prompt blob. The layout is appended to the v1 payload, so v1
 * decoders (which only read the prompt) ignore it. */
static void
encode_sampler_trailer(uint8_t *out, const pai_proto_sampler_t *s) {
  put_le32(out + 0, PAI_PROTO_SAMPLER_MAGIC);
  /* Float bits are stored verbatim (both target platforms are
   * IEEE-754 little-endian); integer fields use the LE helpers. */
  memcpy(out + 4, &s->temperature, sizeof(s->temperature));
  memcpy(out + 8, &s->top_p, sizeof(s->top_p));
  put_le32(out + 12, s->top_k);
  put_le32(out + 16, s->max_tokens);
  put_le64(out + 20, s->seed);
}

static pai_status_t
decode_sampler_trailer(const uint8_t *trailer, uint32_t len,
                       pai_proto_sampler_t *out) {
  if (len != 4u + sizeof(pai_proto_sampler_t) ||
      get_le32(trailer + 0) != PAI_PROTO_SAMPLER_MAGIC) {
    return PAI_ERR_PROTOCOL;
  }
  memcpy(&out->temperature, trailer + 4, sizeof(out->temperature));
  memcpy(&out->top_p, trailer + 8, sizeof(out->top_p));
  out->top_k = get_le32(trailer + 12);
  out->max_tokens = get_le32(trailer + 16);
  out->seed = get_le64(trailer + 20);
  return PAI_OK;
}

pai_status_t
pai_proto_msg_encode_generate(uint8_t *out, uint32_t cap, const char *prompt,
                              uint32_t *out_len) {
  return pai_proto_msg_encode_generate2(out, cap, prompt, NULL, out_len);
}

pai_status_t
pai_proto_msg_decode_generate(const uint8_t *payload, uint32_t len,
                              const char **out_prompt,
                              uint32_t *out_prompt_len) {
  return pai_proto_msg_decode_generate2(payload, len, out_prompt,
                                        out_prompt_len, NULL, NULL);
}

pai_status_t
pai_proto_msg_encode_generate2(uint8_t *out, uint32_t cap, const char *prompt,
                               const pai_proto_sampler_t *sampler,
                               uint32_t *out_len) {
  uint32_t plen;

  if (!prompt) {
    return PAI_ERR_INVALID_ARG;
  }
  plen = (uint32_t)strlen(prompt);
  if (plen > 0xFFFFu) {
    return PAI_ERR_INVALID_ARG;
  }
  if (sampler == NULL) {
    return encode_u16_blob(out, cap, prompt, plen, out_len);
  }
  {
    uint32_t total = 2u + plen + 4u + (uint32_t)sizeof(pai_proto_sampler_t);
    if (cap < total) {
      return PAI_ERR_INVALID_ARG;
    }
    put_le16(out + 0, (uint16_t)plen);
    if (plen > 0) {
      memcpy(out + 2, prompt, plen);
    }
    encode_sampler_trailer(out + 2 + plen, sampler);
    *out_len = total;
    return PAI_OK;
  }
}

pai_status_t
pai_proto_msg_decode_generate2(const uint8_t *payload, uint32_t len,
                               const char **out_prompt,
                               uint32_t *out_prompt_len,
                               pai_proto_sampler_t *out_sampler,
                               int *out_has_sampler) {
  const uint8_t *prompt;
  uint32_t plen;
  pai_status_t st;

  if (out_has_sampler != NULL) {
    *out_has_sampler = 0;
  }
  st = decode_u16_blob(payload, len, &prompt, &plen);
  if (st != PAI_OK) {
    return st;
  }
  if (out_prompt != NULL) {
    *out_prompt = (const char *)prompt;
  }
  if (out_prompt_len != NULL) {
    *out_prompt_len = plen;
  }
  /* Optional trailer: { magic; sampler block }. Anything else after
   * the prompt is tolerated (v1 payloads have no trailer). */
  if (out_sampler != NULL && out_has_sampler != NULL && len > 2u + plen &&
      decode_sampler_trailer(payload + 2u + plen, len - (2u + plen),
                             out_sampler) == PAI_OK) {
    *out_has_sampler = 1;
  }
  return PAI_OK;
}

pai_status_t
pai_proto_msg_encode_token(uint8_t *out, uint32_t cap, const uint8_t *token,
                           uint32_t token_len, uint32_t *out_len) {
  return encode_u16_blob(out, cap, token, token_len, out_len);
}

pai_status_t
pai_proto_msg_decode_token(const uint8_t *payload, uint32_t len,
                           const uint8_t **out_token,
                           uint32_t *out_token_len) {
  return decode_u16_blob(payload, len, out_token, out_token_len);
}

/* ------------------------------------------------------------------ */
/* embed / embedding                                                   */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_msg_encode_embed(uint8_t *out, uint32_t cap, const char *text,
                           uint32_t text_len, uint32_t *out_len) {
  return encode_u16_blob(out, cap, text, text_len, out_len);
}

pai_status_t
pai_proto_msg_decode_embed(const uint8_t *payload, uint32_t len,
                           const char **out_text, uint32_t *out_text_len) {
  const uint8_t *blob;
  uint32_t blob_len;
  pai_status_t st;

  st = decode_u16_blob(payload, len, &blob, &blob_len);
  if (st != PAI_OK) {
    return st;
  }
  /* EMBED has no trailer, so trailing bytes are a protocol violation
   * (unlike GENERATE, whose v2 trailer is intentionally tolerated). */
  if (len != 2u + blob_len) {
    return PAI_ERR_PROTOCOL;
  }
  if (out_text != NULL) {
    *out_text = (const char *)blob;
  }
  if (out_text_len != NULL) {
    *out_text_len = blob_len;
  }
  return PAI_OK;
}

pai_status_t
pai_proto_msg_encode_embedding(uint8_t *out, uint32_t cap,
                               const float *values, uint32_t dim,
                               uint32_t *out_len) {
  uint32_t total;

  if (!out || !out_len) {
    return PAI_ERR_INVALID_ARG;
  }
  if (dim == 0 || dim > PAI_PROTO_MAX_EMBED_DIM || !values) {
    return PAI_ERR_INVALID_ARG;
  }
  total = 4u + dim * 4u;
  if (cap < total) {
    return PAI_ERR_INVALID_ARG;
  }
  put_le32(out + 0, dim);
  /* Float bits verbatim (both target platforms are IEEE-754 LE). */
  memcpy(out + 4, values, (size_t)dim * sizeof(float));
  *out_len = total;
  return PAI_OK;
}

pai_status_t
pai_proto_msg_decode_embedding(const uint8_t *payload, uint32_t len,
                               const float **out_values, uint32_t *out_dim) {
  uint32_t dim;

  if (!payload || !out_values || !out_dim || len < 4u) {
    return PAI_ERR_PROTOCOL;
  }
  dim = get_le32(payload + 0);
  if (dim == 0 || dim > PAI_PROTO_MAX_EMBED_DIM ||
      len != 4u + dim * 4u) {
    return PAI_ERR_PROTOCOL;
  }
  *out_values = (const float *)(const void *)(payload + 4);
  *out_dim = dim;
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* stream_data                                                         */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_msg_encode_stream_data(uint8_t *out, uint32_t cap, uint8_t kind,
                                 uint32_t seq, const uint8_t *data,
                                 uint32_t data_len, uint32_t *out_len) {
  uint32_t total;

  if (!out || !out_len) {
    return PAI_ERR_INVALID_ARG;
  }
  if (data_len > 0 && !data) {
    return PAI_ERR_INVALID_ARG;
  }

  /* 1 kind + 3 reserved + 4 seq + 4 data_len + data. */
  total = 12u + data_len;
  if (cap < total) {
    return PAI_ERR_INVALID_ARG;
  }

  out[0] = kind;
  out[1] = 0;
  out[2] = 0;
  out[3] = 0;
  put_le32(out + 4, seq);
  put_le32(out + 8, data_len);
  if (data_len > 0) {
    memcpy(out + 12, data, data_len);
  }
  *out_len = total;
  return PAI_OK;
}

pai_status_t
pai_proto_msg_decode_stream_data(const uint8_t *payload, uint32_t len,
                                 uint8_t *out_kind, uint32_t *out_seq,
                                 const uint8_t **out_data,
                                 uint32_t *out_data_len) {
  uint32_t data_len;

  if (!payload || !out_kind || !out_seq || !out_data || !out_data_len ||
      len < 12u) {
    return PAI_ERR_PROTOCOL;
  }
  *out_kind = payload[0];
  *out_seq = get_le32(payload + 4);
  data_len = get_le32(payload + 8);
  if (data_len > len - 12u) {
    return PAI_ERR_PROTOCOL;
  }
  *out_data = payload + 12;
  *out_data_len = data_len;
  return PAI_OK;
}
