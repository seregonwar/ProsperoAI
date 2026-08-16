/*
 * ProsperoAI — Prospero Protocol
 *
 * Frame codec: serialization/parsing of the 40-byte frame header and
 * CRC-32 integrity validation (whitepaper §24).
 *
 * Layout (little-endian):
 *   [0]  u32 magic        [4] u8 major   [5] u8 minor   [6] u16 flags
 *   [8]  u32 msg_type     [12] u32 payload_len
 *   [16] u64 request_id   [24] u64 session_id
 *   [32] u32 crc32        [36] u32 reserved (0)
 */

#include <protocol/protocol.h>

#include <string.h>

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

static void
put_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    p[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
  }
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

static uint64_t
get_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) {
    v |= (uint64_t)p[i] << (8 * i);
  }
  return v;
}

uint32_t
pai_proto_frame_size(uint32_t payload_len) {
  return PAI_PROTO_HEADER_SIZE + payload_len;
}

pai_status_t
pai_proto_frame_encode(const pai_proto_frame_t *frame, uint8_t *out,
                       uint32_t cap, uint32_t *out_nbytes) {
  uint32_t total;
  uint32_t crc;
  uint32_t header_crc_bytes;

  if (!frame || !out || !out_nbytes) {
    return PAI_ERR_INVALID_ARG;
  }
  if (frame->payload_len > PAI_PROTO_MAX_PAYLOAD) {
    return PAI_ERR_INVALID_ARG;
  }
  if (frame->flags & PAI_PROTO_FLAG_COMPRESSED) {
    /* v0 receivers do not understand compressed payloads. */
    return PAI_ERR_INVALID_ARG;
  }

  total = pai_proto_frame_size(frame->payload_len);
  if (cap < total) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(out, 0, PAI_PROTO_HEADER_SIZE);
  put_le32(out + 0, PAI_PROTO_MAGIC);
  out[4] = PAI_PROTO_VERSION_MAJOR;
  out[5] = PAI_PROTO_VERSION_MINOR;
  put_le16(out + 6, (uint16_t)frame->flags);
  put_le32(out + 8, frame->msg_type);
  put_le32(out + 12, frame->payload_len);
  put_le64(out + 16, frame->request_id);
  put_le64(out + 24, frame->session_id);
  /* reserved [36] stays 0 */

  if (frame->payload_len > 0) {
    memcpy(out + PAI_PROTO_HEADER_SIZE, frame->payload, frame->payload_len);
  }

  /* CRC over header bytes [0,32) plus the payload. */
  crc = pai_proto_crc32_init();
  header_crc_bytes = PAI_PROTO_HEADER_SIZE - 8u; /* up to the crc field */
  crc = pai_proto_crc32_upd(crc, out, header_crc_bytes);
  crc = pai_proto_crc32_upd(crc, out + PAI_PROTO_HEADER_SIZE,
                            frame->payload_len);
  put_le32(out + 32, pai_proto_crc32_fin(crc));

  *out_nbytes = total;
  return PAI_OK;
}

pai_status_t
pai_proto_frame_decode(const uint8_t *data, pai_proto_frame_t *out) {
  if (!data || !out) {
    return PAI_ERR_INVALID_ARG;
  }
  if (get_le32(data + 0) != PAI_PROTO_MAGIC) {
    return PAI_ERR_PROTOCOL;
  }
  if (get_le32(data + 36) != 0) {
    return PAI_ERR_PROTOCOL; /* reserved field must be zero */
  }
  if (get_le32(data + 12) > PAI_PROTO_MAX_PAYLOAD) {
    return PAI_ERR_PROTOCOL;
  }

  memset(out, 0, sizeof(*out));
  out->version = (uint16_t)((data[4] << 8) | data[5]);
  out->flags = get_le16(data + 6);
  out->msg_type = get_le32(data + 8);
  out->payload_len = get_le32(data + 12);
  out->request_id = get_le64(data + 16);
  out->session_id = get_le64(data + 24);
  out->payload = NULL;
  return PAI_OK;
}

pai_status_t
pai_proto_frame_validate(const uint8_t *header, const uint8_t *payload) {
  uint32_t expected;
  uint32_t crc;
  uint32_t header_crc_bytes;
  pai_proto_frame_t f;

  if (!header) {
    return PAI_ERR_INVALID_ARG;
  }
  if (pai_proto_frame_decode(header, &f) != PAI_OK) {
    return PAI_ERR_PROTOCOL;
  }
  if (f.payload_len > 0 && !payload) {
    return PAI_ERR_INVALID_ARG;
  }

  crc = pai_proto_crc32_init();
  header_crc_bytes = PAI_PROTO_HEADER_SIZE - 8u;
  crc = pai_proto_crc32_upd(crc, header, header_crc_bytes);
  crc = pai_proto_crc32_upd(crc, payload, f.payload_len);
  expected = pai_proto_crc32_fin(crc);

  if (get_le32(header + 32) != expected) {
    return PAI_ERR_PROTOCOL;
  }
  return PAI_OK;
}
