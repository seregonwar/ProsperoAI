/*
 * ProsperoAI — Prospero Protocol
 *
 * CRC-32 (ISO-HDLC / CRC-32/MPEG-2, reflected zlib polynomial
 * 0xEDB88320): per-frame integrity on the wire (§24), reusable for
 * per-chunk hashes in the model transfer engine (§23).
 */

#include <protocol/protocol.h>

static uint32_t crc_table[256];
static int crc_table_ready;

static void
crc_table_build(void) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t crc = i;
    for (int b = 0; b < 8; b++) {
      crc = (crc & 1u) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    crc_table[i] = crc;
  }
  crc_table_ready = 1;
}

uint32_t
pai_proto_crc32_init(void) {
  return 0xFFFFFFFFu;
}

uint32_t
pai_proto_crc32_upd(uint32_t running, const void *data, uint32_t nbytes) {
  const uint8_t *p = (const uint8_t *)data;

  if (!crc_table_ready) {
    crc_table_build();
  }

  for (uint32_t i = 0; i < nbytes; i++) {
    running = crc_table[(running ^ p[i]) & 0xFFu] ^ (running >> 8);
  }
  return running;
}

uint32_t
pai_proto_crc32_fin(uint32_t running) {
  return running ^ 0xFFFFFFFFu;
}

uint32_t
pai_proto_crc32(const void *data, uint32_t nbytes) {
  return pai_proto_crc32_fin(pai_proto_crc32_upd(pai_proto_crc32_init(), data,
                                                 nbytes));
}
