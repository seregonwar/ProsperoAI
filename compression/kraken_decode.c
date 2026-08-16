/*
 * ProsperoAI — Kraken-family decoder (clean-room, see pai_compress.h)
 *
 * Format notes (the contract, re-derived from stream behavior; the
 * open-source decompressors by Powzix/ooz and drakmor/ampr_emu were
 * used as documentation only — no code was copied):
 *
 * Container: at each 0x40000 output boundary a 2-byte header appears:
 *   b0: low nibble must be 0xC, bits 4-5 zero, bit6 = uncompressed,
 *       bit7 = restart decoder
 *   b1: bits 0-6 = method (5 LZNA, 6 Kraken, 10 Mermaid, 11 Bitknit,
 *       12 Leviathan), bit7 = quantum checksums present
 * Then per quantum (outputs min(remaining, 0x40000) bytes), a big-
 * endian u24 v:
 *   size = v & 0x3FFFF. size != 0x3FFFF: compressed_size = size + 1,
 *   flag1 = bit18, flag2 = bit19 (whole-match, unsupported here);
 *   if checksums, a u24 checksum follows (not verified here).
 *   size == 0x3FFFF and v >> 18 == 1: "fill", the next byte is the fill
 *   value for the whole quantum. size == 0x3FFFF and v >> 18 == 3:
 *   PAI extension, raw full 0x40000 quantum (the plain size field
 *   cannot represent it). compressed_size == dst size -> raw copy.
 *
 * Mermaid/Selkie quantum: sequence of chunks each producing
 * min(remaining, 0x20000) bytes; chunk header is a big-endian u24 h:
 *   if bit23 == 0: the chunk is a nested chunk producing exactly
 *   dst_count bytes. Else: mode = bits 19-22 (0 or 1), src_used =
 *   bits 0-18. src_used == dst_count && mode == 0 -> raw copy.
 *   Otherwise an LZ table follows:
 *     - if the quantum starts at output offset 0, the first 8 bytes are
 *       stored raw at the start of the payload
 *     - lit stream  = nested chunk
 *     - cmd stream  = nested chunk
 *     - if dst_count > 0x10000: u16 cmd split (commands of first half)
 *     - u16 off16_count; 0xFFFF means two entropy-coded half streams
 *       (unsupported here), otherwise inline u16s
 *     - u24 off32 sizes (little-endian): hi 12 = first-half count,
 *       lo 12 = second-half count (each 4095 -> u16 extension follows)
 *     - far offsets: 3-byte little-endian each; values >= 0xC00000 get
 *       one more byte shifted left by 22
 *     - the rest is the raw length stream
 *   Runs (per half of 0x10000, own off32 stream; recent_offs starts at
 *   -8 for the very first half of the stream and carries over):
 *     cmd >= 24: litlen = cmd & 7, matchlen = (cmd >> 3) & 0xF,
 *       use_distance = cmd >> 7; if use_distance, recent_offs =
 *       -(next off16). Copy litlen literals then matchlen bytes from
 *       dst + recent_offs (matches may reach back into earlier
 *       quanta, up to 64 KiB via off16).
 *     cmd > 2: matchlen = cmd + 5, far offset relative to chunk start
 *     cmd == 0: length = length stream (byte L; L > 251 -> L += 4*u16)
 *       + 64 -> copy literals
 *     cmd == 1: length + 91 -> copy from off16 match
 *     cmd == 2: length + 29 -> copy from far-offset match
 *     Tail: remaining dst bytes come from the lit stream.
 *   Mode 0: literals are delta-coded against dst[recent_offs];
 *   mode 1: plain literals.
 *
 * Nested chunk: u8 t = (src[0] >> 4) & 7, bit7 = short-size mode:
 *   t == 0: raw copy, 12-bit size (2 bytes) or 18-bit (3 bytes),
 *   sizes stored big-endian
 *   t != 0: sizes in the header; dst = src_size + hi10 + 1 (short) or
 *   from 18-bit fields (long); t == 3 is RLE, others unsupported here.
 */

#include "pai_compress.h"

#include <pai/log.h>

#include <stdlib.h>
#include <string.h>

#define KK_QUANTUM_OUT 0x40000u
#define KK_HALF_OUT 0x10000u
#define KK_CHUNK_OUT 0x20000u

typedef struct {
  const uint8_t *p;
  const uint8_t *end;
} kk_in_t;

static int kk_nested_chunk_decode(kk_in_t *in, uint8_t *dst,
                                  uint32_t dst_cap, uint32_t *out_dst);

static int
kk_in_u8(kk_in_t *in, uint32_t *out) {
  if (in->p >= in->end) {
    return -1;
  }
  *out = *in->p++;
  return 0;
}

static int
kk_in_u16(kk_in_t *in, uint32_t *out) {
  if (in->end - in->p < 2) {
    return -1;
  }
  *out = (uint32_t)in->p[0] | ((uint32_t)in->p[1] << 8);
  in->p += 2;
  return 0;
}

/* Little-endian u24. */
static int
kk_in_u24(kk_in_t *in, uint32_t *out) {
  if (in->end - in->p < 3) {
    return -1;
  }
  *out = (uint32_t)in->p[0] | ((uint32_t)in->p[1] << 8) |
         ((uint32_t)in->p[2] << 16);
  in->p += 3;
  return 0;
}

/* Big-endian u24: the container quantum header and the chunk headers
 * are stored high byte first. */
static int
kk_in_u24_be(kk_in_t *in, uint32_t *out) {
  if (in->end - in->p < 3) {
    return -1;
  }
  *out = ((uint32_t)in->p[0] << 16) | ((uint32_t)in->p[1] << 8) |
         (uint32_t)in->p[2];
  in->p += 3;
  return 0;
}

static int
kk_in_bytes(kk_in_t *in, const uint8_t **out, uint32_t n) {
  if ((uint32_t)(in->end - in->p) < n) {
    return -1;
  }
  *out = in->p;
  in->p += n;
  return 0;
}

/* Copy a match; source always precedes destination (overlapping OK). */
static void
kk_copy_match(uint8_t *dst, const uint8_t *match, size_t n) {
  size_t i = 0;
  while (i + 8 <= n) {
    uint64_t v;
    memcpy(&v, match + i, 8);
    memcpy(dst + i, &v, 8);
    i += 8;
  }
  while (i < n) {
    dst[i] = match[i];
    i++;
  }
}

/* ------------------------------------------------------------------ */
/* Nested chunks (raw + RLE)                                           */
/* ------------------------------------------------------------------ */

static int
kk_nested_rle(kk_in_t *in, uint8_t *dst, uint32_t dst_size,
              uint8_t *scratch, size_t scratch_size) {
  uint32_t src_size = (uint32_t)(in->end - in->p);
  const uint8_t *cmd;
  const uint8_t *cmd_end;
  uint8_t *dst_cur = dst;
  uint8_t *dst_end = dst + dst_size;
  uint32_t rle_byte = 0;

  if (src_size == 0) {
    return -1;
  }
  if (src_size == 1) {
    uint32_t b;
    if (kk_in_u8(in, &b) != 0) {
      return -1;
    }
    memset(dst, b, dst_size);
    return 0;
  }

  /* The command buffer may itself start with a nested chunk decoded
   * into scratch, with the remaining raw bytes appended. */
  if (in->p[0] != 0) {
    uint32_t dec;
    if (scratch == NULL || scratch_size < (size_t)dst_size * 2u + 64u) {
      return -1;
    }
    if (kk_nested_chunk_decode(in, scratch, (uint32_t)scratch_size,
                               &dec) != 0) {
      return -1;
    }
    if ((uint32_t)(in->end - in->p) > scratch_size - dec) {
      return -1;
    }
    memcpy(scratch + dec, in->p, (size_t)(in->end - in->p));
    cmd = scratch;
    cmd_end = scratch + dec + (in->end - in->p);
  } else {
    cmd = in->p + 1;
    cmd_end = in->end;
  }

  while (cmd_end > cmd) {
    uint32_t c = cmd_end[-1];
    if (c - 1 >= 0x2Fu) {
      uint32_t ncopy = (~c) & 0xFu;
      uint32_t nrle = c >> 4;
      if (cmd_end - ncopy < cmd ||
          (uint32_t)(dst_end - dst_cur) < ncopy + nrle) {
        return -1;
      }
      cmd_end -= ncopy;
      memcpy(dst_cur, cmd_end, ncopy);
      dst_cur += ncopy;
      memset(dst_cur, rle_byte, nrle);
      dst_cur += nrle;
    } else if (c >= 0x10u) {
      uint32_t data, ncopy, nrle;
      if (cmd_end - cmd < 2) {
        return -1;
      }
      data = (uint32_t)cmd_end[-2] | ((uint32_t)cmd_end[-1] << 8);
      data -= 4096u;
      ncopy = data & 0x3Fu;
      nrle = data >> 6;
      cmd_end -= 2;
      if (cmd_end - cmd < ncopy ||
          (uint32_t)(dst_end - dst_cur) < ncopy + nrle) {
        return -1;
      }
      memcpy(dst_cur, cmd, ncopy);
      dst_cur += ncopy;
      cmd += ncopy;
      memset(dst_cur, rle_byte, nrle);
      dst_cur += nrle;
    } else if (c == 1) {
      if (cmd_end == cmd) {
        return -1;
      }
      rle_byte = *cmd++;
      cmd_end--;
    } else if (c >= 9) {
      uint32_t data, nrle;
      if (cmd_end - cmd < 2) {
        return -1;
      }
      data = (uint32_t)cmd_end[-2] | ((uint32_t)cmd_end[-1] << 8);
      data -= 0x8FFu;
      nrle = data * 128u;
      cmd_end -= 2;
      if ((uint32_t)(dst_end - dst_cur) < nrle) {
        return -1;
      }
      memset(dst_cur, rle_byte, nrle);
      dst_cur += nrle;
    } else { /* 0, 2..8 */
      uint32_t data, ncopy;
      if (cmd_end - cmd < 2) {
        return -1;
      }
      data = (uint32_t)cmd_end[-2] | ((uint32_t)cmd_end[-1] << 8);
      data -= 511u;
      ncopy = data * 64u;
      cmd_end -= 2;
      if (cmd_end - cmd < ncopy ||
          (uint32_t)(dst_end - dst_cur) < ncopy) {
        return -1;
      }
      memcpy(dst_cur, cmd, ncopy);
      dst_cur += ncopy;
      cmd += ncopy;
    }
  }
  if (cmd_end != cmd || dst_cur != dst_end) {
    return -1;
  }
  in->p = in->end;
  return 0;
}

static int
kk_nested_chunk_decode(kk_in_t *in, uint8_t *dst, uint32_t dst_cap,
                       uint32_t *out_dst) {
  uint32_t b0, t, src_size = 0, dst_size = 0;
  const uint8_t *payload;

  if (kk_in_u8(in, &b0) != 0) {
    return -1;
  }
  t = (b0 >> 4) & 7u;

  if (t == 0) {
    if (b0 >= 0x80u) {
      uint32_t b1;
      if (kk_in_u8(in, &b1) != 0) {
        return -1;
      }
      src_size = ((b0 & 0x0F) << 8) | b1;
    } else {
      uint32_t b1, b2;
      if (kk_in_u8(in, &b1) != 0 || kk_in_u8(in, &b2) != 0) {
        return -1;
      }
      src_size = (b0 << 16) | (b1 << 8) | b2;
      if (src_size > 0x3FFFFu) {
        return -1;
      }
    }
    dst_size = src_size;
  } else {
    if (t >= 6) {
      return -1;
    }
    if (b0 >= 0x80u) {
      uint32_t b1, b2, bits;
      if (kk_in_u8(in, &b1) != 0 || kk_in_u8(in, &b2) != 0) {
        return -1;
      }
      bits = ((b0 & 0x0F) << 16) | (b1 << 8) | b2;
      src_size = bits & 0x3FFu;
      dst_size = src_size + ((bits >> 10) & 0x3FFu) + 1u;
    } else {
      uint32_t b1, b2, b3, b4, bits;
      if (kk_in_u8(in, &b1) != 0 || kk_in_u8(in, &b2) != 0 ||
          kk_in_u8(in, &b3) != 0 || kk_in_u8(in, &b4) != 0) {
        return -1;
      }
      bits = (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
      src_size = bits & 0x3FFFFu;
      dst_size = (((bits >> 18) | (b0 << 14)) & 0x3FFFFu) + 1u;
      if (src_size >= dst_size) {
        return -1;
      }
    }
  }

  if (dst_size > dst_cap || kk_in_bytes(in, &payload, src_size) != 0) {
    return -1;
  }

  if (t == 0) {
    memcpy(dst, payload, src_size);
    *out_dst = dst_size;
    return 0;
  }
  if (t == 3) {
    kk_in_t rle_in = {payload, payload + src_size};
    if (kk_nested_rle(&rle_in, dst, dst_size, NULL, 0) != 0) {
      return -1;
    }
    *out_dst = dst_size;
    return 0;
  }

  PAI_LOG_ERROR_(PAI_SUB_CORE, "compress: nested chunk type %u unsupported\n",
                 t);
  return -1;
}

/* ------------------------------------------------------------------ */
/* Mermaid / Selkie quantum                                            */
/* ------------------------------------------------------------------ */

typedef struct {
  const uint8_t *cmd;
  const uint8_t *cmd_split; /* start of the second half's commands    */
  const uint8_t *cmd_end;
  const uint8_t *lit;
  const uint8_t *lit_end;
  const uint8_t *len;
  const uint8_t *len_end;
  const uint8_t *off16;
  const uint8_t *off16_end;
  uint32_t *off32;      /* first half                                */
  uint32_t *off32_end;  /* end of first half                          */
  uint32_t *off32_2;    /* second half                                */
  uint32_t *off32_2_end;
} kk_lz_t;

static int
kk_lz_read(kk_in_t *in, kk_lz_t *lz, uint32_t dst_count, uint64_t out_offs,
           uint8_t *scratch, size_t scratch_size, uint8_t *dst) {
  uint32_t dec, used = 0, off16_count, tmp;
  uint32_t size1 = 0, size2 = 0;
  uint32_t i;
  uint64_t base;

  if (out_offs == 0) {
    const uint8_t *raw;
    if (kk_in_bytes(in, &raw, 8) != 0) {
      return -1;
    }
    memcpy(dst, raw, 8);
  }

  /* Literal stream. */
  if (kk_nested_chunk_decode(in, scratch, (uint32_t)scratch_size, &dec) !=
      0) {
    return -1;
  }
  lz->lit = scratch;
  lz->lit_end = scratch + dec;
  used += dec;

  /* Command stream. */
  if (kk_nested_chunk_decode(in, scratch + used,
                             (uint32_t)scratch_size - used, &dec) != 0) {
    return -1;
  }
  lz->cmd = scratch + used;
  lz->cmd_end = lz->cmd + dec;
  lz->cmd_split = lz->cmd_end;
  used += dec;

  if (dst_count > KK_HALF_OUT) {
    uint32_t split;
    if (kk_in_u16(in, &split) != 0 || split > dec) {
      return -1;
    }
    lz->cmd_split = lz->cmd + split;
  }

  if (kk_in_u16(in, &off16_count) != 0) {
    return -1;
  }
  if (off16_count == 0xFFFFu) {
    PAI_LOG_ERROR_(PAI_SUB_CORE,
                   "compress: entropy-coded off16 streams unsupported\n");
    return -1;
  }
  if (kk_in_bytes(in, &lz->off16, off16_count * 2u) != 0) {
    return -1;
  }
  lz->off16_end = lz->off16 + off16_count * 2u;

  if (kk_in_u24(in, &tmp) != 0) {
    return -1;
  }
  if (tmp != 0) {
    size1 = tmp >> 12;
    size2 = tmp & 0xFFFu;
    if (size1 == 4095u) {
      uint32_t x;
      if (kk_in_u16(in, &x) != 0) {
        return -1;
      }
      size1 = x;
    }
    if (size2 == 4095u) {
      uint32_t x;
      if (kk_in_u16(in, &x) != 0) {
        return -1;
      }
      size2 = x;
    }
  }

  if (scratch_size - used < (size_t)(size1 + size2) * 4u + 64u) {
    return -1;
  }
  lz->off32 = (uint32_t *)(scratch + used);
  lz->off32_end = lz->off32 + size1;
  lz->off32_2 = lz->off32_end;
  lz->off32_2_end = lz->off32_2 + size2;

  base = out_offs;
  for (i = 0; i < size1; i++) {
    uint32_t off;
    if (kk_in_u24(in, &off) != 0) {
      return -1;
    }
    if (base >= (0xC00000u - 1u)) {
      if (off >= 0xC00000u) {
        uint32_t hi;
        if (kk_in_u8(in, &hi) != 0) {
          return -1;
        }
        off += hi << 22;
      }
    }
    if (off > base) {
      return -1;
    }
    lz->off32[i] = off;
  }
  base = out_offs + KK_HALF_OUT;
  for (i = 0; i < size2; i++) {
    uint32_t off;
    if (kk_in_u24(in, &off) != 0) {
      return -1;
    }
    if (base >= (0xC00000u - 1u)) {
      if (off >= 0xC00000u) {
        uint32_t hi;
        if (kk_in_u8(in, &hi) != 0) {
          return -1;
        }
        off += hi << 22;
      }
    }
    if (off > base) {
      return -1;
    }
    lz->off32_2[i] = off;
  }

  lz->len = in->p;
  lz->len_end = in->end;
  return 0;
}

static int
kk_runs(kk_lz_t *lz, uint8_t *dst, uint32_t dst_count, uint8_t *chunk_start,
        const uint8_t *stream_base, int mode, uint32_t startoff,
        int32_t *saved_recent) {
  uint8_t *dst_cur = dst + startoff;
  uint8_t *dst_end = dst + dst_count;
  const uint8_t *cmd = lz->cmd;
  const uint8_t *cmd_end = lz->cmd_end;
  const uint8_t *lit = lz->lit;
  const uint8_t *lit_end = lz->lit_end;
  const uint8_t *len = lz->len;
  const uint8_t *len_end = lz->len_end;
  const uint8_t *off16 = lz->off16;
  const uint8_t *off16_end = lz->off16_end;
  uint32_t *off32 = lz->off32;
  uint32_t *off32_end = lz->off32_end;
  int32_t recent = *saved_recent;

  while (cmd < cmd_end) {
    uint32_t c = *cmd++;
    if (c >= 24) {
      uint32_t litlen = c & 7u;
      uint32_t matchlen = (c >> 3) & 0xFu;
      if (c & 0x80u) {
        if (off16_end - off16 < 2) {
          return -1;
        }
        recent = -(int32_t)((uint32_t)off16[0] | ((uint32_t)off16[1] << 8));
        off16 += 2;
      }
      if ((uint32_t)(dst_end - dst_cur) < litlen + matchlen ||
          (uint32_t)(lit_end - lit) < litlen ||
          dst_cur + recent < stream_base) {
        return -1;
      }
      if (mode == 1) {
        memcpy(dst_cur, lit, litlen);
      } else {
        uint32_t i;
        for (i = 0; i < litlen; i++) {
          dst_cur[i] = lit[i] + dst_cur[recent];
        }
      }
      dst_cur += litlen;
      lit += litlen;
      kk_copy_match(dst_cur, dst_cur + recent, matchlen);
      dst_cur += matchlen;
    } else if (c > 2) {
      uint32_t matchlen = c + 5;
      const uint8_t *match;
      if (off32 == off32_end) {
        return -1;
      }
      match = chunk_start - *off32++;
      recent = (int32_t)(match - dst_cur);
      if (match < chunk_start ||
          (uint32_t)(dst_end - dst_cur) < matchlen) {
        return -1;
      }
      kk_copy_match(dst_cur, match, matchlen);
      dst_cur += matchlen;
    } else if (c == 0) {
      uint32_t length;
      if (len >= len_end) {
        return -1;
      }
      length = *len++;
      if (length > 251) {
        if (len_end - len < 2) {
          return -1;
        }
        length += 4u * ((uint32_t)len[0] | ((uint32_t)len[1] << 8));
        len += 2;
      }
      length += 64;
      if ((uint32_t)(dst_end - dst_cur) < length ||
          (uint32_t)(lit_end - lit) < length) {
        return -1;
      }
      if (mode == 1) {
        memcpy(dst_cur, lit, length);
      } else {
        uint32_t i;
        for (i = 0; i < length; i++) {
          dst_cur[i] = lit[i] + dst_cur[recent];
        }
      }
      dst_cur += length;
      lit += length;
    } else if (c == 1) {
      uint32_t length;
      if (len >= len_end) {
        return -1;
      }
      length = *len++;
      if (length > 251) {
        if (len_end - len < 2) {
          return -1;
        }
        length += 4u * ((uint32_t)len[0] | ((uint32_t)len[1] << 8));
        len += 2;
      }
      length += 91;
      if (off16_end - off16 < 2) {
        return -1;
      }
      recent = -(int32_t)((uint32_t)off16[0] | ((uint32_t)off16[1] << 8));
      off16 += 2;
      if (dst_cur + recent < stream_base ||
          (uint32_t)(dst_end - dst_cur) < length) {
        return -1;
      }
      kk_copy_match(dst_cur, dst_cur + recent, length);
      dst_cur += length;
    } else { /* c == 2 */
      uint32_t length;
      const uint8_t *match;
      if (len >= len_end) {
        return -1;
      }
      length = *len++;
      if (length > 251) {
        if (len_end - len < 2) {
          return -1;
        }
        length += 4u * ((uint32_t)len[0] | ((uint32_t)len[1] << 8));
        len += 2;
      }
      length += 29;
      if (off32 == off32_end) {
        return -1;
      }
      match = chunk_start - *off32++;
      recent = (int32_t)(match - dst_cur);
      if (match < chunk_start ||
          (uint32_t)(dst_end - dst_cur) < length) {
        return -1;
      }
      kk_copy_match(dst_cur, match, length);
      dst_cur += length;
    }
  }

  /* Tail: the rest of this half comes from the literal stream. */
  {
    uint32_t length = (uint32_t)(dst_end - dst_cur);
    if ((uint32_t)(lit_end - lit) < length) {
      return -1;
    }
    if (mode == 1) {
      memcpy(dst_cur, lit, length);
    } else {
      uint32_t i;
      for (i = 0; i < length; i++) {
        dst_cur[i] = lit[i] + dst_cur[recent];
      }
    }
    lit += length;
  }

  lz->len = len;
  lz->off16 = off16;
  lz->lit = lit;
  *saved_recent = recent;
  return 0;
}

static int
kk_mermaid_quantum_decode(kk_in_t *in, uint8_t *dst, uint32_t dst_count,
                          uint64_t out_offs, uint8_t *scratch,
                          size_t scratch_size) {
  uint32_t produced = 0;

  while (produced < dst_count) {
    uint32_t chunk = dst_count - produced;
    uint32_t h, src_used, mode;

    if (chunk > KK_CHUNK_OUT) {
      chunk = KK_CHUNK_OUT;
    }
    if (kk_in_u24_be(in, &h) != 0) {
      return -1;
    }
    if (!(h & 0x800000u)) {
      uint32_t dec;
      if (kk_nested_chunk_decode(in, dst + produced, chunk, &dec) != 0 ||
          dec != chunk) {
        return -1;
      }
    } else {
      src_used = h & 0x7FFFFu;
      mode = (h >> 19) & 0xFu;
      if (mode > 1 || (uint32_t)(in->end - in->p) < src_used) {
        return -1;
      }
      if (src_used == chunk && mode == 0) {
        const uint8_t *raw;
        if (kk_in_bytes(in, &raw, src_used) != 0) {
          return -1;
        }
        memcpy(dst + produced, raw, src_used);
      } else {
        kk_in_t qin = {in->p, in->p + src_used};
        kk_lz_t lz;
        int32_t saved = -8;
        uint32_t half1, half2;
        int startoff;

        memset(&lz, 0, sizeof(lz));
        if (kk_lz_read(&qin, &lz, chunk, out_offs + produced, scratch,
                       scratch_size, dst + produced) != 0) {
          return -1;
        }

        half1 = chunk > KK_HALF_OUT ? KK_HALF_OUT : chunk;
        half2 = chunk - half1;
        startoff = (out_offs + produced == 0) ? 8 : 0;

        {
          kk_lz_t half = lz;
          half.cmd_end = lz.cmd_split;
          half.off32_end = lz.off32_end;
          if (kk_runs(&half, dst + produced, half1, dst + produced,
                      dst - out_offs, (int)mode, (uint32_t)startoff,
                      &saved) != 0) {
            return -1;
          }
          lz.len = half.len;
          lz.off16 = half.off16;
          lz.lit = half.lit;
        }
        if (half2 > 0) {
          kk_lz_t half = lz;
          half.cmd = lz.cmd_split;
          half.cmd_end = lz.cmd_end;
          half.off32 = lz.off32_2;
          half.off32_end = lz.off32_2_end;
          if (kk_runs(&half, dst + produced + half1, half2, dst + produced,
                      dst - out_offs, (int)mode, 0, &saved) != 0) {
            return -1;
          }
          lz.len = half.len;
          lz.off16 = half.off16;
          lz.lit = half.lit;
        }
        /* every length-stream byte must have been consumed by the runs */
        if (lz.len != lz.len_end) {
          return -1;
        }
        in->p += src_used;
      }
    }
    produced += chunk;
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Container                                                           */
/* ------------------------------------------------------------------ */

static int
kk_container_decode(kk_in_t *in, uint8_t *dst, uint64_t dst_len,
                    uint8_t *scratch, size_t scratch_size) {
  uint64_t out_offs = 0;
  uint32_t method = 0, uncompressed = 0, checksums = 0;

  while (out_offs < dst_len) {
    uint64_t quantum = dst_len - out_offs;
    uint32_t v, comp_size, flag1, flag2;

    if (quantum > KK_QUANTUM_OUT) {
      quantum = KK_QUANTUM_OUT;
    }

    if ((out_offs & 0x3FFFFu) == 0) {
      uint32_t b0, b1;
      if (kk_in_u8(in, &b0) != 0 || kk_in_u8(in, &b1) != 0) {
        return -1;
      }
      if ((b0 & 0xFu) != 0xCu || ((b0 >> 4) & 3u) != 0) {
        return -1;
      }
      method = b1 & 0x7Fu;
      uncompressed = (b0 >> 6) & 1u;
      checksums = b1 >> 7;
      if (method != 6 && method != 10 && method != 12 && method != 5 &&
          method != 11) {
        return -1;
      }
    }

    if (uncompressed) {
      const uint8_t *raw;
      if (kk_in_bytes(in, &raw, (uint32_t)quantum) != 0) {
        return -1;
      }
      memcpy(dst + out_offs, raw, (size_t)quantum);
      out_offs += quantum;
      continue;
    }

    if (kk_in_u24_be(in, &v) != 0) {
      return -1;
    }
    comp_size = v & 0x3FFFFu;
    flag1 = (v >> 18) & 1u;
    flag2 = (v >> 19) & 1u;

    if (comp_size == 0x3FFFFu) {
      uint32_t kind = v >> 18;
      if (kind == 1) {
        uint32_t fill;
        if (kk_in_u8(in, &fill) != 0) {
          return -1;
        }
        memset(dst + out_offs, fill, (size_t)quantum);
        out_offs += quantum;
        continue;
      }
      if (kind == 3) {
        /* PAI extension: raw full 0x40000 quantum */
        comp_size = 0x40000u;
      } else {
        return -1; /* whole-match quanta unsupported */
      }
    } else {
      if (flag1 || flag2) {
        return -1; /* whole-match quanta unsupported */
      }
      comp_size += 1;
    }

    if (checksums) {
      uint32_t csum;
      if (kk_in_u24(in, &csum) != 0) {
        return -1;
      }
      (void)csum; /* not verified (v1) */
    }
    if ((uint32_t)(in->end - in->p) < comp_size || comp_size > quantum) {
      return -1;
    }

    if (comp_size == (uint32_t)quantum) {
      const uint8_t *raw;
      if (kk_in_bytes(in, &raw, comp_size) != 0) {
        return -1;
      }
      memcpy(dst + out_offs, raw, comp_size);
    } else {
      kk_in_t qin = {in->p, in->p + comp_size};
      int r;
      if (method == 10) {
        r = kk_mermaid_quantum_decode(&qin, dst + out_offs,
                                      (uint32_t)quantum, out_offs, scratch,
                                      scratch_size);
      } else {
        PAI_LOG_ERROR_(PAI_SUB_CORE, "compress: method %u unsupported\n",
                       method);
        return -1;
      }
      if (r != 0) {
        return -1;
      }
      in->p += comp_size;
    }
    out_offs += quantum;
  }
  return 0;
}

int
kk_decode_stream(const uint8_t *src, uint64_t len, uint8_t *dst,
                 uint64_t dst_len) {
  kk_in_t in = {src, src + len};
  size_t scratch_size = 2u * KK_CHUNK_OUT + 0x100000u;
  uint8_t *scratch = (uint8_t *)malloc(scratch_size);
  int r;

  if (scratch == NULL) {
    return -1;
  }
  r = kk_container_decode(&in, dst, dst_len, scratch, scratch_size);
  free(scratch);
  if (r != 0 || in.p != in.end) {
    return -1;
  }
  return 0;
}
