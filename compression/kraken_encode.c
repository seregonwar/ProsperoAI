/*
 * ProsperoAI — Kraken-family encoder (clean-room, see pai_compress.h)
 *
 * Our own greedy LZ77 encoder producing a valid Kraken container with
 * Mermaid-method quanta (mode 1, plain literals, raw nested chunks).
 * Not an Oodle compressor: it trades some ratio for a small,
 * dependency-free implementation.
 *
 * Encoding choices, kept deliberately simple:
 *   - literals are appended to the literal stream as encountered; a
 *     command's 3-bit litlen field or a cmd == 0 run (64+ bytes)
 *     "claims" them; anything left is the half's tail
 *   - a match is taken only when <= 7 literals are unclaimed
 *   - distances <= 0xFFFF: cmd >= 24 (1..15 bytes) or cmd == 1 (91+)
 *   - larger distances (chunk-local): cmd > 2 (5..20) or cmd == 2 (29+)
 *   - every match command carries its distance explicitly
 */

#include "pai_compress.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KE_QUANTUM 0x40000u
#define KE_CHUNK 0x20000u
#define KE_HALF 0x10000u
#define KE_MIN_MATCH 5u
#define KE_HASH_BITS 18u
#define KE_HASH_SIZE (1u << KE_HASH_BITS)
#define KE_WINDOW 0xFFFFu

typedef struct {
  uint8_t *p;
  uint32_t n;
  uint32_t cap;
} ke_buf_t;

static int
ke_put(ke_buf_t *b, uint8_t v) {
  if (b->n >= b->cap) {
    return -1;
  }
  b->p[b->n++] = v;
  return 0;
}

static int
ke_put16(ke_buf_t *b, uint32_t v) {
  return ke_put(b, (uint8_t)(v & 0xFF)) || ke_put(b, (uint8_t)(v >> 8));
}

static int
ke_put24(ke_buf_t *b, uint32_t v) {
  return ke_put(b, (uint8_t)(v & 0xFF)) ||
         ke_put(b, (uint8_t)((v >> 8) & 0xFF)) ||
         ke_put(b, (uint8_t)(v >> 16));
}

/* Raw nested chunk (type 0). Sizes are stored big-endian (high byte
 * first): 12-bit form with bit7 of the first byte set, 18-bit form
 * otherwise. */
static int
ke_raw_chunk(ke_buf_t *b, const uint8_t *data, uint32_t n) {
  if (n <= 0xFFFu) {
    if (ke_put(b, (uint8_t)(0x80u | (n >> 8))) != 0 ||
        ke_put(b, (uint8_t)(n & 0xFF)) != 0) {
      return -1;
    }
  } else {
    if (n > 0x3FFFFu || ke_put(b, (uint8_t)(n >> 16)) != 0 ||
        ke_put(b, (uint8_t)(n >> 8)) != 0 || ke_put(b, (uint8_t)n) != 0) {
      return -1;
    }
  }
  if (b->cap - b->n < n) {
    return -1;
  }
  memcpy(b->p + b->n, data, n);
  b->n += n;
  return 0;
}

/* Length-stream value: L = len - base. L <= 251 is one byte; larger
 * lengths use byte b in 252..255 followed by u16 k with L = b + 4k
 * (the decoder reads the extension only for bytes > 251). Returns the
 * actual length emitted (the largest encodable <= len). */
static uint32_t
ke_len_emit(ke_buf_t *lb, uint32_t len, uint32_t base) {
  uint32_t L = len - base;
  if (L > 251u) {
    uint32_t b = 252u + ((L - 252u) & 3u);
    uint32_t k = (L - b) / 4u;
    L = b + 4u * k;
    ke_put(lb, (uint8_t)b);
    ke_put16(lb, k);
  } else {
    ke_put(lb, (uint8_t)L);
  }
  return base + L;
}

typedef struct {
  const uint8_t *src;
  uint64_t src_len;
  int32_t table[KE_HASH_SIZE];
  int32_t *chain;
  ke_buf_t lit, cmd, off16, off32, lenb;
  uint32_t lit_claimed; /* literals already covered by commands */
  uint32_t lit_chunk0;  /* lit.n at the current chunk start */
  uint32_t cmd_chunk0;
  uint32_t off16_chunk0;
  uint32_t off32_chunk0;
  uint32_t lenb_chunk0;
  uint32_t cmd_half1;
  uint32_t off32_half1;
} ke_enc_t;

static uint32_t
ke_hash4(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return (v * 0x9E3779B1u) >> (32 - KE_HASH_BITS);
}

static uint32_t
ke_pending(ke_enc_t *e) {
  return e->lit.n - e->lit_claimed;
}

/* Flush accumulated literals via cmd == 0 runs. Runs must not cross
 * the current half's end (each half decodes independently); when the
 * remaining half space is < 64 the pending literals ride the tail. */
static int
ke_flush_lits(ke_enc_t *e, uint64_t pos, uint64_t half_end) {
  while (ke_pending(e) >= 64u && half_end - pos >= 64u) {
    uint32_t run = ke_pending(e);
    if (run > half_end - pos) {
      run = (uint32_t)(half_end - pos);
    }
    if (run > 64u + 251u) {
      run = 64u + 251u;
    }
    if (ke_put(&e->cmd, 0u) != 0) {
      return -1;
    }
    ke_len_emit(&e->lenb, run, 64u);
    e->lit_claimed += run;
  }
  return 0;
}

/* Find the longest match at pos. *out_q receives the match position. */
static int
ke_find(ke_enc_t *e, uint64_t pos, uint32_t max_len, uint32_t *out_len,
        uint64_t *out_q) {
  uint32_t hash, best = 0, walk = 0;
  int32_t cand;
  uint64_t best_q = 0;

  if (max_len < KE_MIN_MATCH) {
    return 0;
  }
  hash = ke_hash4(e->src + pos);
  cand = e->table[hash];
  while (cand >= 0 && walk < 64) {
    uint64_t q = (uint64_t)cand;
    uint64_t dist = pos - q;
    uint32_t n = 0;
    if (dist == 0 || dist > 0xFFFFFFFFu) {
      break;
    }
    while (n < max_len && e->src[q + n] == e->src[pos + n]) {
      n++;
    }
    if (n > best) {
      best = n;
      best_q = q;
      if (n >= max_len) {
        break;
      }
    }
    cand = e->chain[cand];
    walk++;
  }
  if (best < KE_MIN_MATCH) {
    return 0;
  }
  *out_len = best;
  *out_q = best_q;
  return 1;
}

/* Emit one match. Returns the consumed length. */
static int
ke_emit_match(ke_enc_t *e, uint64_t pos, uint64_t q, uint32_t mlen,
              uint64_t chunk_start, uint32_t *out_taken) {
  uint32_t dist = (uint32_t)(pos - q);
  uint32_t litlen = ke_pending(e);
  int far = dist > KE_WINDOW;

  if (pos >= 0x40000u && pos < 0x42000u) {
    fprintf(stderr, "DBGENC1 pos=%llx mlen=%u dist=%u litlen=%u\n",
            (unsigned long long)pos, mlen, dist, litlen);
  }

  if (far && q < chunk_start) {
    return 0; /* not encodable */
  }

  if (!far) {
    if (mlen <= 15u) {
      /* the command encodes matchlen - 3 in bits 3..6 */
      if (ke_put(&e->cmd,
                 (uint8_t)(0x80u | (mlen << 3) | litlen)) !=
          0) {
        return -1;
      }
      ke_put16(&e->off16, dist);
      e->lit_claimed += litlen;
      *out_taken = mlen;
      return 1;
    }
    if (mlen >= 91u) {
      if (ke_put(&e->cmd, 1u) != 0) {
        return -1;
      }
      mlen = ke_len_emit(&e->lenb, mlen, 91u);
      ke_put16(&e->off16, dist);
      e->lit_claimed += litlen;
      *out_taken = mlen;
      return 1;
    }
    /* 16..90: fall through to far forms when chunk-local, else
     * truncate to 15. */
    if (q < chunk_start) {
      mlen = 15u;
      if (ke_put(&e->cmd,
                 (uint8_t)(0x80u | (mlen << 3) | litlen)) !=
          0) {
        return -1;
      }
      ke_put16(&e->off16, dist);
      e->lit_claimed += litlen;
      *out_taken = mlen;
      return 1;
    }
    far = 1;
  }

  /* Far forms: chunk-relative offset. */
  {
    uint32_t off = (uint32_t)(chunk_start - q);
    uint64_t base = chunk_start;
    if (mlen >= 5u && mlen <= 20u) {
      if (ke_put(&e->cmd, (uint8_t)(3u + mlen - 5u)) != 0) {
        return -1;
      }
    } else if (mlen >= 29u) {
      if (ke_put(&e->cmd, 2u) != 0) {
        return -1;
      }
      mlen = ke_len_emit(&e->lenb, mlen, 29u);
    } else {
      mlen = 20u;
      if (ke_put(&e->cmd, (uint8_t)(3u + mlen - 5u)) != 0) {
        return -1;
      }
    }
    if (base >= (0xC00000u - 1u) && off >= 0xC00000u) {
      ke_put24(&e->off32, off);
      ke_put(&e->off32, (uint8_t)(off >> 22));
    } else {
      ke_put24(&e->off32, off);
    }
    e->lit_claimed += litlen;
    *out_taken = mlen;
    return 1;
  }
}

static int
ke_encode_chunk(ke_enc_t *e, ke_buf_t *q, uint64_t chunk_start,
                uint32_t dst_count, uint64_t out_offs) {
  uint64_t p = chunk_start + ((out_offs == 0) ? 8u : 0u);
  uint64_t end = chunk_start + dst_count;
  uint32_t chunk_n0 = q->n;
  uint32_t half = 0;
  int r;

  /* too small for the 8-byte stream prefix: store the chunk raw */
  if (out_offs == 0 && dst_count < 8u) {
    ke_buf_t w = {q->p + q->n, 0, q->cap - q->n};
    if (ke_put24(&w, 0) != 0 ||
        ke_raw_chunk(&w, e->src + chunk_start, dst_count) != 0) {
      return -1;
    }
    q->n += w.n;
    return 0;
  }

  e->lit_chunk0 = e->lit.n;
  e->cmd_chunk0 = e->cmd.n;
  e->off16_chunk0 = e->off16.n;
  e->off32_chunk0 = e->off32.n;
  e->lenb_chunk0 = e->lenb.n;
  e->cmd_half1 = e->cmd.n;
  e->off32_half1 = e->off32.n;
  /* each chunk owns its literal range: the previous chunk's tail is
   * already part of that chunk's stream */
  e->lit_claimed = e->lit.n;

  while (p < end) {
    uint32_t mlen, taken;
    uint64_t mq;
    uint64_t half_end;

    if (half == 0 && p - chunk_start >= KE_HALF) {
      half = 1;
      e->cmd_half1 = e->cmd.n;
      e->off32_half1 = e->off32.n;
    }
    half_end = (half == 0) ? chunk_start + KE_HALF : end;

    if (ke_pending(e) <= 7u &&
        ke_find(e, p, (uint32_t)(end - p), &mlen, &mq)) {
      /* matches must not cross the half boundary */
      if (mlen > half_end - p) {
        mlen = (uint32_t)(half_end - p);
      }
      if (mlen >= KE_MIN_MATCH) {
        r = ke_emit_match(e, p, mq, mlen, chunk_start, &taken);
        if (r < 0) {
          return -1;
        }
        if (r > 0) {
          p += taken;
          continue;
        }
      }
    }

    /* literal: append enough to make cmd == 0 runs profitable, or at
     * least one byte */
    {
      uint64_t n = p + 1;
      while (n < end && ke_pending(e) + (uint32_t)(n - p) < 64u) {
        n++;
      }
      if (e->lit.cap - e->lit.n < (uint32_t)(n - p)) {
        return -1;
      }
      memcpy(e->lit.p + e->lit.n, e->src + p, (size_t)(n - p));
      e->lit.n += (uint32_t)(n - p);
      p = n;
      if (ke_flush_lits(e, p, half_end) != 0) {
        return -1;
      }
    }
  }

  /* Single-half chunks: everything belongs to the first half. */
  if (dst_count <= KE_HALF) {
    e->cmd_half1 = e->cmd.n;
    e->off32_half1 = e->off32.n;
  }

  /* Assemble the chunk payload (after a 3-byte header slot). */
  {
    uint32_t lit_n = e->lit.n - e->lit_chunk0;
    uint32_t cmd_n = e->cmd.n - e->cmd_chunk0;
    uint32_t off16_n = e->off16.n - e->off16_chunk0;
    uint32_t off32_n = e->off32.n - e->off32_chunk0;
    uint32_t lenb_n = e->lenb.n - e->lenb_chunk0;
    uint32_t payload;
    ke_buf_t w = {q->p + q->n + 3u, 0, q->cap - q->n - 3u};

    /* the first 8 bytes of the whole stream are stored raw in the
     * payload; the LZ loop starts after them */
    if (out_offs == 0) {
      if (w.cap - w.n < 8u) {
        return -1;
      }
      memcpy(w.p + w.n, e->src + chunk_start, 8u);
      w.n += 8u;
    }

    if (ke_raw_chunk(&w, e->lit.p + e->lit_chunk0, lit_n) != 0 ||
        ke_raw_chunk(&w, e->cmd.p + e->cmd_chunk0, cmd_n) != 0) {
      return -1;
    }
    if (dst_count > KE_HALF) {
      ke_put16(&w, e->cmd_half1 - e->cmd_chunk0);
    }
    ke_put16(&w, off16_n / 2u);
    if (w.cap - w.n < off16_n) {
      return -1;
    }
    memcpy(w.p + w.n, e->off16.p + e->off16_chunk0, off16_n);
    w.n += off16_n;

    if (off32_n != 0) {
      ke_put24(&w, (((e->off32_half1 - e->off32_chunk0) / 3u) << 12) |
                        ((off32_n - (e->off32_half1 - e->off32_chunk0)) /
                         3u));
      if (w.cap - w.n < off32_n) {
        return -1;
      }
      memcpy(w.p + w.n, e->off32.p + e->off32_chunk0, off32_n);
      w.n += off32_n;
    } else {
      ke_put24(&w, 0);
    }

    if (w.cap - w.n < lenb_n) {
      return -1;
    }
    memcpy(w.p + w.n, e->lenb.p + e->lenb_chunk0, lenb_n);
    w.n += lenb_n;

    payload = w.n;
    if (payload >= dst_count) {
      /* incompressible chunk: raw nested chunk form (bit23 == 0) */
      q->n = chunk_n0;
      w.p = q->p + q->n;
      w.n = 0;
      w.cap = q->cap - q->n;
      if (ke_put24(&w, 0) != 0) {
        return -1;
      }
      if (ke_raw_chunk(&w, e->src + chunk_start, dst_count) != 0) {
        return -1;
      }
      q->n += w.n;
      return 0;
    }

    /* chunk header: bit23 set, mode 1, src_used = payload */
    {
      uint8_t h0 = (uint8_t)(0x80u | (1u << 3) | (payload >> 16));
      if (ke_put(q, h0) != 0 || ke_put(q, (uint8_t)(payload >> 8)) != 0 ||
          ke_put(q, (uint8_t)payload) != 0) {
        return -1;
      }
    }
    q->n += w.n;
  }
  return 0;
}

int
kk_encode_stream(const uint8_t *src, uint64_t len, ke_buf_t *out) {
  ke_enc_t *e;
  uint64_t pos = 0;

  e = (ke_enc_t *)calloc(1, sizeof(*e));
  if (e == NULL) {
    return -1;
  }
  e->src = src;
  e->src_len = len;
  e->chain = (int32_t *)malloc((size_t)len * sizeof(int32_t));
  if (e->chain == NULL) {
    free(e);
    return -1;
  }
  {
    uint32_t i;
    for (i = 0; i < KE_HASH_SIZE; i++) {
      e->table[i] = -1;
    }
  }
  e->lit.cap = KE_QUANTUM + 64u;
  e->cmd.cap = KE_QUANTUM + 64u;
  e->off16.cap = 2u * KE_QUANTUM + 64u;
  e->off32.cap = KE_QUANTUM + 64u;
  e->lenb.cap = KE_QUANTUM + 64u;
  e->lit.p = (uint8_t *)malloc(e->lit.cap);
  e->cmd.p = (uint8_t *)malloc(e->cmd.cap);
  e->off16.p = (uint8_t *)malloc(e->off16.cap);
  e->off32.p = (uint8_t *)malloc(e->off32.cap);
  e->lenb.p = (uint8_t *)malloc(e->lenb.cap);
  if (!e->lit.p || !e->cmd.p || !e->off16.p || !e->off32.p || !e->lenb.p) {
    goto fail;
  }

  while (pos < len) {
    uint64_t quantum = len - pos;
    uint32_t qlen;
    uint32_t produced = 0;
    ke_buf_t q;

    if (quantum > KE_QUANTUM) {
      quantum = KE_QUANTUM;
    }
    qlen = (uint32_t)quantum;

    /* the 2-byte container header repeats at every 0x40000 boundary */
    if ((pos & 0x3FFFFu) == 0) {
      if (ke_put(out, 0x0Cu) != 0 || ke_put(out, 10u) != 0) {
        goto fail;
      }
    }

    q.p = out->p + out->n + 3u; /* room for the quantum header */
    q.n = 0;
    q.cap = out->cap - out->n - 3u;

    e->lit.n = 0;
    e->cmd.n = 0;
    e->off16.n = 0;
    e->off32.n = 0;
    e->lenb.n = 0;
    e->lit_claimed = 0;

    while (produced < qlen) {
      uint32_t chunk = qlen - produced;
      if (chunk > KE_CHUNK) {
        chunk = KE_CHUNK;
      }
      if (ke_encode_chunk(e, &q, pos + produced, chunk,
                          pos + produced) != 0) {
        goto fail;
      }
      produced += chunk;
    }

    if (q.n >= qlen) {
      if (qlen == KE_QUANTUM) {
        /* PAI extension: raw full 0x40000 quantum (the plain size
         * field collides with the special marker) */
        uint8_t hdr[3] = {0x0Fu, 0xFFu, 0xFFu};
        memcpy(out->p + out->n, hdr, 3);
        out->n += 3;
      } else {
        uint8_t hdr[3];
        hdr[0] = (uint8_t)((qlen - 1u) >> 16);
        hdr[1] = (uint8_t)((qlen - 1u) >> 8);
        hdr[2] = (uint8_t)(qlen - 1u);
        memcpy(out->p + out->n, hdr, 3);
        out->n += 3;
      }
      if (out->cap - out->n < qlen) {
        goto fail;
      }
      memcpy(out->p + out->n, src + pos, qlen);
      out->n += qlen;
    } else {
      uint8_t hdr[3];
      hdr[0] = (uint8_t)((q.n - 1u) >> 16);
      hdr[1] = (uint8_t)((q.n - 1u) >> 8);
      hdr[2] = (uint8_t)(q.n - 1u);
      memcpy(out->p + out->n, hdr, 3);
      out->n += 3 + q.n;
    }

    /* hash the chunk window for future matches */
    {
      uint64_t hp = pos;
      uint64_t hend = pos + quantum;
      for (; hp + 4 <= hend; hp++) {
        uint32_t h = ke_hash4(src + hp);
        e->chain[hp] = e->table[h];
        e->table[h] = (int32_t)hp;
      }
    }

    pos += quantum;
  }

  free(e->lit.p);
  free(e->cmd.p);
  free(e->off16.p);
  free(e->off32.p);
  free(e->lenb.p);
  free(e->chain);
  free(e);
  return 0;

fail:
  free(e->lit.p);
  free(e->cmd.p);
  free(e->off16.p);
  free(e->off32.p);
  free(e->lenb.p);
  free(e->chain);
  free(e);
  return -1;
}
