/*
 * test_fuzz.c — deterministic hostile-input fuzzer for the Prospero
 * Protocol connection layer (whitepaper §38 "fuzzing" in Host CI).
 *
 * Three in-process surfaces over the pipe transport:
 *   A) codec fuzz      — every message decoder hammered with random
 *                        and structured byte patterns
 *   B) raw wire fuzz   — byte-mutated frames and random blobs fed
 *                        into a fresh server connection; invariant:
 *                        a PROTOCOL rejection must land the conn in
 *                        CLOSED and nothing else may happen
 *   C) structured fuzz — a negotiated pair where the client sends
 *                        hostile (msg, flags, payload, ids) combos
 *                        through the OPEN connection, covering the
 *                        stream state machine, misplaced negotiation,
 *                        version mismatches and the auto-ERROR reply
 *
 * The PRNG is seeded (argv[1], default fixed) so any crash is
 * reproducible; argv[2] scales the raw-wire iteration count (default
 * 25000). The host-sanitized preset (ASan+UBSan) is the crash oracle.
 */

#include "test.h"

#include <protocol/protocol.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* seeded PRNG (xorshift64*)                                           */
/* ------------------------------------------------------------------ */

static uint64_t g_rng;

static uint64_t
rnd(void) {
  uint64_t x = g_rng;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  g_rng = x;
  return x * 0x2545F4914F6CDD1Dull;
}

static uint8_t
rnd_byte(void) {
  return (uint8_t)(rnd() >> 32);
}

static uint32_t
rnd_u32(void) {
  return (uint32_t)(rnd() >> 16);
}

static size_t
rnd_len(size_t max) {
  return (size_t)(rnd() % (max + 1));
}

static void
rnd_fill(uint8_t *out, size_t n) {
  for (size_t i = 0; i < n; i++) {
    out[i] = rnd_byte();
  }
}

static uint32_t
accept_hello(void *user, uint32_t remote_caps) {
  (void)user;
  (void)remote_caps;
  return PAI_PROTO_CAP_KNOWN;
}

/* Pump until idle or closed; returns the last poll status. */
static pai_status_t
pump_conn(pai_proto_conn_t *c) {
  uint32_t frames = 0;
  for (int i = 0; i < 64; i++) {
    pai_status_t st = pai_proto_conn_poll(c, &frames);
    if (st != PAI_OK) {
      return st;
    }
    if (pai_proto_conn_state(c) == PAI_PROTO_STATE_CLOSED || frames == 0) {
      break;
    }
  }
  return PAI_OK;
}

static void
check_rejection(pai_status_t st, const pai_proto_conn_t *server) {
  CHECK(st == PAI_OK || st == PAI_ERR_PROTOCOL || st == PAI_ERR_NOMEM);
  if (st == PAI_ERR_PROTOCOL) {
    /* A protocol violation must terminate the connection. */
    CHECK(pai_proto_conn_state(server) == PAI_PROTO_STATE_CLOSED);
  }
}

/* ------------------------------------------------------------------ */
/* surface A — codec fuzz                                              */
/* ------------------------------------------------------------------ */

typedef pai_status_t (*decoder_fn)(const uint8_t *, uint32_t);

static pai_status_t
d_caps(const uint8_t *p, uint32_t n) {
  uint32_t v = 0;
  return pai_proto_msg_decode_caps(p, n, &v);
}

static pai_status_t
d_u32(const uint8_t *p, uint32_t n) {
  uint32_t v = 0;
  return pai_proto_msg_decode_u32(p, n, &v);
}

static pai_status_t
d_error(const uint8_t *p, uint32_t n) {
  pai_status_t s = PAI_OK;
  char m[64];
  return pai_proto_msg_decode_error(p, n, &s, m, sizeof(m));
}

static pai_status_t
d_generate(const uint8_t *p, uint32_t n) {
  const char *t = NULL;
  uint32_t l = 0;
  return pai_proto_msg_decode_generate(p, n, &t, &l);
}

static pai_status_t
d_generate2(const uint8_t *p, uint32_t n) {
  const char *t = NULL;
  uint32_t l = 0;
  pai_proto_sampler_t sp;
  int h = 0;
  return pai_proto_msg_decode_generate2(p, n, &t, &l, &sp, &h);
}

static pai_status_t
d_token(const uint8_t *p, uint32_t n) {
  const uint8_t *t = NULL;
  uint32_t l = 0;
  return pai_proto_msg_decode_token(p, n, &t, &l);
}

static pai_status_t
d_embed(const uint8_t *p, uint32_t n) {
  const char *t = NULL;
  uint32_t l = 0;
  return pai_proto_msg_decode_embed(p, n, &t, &l);
}

static pai_status_t
d_embedding(const uint8_t *p, uint32_t n) {
  const float *v = NULL;
  uint32_t d = 0;
  return pai_proto_msg_decode_embedding(p, n, &v, &d);
}

static pai_status_t
d_stream_data(const uint8_t *p, uint32_t n) {
  uint8_t k = 0;
  uint32_t s = 0;
  const uint8_t *d = NULL;
  uint32_t l = 0;
  return pai_proto_msg_decode_stream_data(p, n, &k, &s, &d, &l);
}

static const decoder_fn k_decoders[] = {
    d_caps,       d_u32,    d_error,     d_generate, d_generate2,
    d_token,      d_embed,  d_embedding, d_stream_data,
};

static void
fuzz_codec(void) {
  uint8_t buf[512];
  uint8_t pat_zero[64];
  uint8_t pat_ff[64];
  uint8_t pat_ramp[64];

  memset(pat_zero, 0, sizeof(pat_zero));
  memset(pat_ff, 0xFF, sizeof(pat_ff));
  for (int i = 0; i < 64; i++) {
    pat_ramp[i] = (uint8_t)i;
  }

  for (size_t di = 0; di < sizeof(k_decoders) / sizeof(k_decoders[0]); di++) {
    decoder_fn fn = k_decoders[di];

    /* Exhaustive small lengths with fixed patterns: empty, single
     * byte, header-shaped, length-field-shaped, etc. */
    for (uint32_t len = 0; len <= 64; len++) {
      pai_status_t st;
      st = fn(pat_zero, len);
      CHECK(st == PAI_OK || st == PAI_ERR_PROTOCOL);
      st = fn(pat_ff, len);
      CHECK(st == PAI_OK || st == PAI_ERR_PROTOCOL);
      st = fn(pat_ramp, len);
      CHECK(st == PAI_OK || st == PAI_ERR_PROTOCOL);
    }

    /* Random payloads. */
    for (int i = 0; i < 300; i++) {
      uint32_t len = (uint32_t)rnd_len(sizeof(buf));
      pai_status_t st;
      rnd_fill(buf, len);
      st = fn(buf, len);
      CHECK(st == PAI_OK || st == PAI_ERR_PROTOCOL);
    }
  }
}

/* ------------------------------------------------------------------ */
/* surface B — raw wire fuzz                                           */
/* ------------------------------------------------------------------ */

/* Encode a valid frame for a random corpus message type. */
static uint32_t
build_corpus_frame(uint8_t *wire, uint32_t cap) {
  static const uint32_t k_msgs[] = {
      PAI_PROTO_MSG_HELLO,        PAI_PROTO_MSG_HELLO_ACK,
      PAI_PROTO_MSG_HELLO_NACK,   PAI_PROTO_MSG_PING,
      PAI_PROTO_MSG_PONG,         PAI_PROTO_MSG_ERROR,
      PAI_PROTO_MSG_CLOSE,        PAI_PROTO_MSG_SESSION_CREATE,
      PAI_PROTO_MSG_SESSION_CREATED,
      PAI_PROTO_MSG_SESSION_CLOSE, PAI_PROTO_MSG_SESSION_CLOSED,
      PAI_PROTO_MSG_GENERATE,     PAI_PROTO_MSG_ACCEPTED,
      PAI_PROTO_MSG_TOKEN,        PAI_PROTO_MSG_COMPLETE,
      PAI_PROTO_MSG_EMBED,        PAI_PROTO_MSG_EMBEDDING,
      PAI_PROTO_MSG_STREAM_DATA,  PAI_PROTO_MSG_STREAM_CLOSE,
      0x0000,                     0xFFFF,
  };
  uint8_t pay[256];
  uint32_t plen = 0;
  pai_proto_frame_t f;
  uint32_t n = 0;
  uint32_t msg = k_msgs[rnd() % (sizeof(k_msgs) / sizeof(k_msgs[0]))];

  switch (msg) {
  case PAI_PROTO_MSG_HELLO:
    (void)pai_proto_msg_encode_caps(pay, sizeof(pay), PAI_PROTO_CAP_KNOWN,
                                    &plen);
    break;
  case PAI_PROTO_MSG_ERROR:
    (void)pai_proto_msg_encode_error(pay, sizeof(pay), PAI_ERR_IO, "fuzz",
                                     &plen);
    break;
  case PAI_PROTO_MSG_GENERATE:
    if (rnd() % 2) {
      (void)pai_proto_msg_encode_generate(pay, sizeof(pay), "fuzz me", &plen);
    } else {
      pai_proto_sampler_t sp;
      memset(&sp, 0, sizeof(sp));
      sp.top_p = 0.9f;
      (void)pai_proto_msg_encode_generate2(pay, sizeof(pay), "fuzz me", &sp,
                                           &plen);
    }
    break;
  case PAI_PROTO_MSG_TOKEN:
    (void)pai_proto_msg_encode_token(pay, sizeof(pay), (const uint8_t *)"ab",
                                     2, &plen);
    break;
  case PAI_PROTO_MSG_EMBED:
    (void)pai_proto_msg_encode_embed(pay, sizeof(pay), "embed", 5, &plen);
    break;
  case PAI_PROTO_MSG_EMBEDDING: {
    static const float vec[2] = {1.0f, -2.0f};
    (void)pai_proto_msg_encode_embedding(pay, sizeof(pay), vec, 2, &plen);
    break;
  }
  case PAI_PROTO_MSG_STREAM_DATA:
    (void)pai_proto_msg_encode_stream_data(pay, sizeof(pay),
                                           PAI_PROTO_STREAM_LOG, 3,
                                           (const uint8_t *)"xyz", 3, &plen);
    break;
  default:
    plen = (uint32_t)rnd_len(sizeof(pay));
    rnd_fill(pay, plen);
    break;
  }

  memset(&f, 0, sizeof(f));
  f.msg_type = msg;
  f.request_id = 42;
  f.session_id = 7;
  f.payload = pay;
  f.payload_len = plen;
  if (rnd() % 8 == 0) {
    f.flags = PAI_PROTO_FLAG_STREAM_START;
  }
  if (pai_proto_frame_encode(&f, wire, cap, &n) != PAI_OK) {
    return 0;
  }
  return n;
}

/* Apply 1–3 random mutations to a wire frame; occasionally emits a
 * pure random blob instead. Returns the mutated length. */
static uint32_t
mutate(const uint8_t *in, uint32_t inlen, uint8_t *out, uint32_t cap) {
  uint32_t len = inlen < cap ? inlen : cap;

  memcpy(out, in, len);

  if (rnd() % 5 == 0) {
    /* Pure random blob (0..768 bytes). */
    len = (uint32_t)rnd_len(768);
    if (len > cap) {
      len = cap;
    }
    rnd_fill(out, len);
    return len;
  }

  {
    uint32_t muts = 1 + (uint32_t)(rnd() % 3);
    for (uint32_t m = 0; m < muts && len > 0; m++) {
      switch (rnd() % 10) {
      case 0: /* flip a bit */
        out[rnd() % len] ^= (uint8_t)(1u << (rnd() % 8));
        break;
      case 1: /* zero a byte */
        out[rnd() % len] = 0x00;
        break;
      case 2: /* saturate a byte */
        out[rnd() % len] = 0xFF;
        break;
      case 3: /* random byte */
        out[rnd() % len] = rnd_byte();
        break;
      case 4: /* truncate */
        if (len > 1) {
          len = 1 + (uint32_t)(rnd() % len);
        }
        break;
      case 5: /* append random bytes */
        if (len + 64 < cap) {
          uint32_t add = (uint32_t)rnd_len(64);
          rnd_fill(out + len, add);
          len += add;
        }
        break;
      case 6: /* patch payload_len field [12,16) */
        if (len >= 16) {
          uint32_t v;
          switch (rnd() % 7) {
          case 0: v = 0; break;
          case 1: v = 1; break;
          case 2: v = 8; break;
          case 3: v = 40; break;
          case 4: v = 0xFFFF; break;
          case 5: v = PAI_PROTO_MAX_PAYLOAD; break;
          default: v = rnd_u32(); break;
          }
          out[12] = (uint8_t)v;
          out[13] = (uint8_t)(v >> 8);
          out[14] = (uint8_t)(v >> 16);
          out[15] = (uint8_t)(v >> 24);
        }
        break;
      case 7: /* patch magic [0,4) */
        if (len >= 4) {
          out[rnd() % 4] = rnd_byte();
        }
        break;
      case 8: /* patch flags [6,8) */
        if (len >= 8) {
          out[6] = rnd_byte();
          out[7] = rnd_byte();
        }
        break;
      default: /* patch reserved [36,40) or version byte */
        if (len >= 40 && rnd() % 2 == 0) {
          out[36 + (uint32_t)(rnd() % 4)] = rnd_byte();
        } else if (len >= 5) {
          out[4] = (uint8_t)(rnd() % 256);
        }
        break;
      }
    }
  }
  return len;
}

static void
fuzz_raw(uint32_t iters) {
  uint8_t wire[1024];
  uint8_t out[1024];

  for (uint32_t i = 0; i < iters; i++) {
    pai_proto_pipe_pair_t *pair = NULL;
    pai_proto_transport_t ta, tb;
    pai_proto_callbacks_t cb;
    pai_proto_conn_t server;
    uint32_t wlen;
    uint32_t olen;

    wlen = build_corpus_frame(wire, sizeof(wire));
    olen = mutate(wire, wlen, out, sizeof(out));

    if (pai_proto_pipe_pair_create(&pair) != PAI_OK) {
      continue;
    }
    pai_proto_pipe_endpoint(pair, 0, &ta);
    pai_proto_pipe_endpoint(pair, 1, &tb);

    memset(&cb, 0, sizeof(cb));
    cb.on_hello = accept_hello;
    if (pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cb, NULL) != PAI_OK) {
      pai_proto_pipe_pair_destroy(pair);
      continue;
    }

    if (olen > 0) {
      (void)ta.send(ta.ctx, out, olen);
    }
    check_rejection(pump_conn(&server), &server);

    pai_proto_conn_destroy(&server);
    pai_proto_pipe_pair_destroy(pair);
  }
}

/* ------------------------------------------------------------------ */
/* surface C — structured open-state fuzz                              */
/* ------------------------------------------------------------------ */

/* Server-side handler that consumes hostile frames (exercising the
 * decoders) and occasionally refuses to exercise the auto-ERROR path. */
static pai_status_t
structured_on_message(void *user, const pai_proto_frame_t *frame) {
  (void)user;
  switch (frame->msg_type) {
  case PAI_PROTO_MSG_GENERATE: {
    const char *t = NULL;
    uint32_t l = 0;
    (void)pai_proto_msg_decode_generate2(frame->payload, frame->payload_len,
                                         &t, &l, NULL, NULL);
    break;
  }
  case PAI_PROTO_MSG_TOKEN: {
    const uint8_t *t = NULL;
    uint32_t l = 0;
    (void)pai_proto_msg_decode_token(frame->payload, frame->payload_len, &t,
                                     &l);
    break;
  }
  case PAI_PROTO_MSG_EMBED: {
    const char *t = NULL;
    uint32_t l = 0;
    (void)pai_proto_msg_decode_embed(frame->payload, frame->payload_len, &t,
                                     &l);
    break;
  }
  case PAI_PROTO_MSG_EMBEDDING: {
    const float *v = NULL;
    uint32_t d = 0;
    (void)pai_proto_msg_decode_embedding(frame->payload, frame->payload_len,
                                         &v, &d);
    break;
  }
  default:
    break;
  }
  if (rnd() % 16 == 0) {
    return PAI_ERR_UNSUPPORTED; /* triggers a structured ERROR reply */
  }
  return PAI_OK;
}

static const uint32_t k_msg_pool[] = {
    PAI_PROTO_MSG_HELLO,         PAI_PROTO_MSG_HELLO_ACK,
    PAI_PROTO_MSG_HELLO_NACK,    PAI_PROTO_MSG_PING,
    PAI_PROTO_MSG_PONG,          PAI_PROTO_MSG_ERROR,
    PAI_PROTO_MSG_CLOSE,         PAI_PROTO_MSG_SESSION_CREATE,
    PAI_PROTO_MSG_SESSION_CREATED,
    PAI_PROTO_MSG_SESSION_CLOSE, PAI_PROTO_MSG_SESSION_CLOSED,
    PAI_PROTO_MSG_GENERATE,      PAI_PROTO_MSG_ACCEPTED,
    PAI_PROTO_MSG_TOKEN,         PAI_PROTO_MSG_COMPLETE,
    PAI_PROTO_MSG_EMBED,         PAI_PROTO_MSG_EMBEDDING,
    PAI_PROTO_MSG_STREAM_DATA,   PAI_PROTO_MSG_STREAM_CLOSE,
    0x0000,                      0x0105,
    0x0200,                      0x0207,
    0x0300,                      0x0303,
    0x7FFFFFFFu,                 0xFFFFFFF0u,
};

static const uint32_t k_flag_pool[] = {
    0,
    PAI_PROTO_FLAG_REPLY,
    PAI_PROTO_FLAG_STREAM_START,
    PAI_PROTO_FLAG_STREAM_END,
    PAI_PROTO_FLAG_STREAM_START | PAI_PROTO_FLAG_STREAM_END,
    PAI_PROTO_FLAG_COMPRESSED,
};

static const uint64_t k_id_pool[] = {0, 1, 2, 42, 0xFFFFFFFFFFFFFFFFull};

static void
fuzz_structured(uint32_t iters) {
  for (uint32_t i = 0; i < iters; i++) {
    pai_proto_pipe_pair_t *pair = NULL;
    pai_proto_transport_t ta, tb;
    pai_proto_callbacks_t cbc, cbs;
    pai_proto_conn_t client, server;
    pai_proto_frame_t f;
    uint8_t wire[1024];
    uint8_t pay[128];
    uint32_t plen = 0;
    uint32_t n = 0;
    pai_status_t st;
    int crc_patched = 0;

    /* Build the payload: valid-for-type half the time, garbage the
     * other half (a wrong payload must still be rejected safely). */
    {
      uint32_t msg = k_msg_pool[rnd() % (sizeof(k_msg_pool) / sizeof(k_msg_pool[0]))];
      switch (msg) {
      case PAI_PROTO_MSG_HELLO:
      case PAI_PROTO_MSG_ERROR:
      case PAI_PROTO_MSG_TOKEN:
      case PAI_PROTO_MSG_STREAM_DATA:
      case PAI_PROTO_MSG_EMBED:
        if (rnd() % 2) {
          switch (msg) {
          case PAI_PROTO_MSG_HELLO:
            (void)pai_proto_msg_encode_caps(pay, sizeof(pay),
                                            PAI_PROTO_CAP_KNOWN, &plen);
            break;
          case PAI_PROTO_MSG_ERROR:
            (void)pai_proto_msg_encode_error(pay, sizeof(pay), PAI_ERR_IO,
                                             "boom", &plen);
            break;
          case PAI_PROTO_MSG_TOKEN:
            (void)pai_proto_msg_encode_token(pay, sizeof(pay),
                                             (const uint8_t *)"xy", 2, &plen);
            break;
          case PAI_PROTO_MSG_STREAM_DATA:
            (void)pai_proto_msg_encode_stream_data(pay, sizeof(pay),
                                                   PAI_PROTO_STREAM_LOG, 1,
                                                   (const uint8_t *)"d", 1,
                                                   &plen);
            break;
          default:
            (void)pai_proto_msg_encode_embed(pay, sizeof(pay), "text", 4,
                                             &plen);
            break;
          }
        } else {
          plen = (uint32_t)rnd_len(sizeof(pay));
          rnd_fill(pay, plen);
        }
        break;
      default:
        plen = (uint32_t)rnd_len(sizeof(pay));
        rnd_fill(pay, plen);
        break;
      }
      f.msg_type = msg;
    }
    f.flags = k_flag_pool[rnd() % (sizeof(k_flag_pool) / sizeof(k_flag_pool[0]))];
    f.request_id = k_id_pool[rnd() % (sizeof(k_id_pool) / sizeof(k_id_pool[0]))];
    f.session_id = k_id_pool[rnd() % (sizeof(k_id_pool) / sizeof(k_id_pool[0]))];
    f.payload = pay;
    f.payload_len = plen;

    /* encode refuses COMPRESSED — encode a copy with the flag
     * stripped, then patch the byte in and recompute the CRC so the
     * frame is otherwise valid; the receiver must reject the flag on
     * its own. Oversized payloads are encoder-level refusals already
     * covered elsewhere (payload_len here is bounded by pay[128]). */
    {
      pai_proto_frame_t enc = f;
      enc.flags &= ~PAI_PROTO_FLAG_COMPRESSED;
      if (pai_proto_frame_encode(&enc, wire, sizeof(wire), &n) != PAI_OK) {
        continue;
      }
    }
    if (f.flags & PAI_PROTO_FLAG_COMPRESSED) {
      wire[6] |= PAI_PROTO_FLAG_COMPRESSED;
      crc_patched = 1;
    }
    /* Occasionally claim a bogus protocol major (CRC-valid so the
     * version-mismatch branch itself fires). */
    if (rnd() % 8 == 0) {
      wire[4] = (uint8_t)(1 + rnd() % 255);
      crc_patched = 1;
    }
    if (crc_patched) {
      uint32_t crc = pai_proto_crc32_init();
      crc = pai_proto_crc32_upd(crc, wire, 32);
      crc = pai_proto_crc32_upd(crc, wire + PAI_PROTO_HEADER_SIZE,
                                f.payload_len);
      {
        uint32_t c = pai_proto_crc32_fin(crc);
        wire[32] = (uint8_t)c;
        wire[33] = (uint8_t)(c >> 8);
        wire[34] = (uint8_t)(c >> 16);
        wire[35] = (uint8_t)(c >> 24);
      }
    }

    if (pai_proto_pipe_pair_create(&pair) != PAI_OK) {
      continue;
    }
    pai_proto_pipe_endpoint(pair, 0, &ta);
    pai_proto_pipe_endpoint(pair, 1, &tb);

    memset(&cbc, 0, sizeof(cbc));
    memset(&cbs, 0, sizeof(cbs));
    cbs.on_hello = accept_hello;
    cbs.on_message = structured_on_message;
    if (pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &ta, &cbc, NULL) != PAI_OK ||
        pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cbs, NULL) != PAI_OK) {
      pai_proto_conn_destroy(&client);
      pai_proto_conn_destroy(&server);
      pai_proto_pipe_pair_destroy(pair);
      continue;
    }

    /* Negotiate. */
    (void)pai_proto_conn_start(&client);
    (void)pump_conn(&server);
    (void)pump_conn(&client);
    if (pai_proto_conn_state(&client) != PAI_PROTO_STATE_OPEN) {
      pai_proto_conn_destroy(&client);
      pai_proto_conn_destroy(&server);
      pai_proto_pipe_pair_destroy(pair);
      continue;
    }

    (void)ta.send(ta.ctx, wire, n);
    st = pump_conn(&server);
    check_rejection(st, &server);
    /* Drain any auto-ERROR reply the server may have queued. */
    (void)pump_conn(&client);

    pai_proto_conn_destroy(&client);
    pai_proto_conn_destroy(&server);
    pai_proto_pipe_pair_destroy(pair);
  }
}

/* ------------------------------------------------------------------ */
/* session / request-id stress                                         */
/* ------------------------------------------------------------------ */

static void
fuzz_sessions(void) {
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t tb;
  pai_proto_conn_t server;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 1, &tb);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_SESSIONS, &tb, NULL, NULL) ==
        PAI_OK);

  /* Fill the table to capacity; the next open must report NOMEM. */
  {
    uint64_t ids[PAI_PROTO_MAX_SESSIONS];
    uint64_t last = 0;
    for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS; i++) {
      CHECK(pai_proto_conn_session_open(&server, &ids[i]) == PAI_OK);
      CHECK(ids[i] != 0);
      CHECK(ids[i] != last); /* strictly increasing, no reuse */
      last = ids[i];
    }
    {
      uint64_t x = 0;
      CHECK(pai_proto_conn_session_open(&server, &x) == PAI_ERR_NOMEM);
    }
    /* Close every other slot, then reopen — ids must stay fresh. */
    for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS; i += 2) {
      pai_proto_conn_session_close(&server, ids[i]);
    }
    for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS / 2; i++) {
      uint64_t x = 0;
      CHECK(pai_proto_conn_session_open(&server, &x) == PAI_OK);
      CHECK(x != 0);
      CHECK(x > last); /* monotonically fresh ids, never reused */
      last = x;
    }
  }

  /* Request-id wrap: the allocator must never hand out 0. */
  server.next_request_id = 0xFFFFFFFFFFFFFFFFull;
  CHECK(pai_proto_conn_new_request_id(&server) == 0xFFFFFFFFFFFFFFFFull);
  CHECK(pai_proto_conn_new_request_id(&server) == 1); /* wraps to 1, not 0 */

  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

TEST_MAIN_BEGIN_ARGS()

  unsigned long long seed = 0x9E3779B97F4A7C15ull;
  unsigned long raw_iters = 25000;
  if (argc > 1) {
    seed = strtoull(argv[1], NULL, 0);
  }
  if (argc > 2) {
    raw_iters = strtoul(argv[2], NULL, 0);
  }
  g_rng = seed ? seed : 1;
  printf("seed=%llu raw_iters=%lu\n", seed, raw_iters);

  fuzz_codec();
  fuzz_raw((uint32_t)raw_iters);
  fuzz_structured(3000);
  fuzz_sessions();

TEST_MAIN_END()
