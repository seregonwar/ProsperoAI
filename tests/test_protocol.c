#include "test.h"

#include <protocol/protocol.h>

#include <string.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static void
pump(pai_proto_conn_t *a, pai_proto_conn_t *b, int iters) {
  uint32_t frames = 0;
  for (int i = 0; i < iters; i++) {
    CHECK(pai_proto_conn_poll(b, &frames) == PAI_OK);
    CHECK(pai_proto_conn_poll(a, &frames) == PAI_OK);
  }
}

/* Default server: accept any caps. */
static uint32_t
accept_hello(void *user, uint32_t remote_caps) {
  (void)user;
  (void)remote_caps;
  return PAI_PROTO_CAP_KNOWN;
}

/* Refuse every connection. */
static uint32_t
refuse_hello(void *user, uint32_t remote_caps) {
  (void)user;
  (void)remote_caps;
  return 0;
}

/* ------------------------------------------------------------------ */
/* callbacks for the GENERATE example                                  */
/* ------------------------------------------------------------------ */

typedef struct gen_server {
  pai_proto_conn_t *conn;
  int got_generate;
  char prompt[64];
  uint32_t prompt_len;
  uint64_t session_id;
} gen_server_t;

typedef struct gen_client {
  int begin_count;
  int end_count;
  char out[256];
  uint32_t out_len;
  int accepted;
  int complete;
  uint64_t accepted_req;
  uint64_t complete_req;
  uint32_t expected_seq;
  int seq_ok;
} gen_client_t;

static pai_status_t
gen_server_on_message(void *user, const pai_proto_frame_t *frame) {
  gen_server_t *s = (gen_server_t *)user;
  const char *prompt;
  uint32_t plen;

  if (frame->msg_type != PAI_PROTO_MSG_GENERATE) {
    return PAI_OK;
  }

  CHECK(pai_proto_msg_decode_generate(frame->payload, frame->payload_len,
                                      &prompt, &plen) == PAI_OK);
  s->got_generate = 1;
  s->prompt_len = plen < 63u ? plen : 63u;
  memcpy(s->prompt, prompt, s->prompt_len);
  s->prompt[s->prompt_len] = '\0';

  CHECK(pai_proto_conn_session_open(s->conn, &s->session_id) == PAI_OK);
  CHECK_EQ_UINT(s->session_id, 1);

  CHECK(pai_proto_conn_send_raw(s->conn, PAI_PROTO_MSG_ACCEPTED,
                                PAI_PROTO_FLAG_REPLY, frame->request_id,
                                s->session_id, NULL, 0) == PAI_OK);

  /* Stream the prompt as 2-char tokens (whitepaper §24 diagram). */
  {
    uint32_t off = 0;
    while (off < s->prompt_len) {
      uint32_t n = s->prompt_len - off;
      uint32_t flags = 0;
      uint8_t pay[128];
      uint32_t tlen = 0;
      if (n > 2) {
        n = 2;
      }
      if (off == 0) {
        flags |= PAI_PROTO_FLAG_STREAM_START;
      }
      if (off + n >= s->prompt_len) {
        flags |= PAI_PROTO_FLAG_STREAM_END;
      }
      CHECK(pai_proto_msg_encode_token(pay, sizeof(pay),
                                       (const uint8_t *)s->prompt + off, n,
                                       &tlen) == PAI_OK);
      CHECK(pai_proto_conn_send_raw(s->conn, PAI_PROTO_MSG_TOKEN, flags,
                                    frame->request_id, s->session_id, pay,
                                    tlen) == PAI_OK);
      off += n;
    }
  }

  CHECK(pai_proto_conn_send_raw(s->conn, PAI_PROTO_MSG_COMPLETE, 0,
                                frame->request_id, s->session_id, NULL,
                                0) == PAI_OK);
  return PAI_OK;
}

static void
gen_client_on_stream_begin(void *user, uint64_t request_id,
                           uint64_t session_id) {
  gen_client_t *c = (gen_client_t *)user;
  c->begin_count++;
  (void)request_id;
  (void)session_id;
}

static void
gen_client_on_stream_data(void *user, uint64_t request_id, uint64_t session_id,
                          uint8_t kind, uint32_t seq, const uint8_t *data,
                          uint32_t len) {
  gen_client_t *c = (gen_client_t *)user;
  (void)request_id;
  (void)session_id;
  if (kind != PAI_PROTO_STREAM_GENERATE) {
    c->seq_ok = 0;
  }
  if (seq != c->expected_seq) {
    c->seq_ok = 0;
  }
  c->expected_seq++;
  if (c->out_len + len <= sizeof(c->out)) {
    memcpy(c->out + c->out_len, data, len);
    c->out_len += len;
  } else {
    c->seq_ok = 0;
  }
}

static void
gen_client_on_stream_end(void *user, uint64_t request_id, uint64_t session_id) {
  gen_client_t *c = (gen_client_t *)user;
  c->end_count++;
  (void)request_id;
  (void)session_id;
}

static pai_status_t
gen_client_on_message(void *user, const pai_proto_frame_t *frame) {
  gen_client_t *c = (gen_client_t *)user;
  if (frame->msg_type == PAI_PROTO_MSG_ACCEPTED) {
    c->accepted = 1;
    c->accepted_req = frame->request_id;
  } else if (frame->msg_type == PAI_PROTO_MSG_COMPLETE) {
    c->complete = 1;
    c->complete_req = frame->request_id;
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* callbacks for PING/PONG, ERROR, CLOSE                               */
/* ------------------------------------------------------------------ */

typedef struct ping_client {
  uint64_t pongs[4];
  uint32_t npongs;
} ping_client_t;

static pai_status_t
ping_on_message(void *user, const pai_proto_frame_t *frame) {
  ping_client_t *c = (ping_client_t *)user;
  if (frame->msg_type == PAI_PROTO_MSG_PONG &&
      (frame->flags & PAI_PROTO_FLAG_REPLY)) {
    if (c->npongs < 4) {
      c->pongs[c->npongs++] = frame->request_id;
    }
  }
  return PAI_OK;
}

typedef struct err_client {
  int got_error;
  pai_status_t status;
  uint64_t req;
} err_client_t;

static pai_status_t
err_client_on_message(void *user, const pai_proto_frame_t *frame) {
  err_client_t *c = (err_client_t *)user;
  if (frame->msg_type == PAI_PROTO_MSG_ERROR) {
    c->got_error = 1;
    c->req = frame->request_id;
    pai_proto_msg_decode_error(frame->payload, frame->payload_len, &c->status,
                               NULL, 0);
  }
  return PAI_OK;
}

static pai_status_t
reject_session_create(void *user, const pai_proto_frame_t *frame) {
  (void)user;
  if (frame->msg_type == PAI_PROTO_MSG_SESSION_CREATE) {
    return PAI_ERR_UNSUPPORTED;
  }
  return PAI_OK;
}

typedef struct close_client {
  int on_close;
  uint32_t reason;
} close_client_t;

static void
close_observer(void *user, uint32_t reason) {
  close_client_t *c = (close_client_t *)user;
  c->on_close = 1;
  c->reason = reason;
}

TEST_MAIN_BEGIN()

/* ------------------------------------------------------------------ */
/* CRC-32 golden vector                                                */
/* ------------------------------------------------------------------ */

{
  const char *s = "123456789";
  CHECK_EQ_UINT(pai_proto_crc32(s, 9), 0xCBF43926u);
  /* chunked continuation must match the one-shot result */
  {
    uint32_t crc = pai_proto_crc32_init();
    crc = pai_proto_crc32_upd(crc, s, 4);
    crc = pai_proto_crc32_upd(crc, s + 4, 5);
    CHECK_EQ_UINT(pai_proto_crc32_fin(crc), 0xCBF43926u);
  }
}

/* ------------------------------------------------------------------ */
/* frame codec                                                         */
/* ------------------------------------------------------------------ */

{
  static const uint8_t pay[] = {0x05, 0x00, 'h', 'e', 'l', 'l', 'o'};
  uint8_t wire[256];
  uint32_t n = 0;
  pai_proto_frame_t f;
  pai_proto_frame_t g;

  memset(&f, 0, sizeof(f));
  f.msg_type = PAI_PROTO_MSG_GENERATE;
  f.request_id = 42;
  f.session_id = 7;
  f.payload = pay;
  f.payload_len = sizeof(pay);

  CHECK(pai_proto_frame_encode(&f, wire, sizeof(wire), &n) == PAI_OK);
  CHECK_EQ_UINT(n, PAI_PROTO_HEADER_SIZE + sizeof(pay));
  CHECK_EQ_UINT(pai_proto_frame_size(sizeof(pay)), n);

  /* golden header bytes */
  CHECK_EQ_UINT(wire[0] | ((uint32_t)wire[1] << 8) | ((uint32_t)wire[2] << 16) |
                    ((uint32_t)wire[3] << 24),
                0x50494150u);
  CHECK_EQ_UINT(wire[4], PAI_PROTO_VERSION_MAJOR);
  CHECK_EQ_UINT(wire[5], PAI_PROTO_VERSION_MINOR);
  CHECK_EQ_UINT(wire[6] | ((uint32_t)wire[7] << 8), 0);
  CHECK_EQ_UINT(wire[8] | ((uint32_t)wire[9] << 8) | ((uint32_t)wire[10] << 16) |
                    ((uint32_t)wire[11] << 24),
                PAI_PROTO_MSG_GENERATE);
  CHECK_EQ_UINT(wire[12] | ((uint32_t)wire[13] << 8) |
                    ((uint32_t)wire[14] << 16) | ((uint32_t)wire[15] << 24),
                sizeof(pay));
  CHECK_EQ_UINT(wire[16] | ((uint64_t)wire[17] << 8) |
                    ((uint64_t)wire[18] << 16) | ((uint64_t)wire[19] << 24) |
                    ((uint64_t)wire[20] << 32) | ((uint64_t)wire[21] << 40) |
                    ((uint64_t)wire[22] << 48) | ((uint64_t)wire[23] << 56),
                42u);
  CHECK_EQ_UINT(wire[24] | ((uint64_t)wire[25] << 8) |
                    ((uint64_t)wire[26] << 16) | ((uint64_t)wire[27] << 24) |
                    ((uint64_t)wire[28] << 32) | ((uint64_t)wire[29] << 40) |
                    ((uint64_t)wire[30] << 48) | ((uint64_t)wire[31] << 56),
                7u);
  CHECK_EQ_UINT(wire[36] | ((uint32_t)wire[37] << 8) |
                    ((uint32_t)wire[38] << 16) | ((uint32_t)wire[39] << 24),
                0u);

  CHECK(pai_proto_frame_decode(wire, &g) == PAI_OK);
  CHECK_EQ_UINT(g.msg_type, PAI_PROTO_MSG_GENERATE);
  CHECK_EQ_UINT(g.request_id, 42);
  CHECK_EQ_UINT(g.session_id, 7);
  CHECK_EQ_UINT(g.payload_len, sizeof(pay));
  CHECK_EQ_UINT(g.version, PAI_PROTO_VERSION);
  CHECK(pai_proto_frame_validate(wire, wire + PAI_PROTO_HEADER_SIZE) ==
        PAI_OK);
}

{
  uint8_t wire[64];
  uint32_t n = 0;
  pai_proto_frame_t f;
  pai_proto_frame_t g;

  /* bad magic */
  memset(&f, 0, sizeof(f));
  f.msg_type = PAI_PROTO_MSG_PING;
  CHECK(pai_proto_frame_encode(&f, wire, sizeof(wire), &n) == PAI_OK);
  wire[0] ^= 0xFF;
  CHECK(pai_proto_frame_decode(wire, &g) == PAI_ERR_PROTOCOL);
  wire[0] ^= 0xFF;

  /* corrupt payload byte -> CRC mismatch */
  CHECK(pai_proto_frame_validate(wire, NULL) == PAI_OK);
  wire[32] ^= 0x01; /* flip a CRC byte */
  CHECK(pai_proto_frame_validate(wire, NULL) == PAI_ERR_PROTOCOL);
  wire[32] ^= 0x01;

  /* nonzero reserved */
  wire[36] = 1;
  CHECK(pai_proto_frame_decode(wire, &g) == PAI_ERR_PROTOCOL);
  wire[36] = 0;

  /* oversized payload */
  f.payload_len = PAI_PROTO_MAX_PAYLOAD + 1;
  CHECK(pai_proto_frame_encode(&f, wire, sizeof(wire), &n) ==
        PAI_ERR_INVALID_ARG);
  f.payload_len = 0;

  /* compressed flag is reserved in v0 */
  f.flags = PAI_PROTO_FLAG_COMPRESSED;
  CHECK(pai_proto_frame_encode(&f, wire, sizeof(wire), &n) ==
        PAI_ERR_INVALID_ARG);
}

/* ------------------------------------------------------------------ */
/* message payload codecs                                              */
/* ------------------------------------------------------------------ */

{
  uint8_t buf[256];
  uint32_t len = 0;

  /* caps round-trip */
  CHECK(pai_proto_msg_encode_caps(buf, sizeof(buf), 0x27u, &len) == PAI_OK);
  CHECK_EQ_UINT(len, PAI_PROTO_U32_PAIR_SIZE);
  {
    uint32_t caps = 0;
    CHECK(pai_proto_msg_decode_caps(buf, len, &caps) == PAI_OK);
    CHECK_EQ_UINT(caps, 0x27u);
  }
  CHECK(pai_proto_msg_decode_caps(buf, 4, &caps) == PAI_ERR_PROTOCOL);

  /* u32 (nack/close reason) round-trip */
  CHECK(pai_proto_msg_encode_u32(buf, sizeof(buf), PAI_PROTO_NACK_VERSION,
                                 &len) == PAI_OK);
  {
    uint32_t v = 0;
    CHECK(pai_proto_msg_decode_u32(buf, len, &v) == PAI_OK);
    CHECK_EQ_UINT(v, PAI_PROTO_NACK_VERSION);
  }
  CHECK(pai_proto_msg_decode_u32(buf, 4, &v) == PAI_ERR_PROTOCOL);

  /* error round-trip with message */
  CHECK(pai_proto_msg_encode_error(buf, sizeof(buf), PAI_ERR_TIMEOUT,
                                   "timed out", &len) == PAI_OK);
  {
    pai_status_t st = PAI_OK;
    char msg[64];
    CHECK(pai_proto_msg_decode_error(buf, len, &st, msg, sizeof(msg)) ==
          PAI_OK);
    CHECK_EQ_INT(st, PAI_ERR_TIMEOUT);
    CHECK(strcmp(msg, "timed out") == 0);
  }
  /* truncated error */
  {
    pai_status_t st = PAI_OK;
    CHECK(pai_proto_msg_decode_error(buf, 4, &st, NULL, 0) ==
          PAI_ERR_PROTOCOL);
  }

  /* generate round-trip */
  CHECK(pai_proto_msg_encode_generate(buf, sizeof(buf), "Hello!", &len) ==
        PAI_OK);
  {
    const char *prompt = NULL;
    uint32_t plen = 0;
    CHECK(pai_proto_msg_decode_generate(buf, len, &prompt, &plen) == PAI_OK);
    CHECK_EQ_UINT(plen, 6);
    CHECK(memcmp(prompt, "Hello!", 6) == 0);
  }
  CHECK(pai_proto_msg_encode_generate(buf, sizeof(buf), NULL, &len) ==
        PAI_ERR_INVALID_ARG);

  /* token round-trip */
  CHECK(pai_proto_msg_encode_token(buf, sizeof(buf), (const uint8_t *)"tok",
                                   3, &len) == PAI_OK);
  {
    const uint8_t *tok = NULL;
    uint32_t tlen = 0;
    CHECK(pai_proto_msg_decode_token(buf, len, &tok, &tlen) == PAI_OK);
    CHECK_EQ_UINT(tlen, 3);
    CHECK(memcmp(tok, "tok", 3) == 0);
  }
  /* declared length beyond payload -> malformed */
  {
    uint8_t bad[2] = {0x05, 0x00};
    const uint8_t *tok = NULL;
    uint32_t tlen = 0;
    CHECK(pai_proto_msg_decode_token(bad, sizeof(bad), &tok, &tlen) ==
          PAI_ERR_PROTOCOL);
  }
  /* token length cap */
  CHECK(pai_proto_msg_encode_token(buf, sizeof(buf), NULL, 0x10000u, &len) ==
        PAI_ERR_INVALID_ARG);

  /* stream_data round-trip */
  CHECK(pai_proto_msg_encode_stream_data(buf, sizeof(buf),
                                         PAI_PROTO_STREAM_TELEMETRY, 9,
                                         (const uint8_t *)"abc", 3,
                                         &len) == PAI_OK);
  {
    uint8_t kind = 0;
    uint32_t seq = 0;
    const uint8_t *data = NULL;
    uint32_t dlen = 0;
    CHECK(pai_proto_msg_decode_stream_data(buf, len, &kind, &seq, &data,
                                           &dlen) == PAI_OK);
    CHECK_EQ_UINT(kind, PAI_PROTO_STREAM_TELEMETRY);
    CHECK_EQ_UINT(seq, 9);
    CHECK_EQ_UINT(dlen, 3);
    CHECK(memcmp(data, "abc", 3) == 0);
  }
  CHECK(pai_proto_msg_decode_stream_data(buf, 4, &kind, &seq, &data, &dlen) ==
        PAI_ERR_PROTOCOL);

  /* names */
  CHECK(strcmp(pai_proto_msg_name(PAI_PROTO_MSG_GENERATE), "generate") == 0);
  CHECK(strcmp(pai_proto_msg_name(0xFFFF), "unknown") == 0);
}

/* ------------------------------------------------------------------ */
/* pipe transport                                                      */
/* ------------------------------------------------------------------ */

{
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t a, b;
  char buf[64];
  uint32_t got = 0;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &a);
  pai_proto_pipe_endpoint(pair, 1, &b);

  CHECK(a.send(a.ctx, "hello", 5) == PAI_OK);
  CHECK(b.recv(b.ctx, buf, sizeof(buf), &got) == PAI_OK);
  CHECK_EQ_UINT(got, 5);
  CHECK(memcmp(buf, "hello", 5) == 0);

  /* partial pull */
  CHECK(b.send(b.ctx, "world", 5) == PAI_OK);
  CHECK(a.recv(a.ctx, buf, 2, &got) == PAI_OK);
  CHECK_EQ_UINT(got, 2);
  CHECK(a.recv(a.ctx, buf, 2, &got) == PAI_OK);
  CHECK_EQ_UINT(got, 2);
  CHECK_EQ_UINT(((uint8_t *)buf)[0], 'w');

  /* close: peer sees EOF, sends fail */
  a.close(a.ctx);
  CHECK(b.recv(b.ctx, buf, 2, &got) == PAI_OK);
  CHECK_EQ_UINT(got, 1); /* remaining 'd' */
  CHECK(b.recv(b.ctx, buf, 2, &got) == PAI_ERR_IO);
  CHECK_EQ_UINT(got, 0);
  CHECK(b.send(b.ctx, "x", 1) == PAI_ERR_IO);

  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* negotiation                                                         */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cb;
  uint32_t frames = 0;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cb, 0, sizeof(cb));
  cb.on_hello = accept_hello;
  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_GENERATE | PAI_PROTO_CAP_SESSIONS |
                                PAI_PROTO_CAP_STREAMS |
                                PAI_PROTO_CAP_TELEMETRY,
                            &ta, NULL, NULL) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cb, NULL) == PAI_OK);

  /* cannot send before negotiation */
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 1, 0, NULL,
                                0) == PAI_ERR_INVALID_ARG);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_NEGOTIATING);

  pump(&client, &server, 8);

  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_OPEN);
  CHECK_EQ_UINT(pai_proto_conn_negotiated_caps(&client),
                PAI_PROTO_CAP_GENERATE | PAI_PROTO_CAP_SESSIONS |
                    PAI_PROTO_CAP_STREAMS | PAI_PROTO_CAP_TELEMETRY);
  CHECK_EQ_UINT(pai_proto_conn_negotiated_caps(&server),
                PAI_PROTO_CAP_GENERATE | PAI_PROTO_CAP_SESSIONS |
                    PAI_PROTO_CAP_STREAMS | PAI_PROTO_CAP_TELEMETRY);

  /* idle poll: nothing available, not an error */
  CHECK(pai_proto_conn_poll(&client, &frames) == PAI_OK);
  CHECK_EQ_UINT(frames, 0);

  /* request-id allocator */
  CHECK_EQ_UINT(pai_proto_conn_new_request_id(&client), 2);

  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* version mismatch -> HELLO_NACK(VERSION)                             */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cb;
  uint8_t wire[64];
  uint32_t n = 0;
  uint32_t frames = 0;
  uint32_t got = 0;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cb, 0, sizeof(cb));
  cb.on_hello = accept_hello;
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cb, NULL) == PAI_OK);

  /* Craft a HELLO claiming protocol major 1 and inject it raw. */
  {
    pai_proto_frame_t f;
    uint8_t pay[PAI_PROTO_U32_PAIR_SIZE];
    uint32_t plen = 0;

    memset(&f, 0, sizeof(f));
    f.msg_type = PAI_PROTO_MSG_HELLO;
    f.request_id = 1;
    pai_proto_msg_encode_caps(pay, sizeof(pay), PAI_PROTO_CAP_KNOWN, &plen);
    f.payload = pay;
    f.payload_len = plen;
    CHECK(pai_proto_frame_encode(&f, wire, sizeof(wire), &n) == PAI_OK);
  }
  wire[4] = 1; /* major 1 */
  CHECK(ta.send(ta.ctx, wire, n) == PAI_OK);

  CHECK(pai_proto_conn_poll(&server, &frames) == PAI_OK);
  CHECK_EQ_UINT(frames, 1);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_CLOSED);

  /* Read the NACK the server replied before closing. The server's
   * reply travels b_to_a, which is side A's receive direction. */
  {
    uint8_t hdr[PAI_PROTO_HEADER_SIZE];
    uint8_t pay[PAI_PROTO_U32_PAIR_SIZE];
    pai_proto_frame_t g;

    CHECK(ta.recv(ta.ctx, hdr, PAI_PROTO_HEADER_SIZE, &got) == PAI_OK);
    CHECK_EQ_UINT(got, PAI_PROTO_HEADER_SIZE);
    CHECK(pai_proto_frame_decode(hdr, &g) == PAI_OK);
    CHECK_EQ_UINT(g.msg_type, PAI_PROTO_MSG_HELLO_NACK);
    CHECK(g.flags & PAI_PROTO_FLAG_REPLY);
    CHECK_EQ_UINT(g.request_id, 1);
    CHECK(ta.recv(ta.ctx, pay, PAI_PROTO_U32_PAIR_SIZE, &got) == PAI_OK);
    {
      uint32_t reason = 0;
      CHECK(pai_proto_msg_decode_u32(pay, got, &reason) == PAI_OK);
      CHECK_EQ_UINT(reason, PAI_PROTO_NACK_VERSION);
    }
  }

  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* caps refusal -> HELLO_NACK(CAPS)                                    */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cb;
  uint32_t frames = 0;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cb, 0, sizeof(cb));
  cb.on_hello = refuse_hello;
  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &ta, NULL, NULL) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cb, NULL) == PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  pump(&client, &server, 8);

  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_CLOSED);
  CHECK_EQ_UINT(client.nack_reason, PAI_PROTO_NACK_CAPS);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_CLOSED);

  (void)frames;
  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* pipelined PING/PONG                                                 */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cb;
  ping_client_t pc;
  uint32_t frames = 0;

  memset(&pc, 0, sizeof(pc));
  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cb, 0, sizeof(cb));
  cb.on_hello = accept_hello;
  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &ta, NULL, NULL) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cb, NULL) == PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);

  /* three in-flight pings, no waiting (pipelining) */
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 100, 0, NULL,
                                0) == PAI_OK);
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 101, 0, NULL,
                                0) == PAI_OK);
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 102, 0, NULL,
                                0) == PAI_OK);

  pump(&client, &server, 8);

  CHECK_EQ_UINT(pc.npongs, 3);
  CHECK_EQ_UINT(pc.pongs[0], 100);
  CHECK_EQ_UINT(pc.pongs[1], 101);
  CHECK_EQ_UINT(pc.pongs[2], 102);

  (void)frames;
  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* whitepaper §24 example: GENERATE -> ACCEPTED -> TOKEN* -> COMPLETE  */
/* ------------------------------------------------------------------ */

{
  static const char prompt[] = "Hello world!";
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cbc, cbs;
  gen_server_t gs;
  gen_client_t gc;
  uint32_t frames = 0;

  memset(&gs, 0, sizeof(gs));
  memset(&gc, 0, sizeof(gc));
  gc.expected_seq = 0;
  gc.seq_ok = 1;

  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cbc, 0, sizeof(cbc));
  cbc.on_stream_begin = gen_client_on_stream_begin;
  cbc.on_stream_data = gen_client_on_stream_data;
  cbc.on_stream_end = gen_client_on_stream_end;
  cbc.on_message = gen_client_on_message;

  memset(&cbs, 0, sizeof(cbs));
  cbs.on_hello = accept_hello;
  cbs.on_message = gen_server_on_message;

  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_GENERATE | PAI_PROTO_CAP_SESSIONS |
                                PAI_PROTO_CAP_STREAMS,
                            &ta, &cbc, &gc) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cbs, &gs) == PAI_OK);
  gs.conn = &server;

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_OPEN);

  /* GENERATE #42 — exactly the whitepaper diagram */
  {
    uint8_t pay[128];
    uint32_t plen = 0;
    CHECK(pai_proto_msg_encode_generate(pay, sizeof(pay), prompt, &plen) ==
          PAI_OK);
    CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_GENERATE, 0, 42, 0,
                                  pay, plen) == PAI_OK);
  }

  pump(&client, &server, 8);

  CHECK_EQ_UINT(gs.got_generate, 1);
  CHECK(strcmp(gs.prompt, prompt) == 0);
  CHECK_EQ_UINT(gs.session_id, 1);
  CHECK_EQ_UINT(gs.prompt_len, strlen(prompt));

  CHECK_EQ_UINT(gc.accepted, 1);
  CHECK_EQ_UINT(gc.accepted_req, 42);
  CHECK_EQ_UINT(gc.complete, 1);
  CHECK_EQ_UINT(gc.complete_req, 42);
  CHECK_EQ_UINT(gc.begin_count, 1);
  CHECK_EQ_UINT(gc.end_count, 1);
  CHECK_EQ_UINT(gc.seq_ok, 1);
  CHECK_EQ_UINT(gc.out_len, strlen(prompt));
  CHECK(memcmp(gc.out, prompt, strlen(prompt)) == 0);

  (void)frames;
  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* structured ERROR auto-reply on handler failure                      */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cbc, cbs;
  err_client_t ec;
  uint32_t frames = 0;

  memset(&ec, 0, sizeof(ec));
  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cbc, 0, sizeof(cbc));
  cbc.on_message = err_client_on_message;

  memset(&cbs, 0, sizeof(cbs));
  cbs.on_hello = accept_hello;
  cbs.on_message = reject_session_create;

  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_SESSIONS, &ta, &cbc, &ec) ==
        PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cbs, NULL) == PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);

  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_SESSION_CREATE, 0, 7,
                                0, NULL, 0) == PAI_OK);
  pump(&client, &server, 8);

  CHECK_EQ_UINT(ec.got_error, 1);
  CHECK_EQ_UINT(ec.req, 7);
  CHECK_EQ_INT(ec.status, PAI_ERR_UNSUPPORTED);

  (void)frames;
  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

/* ------------------------------------------------------------------ */
/* CLOSE handshake                                                     */
/* ------------------------------------------------------------------ */

{
  pai_proto_conn_t client, server;
  pai_proto_pipe_pair_t *pair = NULL;
  pai_proto_transport_t ta, tb;
  pai_proto_callbacks_t cbc, cbs;
  close_client_t cc, sc;
  uint32_t frames = 0;

  memset(&cc, 0, sizeof(cc));
  memset(&sc, 0, sizeof(sc));
  CHECK(pai_proto_pipe_pair_create(&pair) == PAI_OK);
  pai_proto_pipe_endpoint(pair, 0, &ta);
  pai_proto_pipe_endpoint(pair, 1, &tb);

  memset(&cbc, 0, sizeof(cbc));
  cbc.on_close = close_observer;
  memset(&cbs, 0, sizeof(cbs));
  cbs.on_hello = accept_hello;
  cbs.on_close = close_observer;

  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &ta, &cbc, &cc) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &tb, &cbs, &sc) == PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);

  {
    uint8_t pay[PAI_PROTO_U32_PAIR_SIZE];
    uint32_t plen = 0;
    CHECK(pai_proto_msg_encode_u32(pay, sizeof(pay), PAI_PROTO_CLOSE_NORMAL,
                                   &plen) == PAI_OK);
    CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_CLOSE, 0, 0, 0, pay,
                                  plen) == PAI_OK);
  }
  pump(&client, &server, 8);

  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_CLOSED);
  CHECK_EQ_UINT(sc.on_close, 1);
  CHECK_EQ_UINT(sc.reason, PAI_PROTO_CLOSE_NORMAL);
  /* client observes peer closure on the transport */
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_CLOSED);
  CHECK_EQ_UINT(cc.on_close, 1);
  CHECK_EQ_UINT(cc.reason, PAI_PROTO_CLOSE_PEER_GONE);

  (void)frames;
  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_pipe_pair_destroy(pair);
}

TEST_MAIN_END()
