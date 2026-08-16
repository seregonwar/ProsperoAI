#include "test.h"

#include <protocol/protocol.h>

#include <string.h>

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

static void
tcp_pump(pai_proto_conn_t *client, pai_proto_conn_t *server, int iters) {
  uint32_t frames = 0;
  for (int i = 0; i < iters; i++) {
    CHECK(pai_proto_conn_poll(client, &frames) == PAI_OK);
    CHECK(pai_proto_conn_poll(server, &frames) == PAI_OK);
  }
}

static uint32_t
accept_hello(void *user, uint32_t remote_caps) {
  (void)user;
  (void)remote_caps;
  return PAI_PROTO_CAP_KNOWN;
}

/* ------------------------------------------------------------------ */
/* GENERATE server + client (whitepaper §24 example, over TCP)         */
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

  CHECK(pai_proto_conn_send_raw(s->conn, PAI_PROTO_MSG_ACCEPTED,
                                PAI_PROTO_FLAG_REPLY, frame->request_id,
                                s->session_id, NULL, 0) == PAI_OK);

  /* Stream the prompt back as 2-char tokens. */
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
  if (kind != PAI_PROTO_STREAM_GENERATE || seq != c->expected_seq) {
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
  } else if (frame->msg_type == PAI_PROTO_MSG_COMPLETE) {
    c->complete = 1;
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* PING/PONG + CLOSE observers                                         */
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
/* full §24 GENERATE exchange over a real TCP socket                   */
/* ------------------------------------------------------------------ */

{
  static const char prompt[] = "Hello world!";
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_transport_t t_server, t_client;
  pai_proto_conn_t server, client;
  pai_proto_callbacks_t cbc, cbs;
  gen_server_t gs;
  gen_client_t gc;
  uint16_t port = 0;
  uint8_t pay[128];
  uint32_t plen = 0;

  memset(&gs, 0, sizeof(gs));
  memset(&gc, 0, sizeof(gc));
  gc.expected_seq = 0;
  gc.seq_ok = 1;

  CHECK(pai_proto_tcp_listen(&lst, "127.0.0.1", 0) == PAI_OK);
  CHECK(lst != NULL);
  CHECK(pai_proto_tcp_listener_port(lst, &port) == PAI_OK);
  CHECK(port != 0);

  /* The handshake completes into the backlog, so connect first, then
   * accept. */
  CHECK(pai_proto_tcp_connect("127.0.0.1", port, &t_client) == PAI_OK);
  CHECK(pai_proto_tcp_accept(lst, &t_server) == PAI_OK);

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
                            &t_client, &cbc, &gc) == PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &t_server, &cbs, &gs) ==
        PAI_OK);
  gs.conn = &server;

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  tcp_pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_OPEN);

  /* GENERATE #42 — the whitepaper §24 diagram, over TCP. */
  CHECK(pai_proto_msg_encode_generate(pay, sizeof(pay), prompt, &plen) ==
        PAI_OK);
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_GENERATE, 0, 42, 0,
                                pay, plen) == PAI_OK);

  tcp_pump(&client, &server, 8);

  CHECK_EQ_UINT(gs.got_generate, 1);
  CHECK(strcmp(gs.prompt, prompt) == 0);
  CHECK_EQ_UINT(gs.session_id, 1);
  CHECK_EQ_UINT(gc.accepted, 1);
  CHECK_EQ_UINT(gc.complete, 1);
  CHECK_EQ_UINT(gc.begin_count, 1);
  CHECK_EQ_UINT(gc.end_count, 1);
  CHECK_EQ_UINT(gc.seq_ok, 1);
  CHECK_EQ_UINT(gc.out_len, strlen(prompt));
  CHECK(memcmp(gc.out, prompt, strlen(prompt)) == 0);

  /* Graceful close: client sends CLOSE, server ACKs by closing. */
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_CLOSE, 0, 0, 0, NULL,
                                0) == PAI_OK);
  tcp_pump(&client, &server, 4);
  CHECK(pai_proto_conn_state(&server) == PAI_PROTO_STATE_CLOSED);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_CLOSED);

  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_tcp_transport_destroy(&t_client);
  pai_proto_tcp_transport_destroy(&t_server);
  pai_proto_tcp_listener_destroy(lst);
}

/* ------------------------------------------------------------------ */
/* pipelined PING/PONG over TCP                                        */
/* ------------------------------------------------------------------ */

{
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_transport_t t_server, t_client;
  pai_proto_conn_t server, client;
  pai_proto_callbacks_t cbc, cbs;
  ping_client_t pc;
  uint16_t port = 0;

  memset(&pc, 0, sizeof(pc));
  CHECK(pai_proto_tcp_listen(&lst, "127.0.0.1", 0) == PAI_OK);
  CHECK(pai_proto_tcp_listener_port(lst, &port) == PAI_OK);
  CHECK(pai_proto_tcp_connect("127.0.0.1", port, &t_client) == PAI_OK);
  CHECK(pai_proto_tcp_accept(lst, &t_server) == PAI_OK);

  memset(&cbc, 0, sizeof(cbc));
  cbc.on_message = ping_on_message;
  memset(&cbs, 0, sizeof(cbs));
  cbs.on_hello = accept_hello;

  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &t_client, &cbc, &pc) ==
        PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &t_server, &cbs, NULL) ==
        PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  tcp_pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);

  /* three in-flight pings (pipelining over TCP) */
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 100, 0, NULL,
                                0) == PAI_OK);
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 101, 0, NULL,
                                0) == PAI_OK);
  CHECK(pai_proto_conn_send_raw(&client, PAI_PROTO_MSG_PING, 0, 102, 0, NULL,
                                0) == PAI_OK);

  tcp_pump(&client, &server, 8);

  CHECK_EQ_UINT(pc.npongs, 3);
  CHECK_EQ_UINT(pc.pongs[0], 100);
  CHECK_EQ_UINT(pc.pongs[1], 101);
  CHECK_EQ_UINT(pc.pongs[2], 102);

  pai_proto_conn_destroy(&client);
  pai_proto_conn_destroy(&server);
  pai_proto_tcp_transport_destroy(&t_client);
  pai_proto_tcp_transport_destroy(&t_server);
  pai_proto_tcp_listener_destroy(lst);
}

/* ------------------------------------------------------------------ */
/* transport EOF without CLOSE -> PAI_PROTO_CLOSE_PEER_GONE            */
/* ------------------------------------------------------------------ */

{
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_transport_t t_server, t_client;
  pai_proto_conn_t server, client;
  pai_proto_callbacks_t cbc, cbs;
  close_client_t cc;
  uint16_t port = 0;

  memset(&cc, 0, sizeof(cc));
  CHECK(pai_proto_tcp_listen(&lst, "127.0.0.1", 0) == PAI_OK);
  CHECK(pai_proto_tcp_listener_port(lst, &port) == PAI_OK);
  CHECK(pai_proto_tcp_connect("127.0.0.1", port, &t_client) == PAI_OK);
  CHECK(pai_proto_tcp_accept(lst, &t_server) == PAI_OK);

  memset(&cbc, 0, sizeof(cbc));
  cbc.on_close = close_observer;
  memset(&cbs, 0, sizeof(cbs));
  cbs.on_hello = accept_hello;

  CHECK(pai_proto_conn_init(&client, PAI_PROTO_ROLE_CLIENT,
                            PAI_PROTO_CAP_KNOWN, &t_client, &cbc, &cc) ==
        PAI_OK);
  CHECK(pai_proto_conn_init(&server, PAI_PROTO_ROLE_SERVER,
                            PAI_PROTO_CAP_KNOWN, &t_server, &cbs, NULL) ==
        PAI_OK);

  CHECK(pai_proto_conn_start(&client) == PAI_OK);
  tcp_pump(&client, &server, 8);
  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_OPEN);

  /* Server vanishes without a CLOSE frame; the client must observe
   * EOF on the transport. (The server conn is gone, so pump only the
   * client side here.) */
  pai_proto_conn_destroy(&server);
  pai_proto_tcp_transport_destroy(&t_server);
  {
    uint32_t frames = 0;
    CHECK(pai_proto_conn_poll(&client, &frames) == PAI_OK);
  }

  CHECK(pai_proto_conn_state(&client) == PAI_PROTO_STATE_CLOSED);
  CHECK_EQ_UINT(cc.on_close, 1);
  CHECK_EQ_UINT(cc.reason, PAI_PROTO_CLOSE_PEER_GONE);

  pai_proto_conn_destroy(&client);
  pai_proto_tcp_transport_destroy(&t_client);
  pai_proto_tcp_listener_destroy(lst);
}

/* ------------------------------------------------------------------ */
/* connecting to a dead port fails cleanly                             */
/* ------------------------------------------------------------------ */

{
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_transport_t t;
  uint16_t port = 0;

  /* Bind, learn the port, then release it — nothing is listening. */
  CHECK(pai_proto_tcp_listen(&lst, "127.0.0.1", 0) == PAI_OK);
  CHECK(pai_proto_tcp_listener_port(lst, &port) == PAI_OK);
  pai_proto_tcp_listener_destroy(lst);

  CHECK(pai_proto_tcp_connect("127.0.0.1", port, &t) == PAI_ERR_IO);
  CHECK(pai_proto_tcp_connect("nonexistent.invalid", 9, &t) == PAI_ERR_IO);
}

/* ------------------------------------------------------------------ */
/* endpoint destroy is idempotent (safe after conn double-close)       */
/* ------------------------------------------------------------------ */

{
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_transport_t t;
  uint16_t port = 0;

  CHECK(pai_proto_tcp_listen(&lst, "127.0.0.1", 0) == PAI_OK);
  CHECK(pai_proto_tcp_listener_port(lst, &port) == PAI_OK);
  CHECK(pai_proto_tcp_connect("127.0.0.1", port, &t) == PAI_OK);

  /* close() twice is safe (conn layer double-close), destroy frees. */
  t.close(t.ctx);
  t.close(t.ctx);
  pai_proto_tcp_transport_destroy(&t);
  pai_proto_tcp_transport_destroy(&t); /* second destroy is a no-op */
  CHECK(t.ctx == NULL);

  /* destroy of a never-connected transport is a no-op */
  {
    pai_proto_transport_t zero;
    memset(&zero, 0, sizeof(zero));
    pai_proto_tcp_transport_destroy(&zero);
    pai_proto_tcp_transport_destroy(NULL);
  }

  pai_proto_tcp_listener_destroy(lst);
}

TEST_MAIN_END()
