/*
 * ProsperoAI — gateway remote bridge (whitepaper §24/§26) — impl
 *
 * Client side of the §24 exchange over the TCP transport (§25):
 *   GENERATE #r -> ACCEPTED #r -> TOKEN #r* -> COMPLETE #r
 * Each gateway request opens its own connection (the gateway already
 * serializes generation per model entry, so a per-request connection
 * is the simplest correct v0 lifecycle).
 */

#include "remote.h"

#include <protocol/protocol.h>

#include <string.h>

#define REMOTE_DEFAULT_TIMEOUT_MS 30000ull
#define REMOTE_TOKEN_BUF 2048u

typedef struct remote_ctx {
  void (*on_token)(const char *token, void *user);
  void *user;
  uint32_t tokens;   /* relayed token count                            */
  int complete;      /* COMPLETE seen                                  */
  int error;         /* ERROR frame seen (aborts the stream)           */
  pai_status_t error_status;
} remote_ctx_t;

static void
remote_on_stream_data(void *user, uint64_t request_id, uint64_t session_id,
                      uint8_t kind, uint32_t seq, const uint8_t *data,
                      uint32_t len) {
  remote_ctx_t *c = (remote_ctx_t *)user;
  (void)request_id;
  (void)session_id;
  (void)seq;
  if (kind != PAI_PROTO_STREAM_GENERATE || c->on_token == NULL ||
      len == 0) {
    return;
  }
  /* The callback expects a NUL-terminated string; wire chunks are
   * length-delimited. Long chunks are relayed in slices. */
  {
    uint32_t off = 0;
    while (off < len) {
      uint32_t n = len - off;
      char buf[REMOTE_TOKEN_BUF];
      if (n >= sizeof(buf)) {
        n = (uint32_t)sizeof(buf) - 1;
      }
      memcpy(buf, data + off, n);
      buf[n] = '\0';
      c->on_token(buf, c->user);
      c->tokens++;
      off += n;
    }
  }
}

static pai_status_t
remote_on_message(void *user, const pai_proto_frame_t *frame) {
  remote_ctx_t *c = (remote_ctx_t *)user;

  switch (frame->msg_type) {
  case PAI_PROTO_MSG_ACCEPTED:
    break; /* the GENERATE stream may begin directly; no gate needed */
  case PAI_PROTO_MSG_COMPLETE:
    c->complete = 1;
    break;
  case PAI_PROTO_MSG_ERROR: {
    pai_status_t es = PAI_ERR_IO;
    char msg[128];
    (void)pai_proto_msg_decode_error(frame->payload, frame->payload_len, &es,
                                     msg, sizeof(msg));
    c->error = 1;
    c->error_status = es != PAI_OK ? es : PAI_ERR_IO;
    break;
  }
  default:
    break;
  }
  return PAI_OK;
}

pai_status_t
pai_gw_remote_generate(const char *host, uint16_t port, const char *prompt,
                       void (*on_token)(const char *, void *), void *user,
                       uint32_t *out_tokens, uint64_t timeout_ms) {
  pai_proto_transport_t transport;
  pai_proto_conn_t conn;
  pai_proto_callbacks_t cb;
  remote_ctx_t ctx;
  pai_status_t st;
  uint64_t deadline;
  uint64_t req;
  uint32_t frames = 0;
  uint32_t plen = 0;
  uint8_t pay[65535 + 2];

  if (host == NULL || port == 0 || prompt == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (timeout_ms == 0) {
    timeout_ms = REMOTE_DEFAULT_TIMEOUT_MS;
  }
  if (strlen(prompt) > 65535) {
    return PAI_ERR_INVALID_ARG; /* v0 GENERATE prompt length is u16 */
  }

  memset(&transport, 0, sizeof(transport));
  memset(&conn, 0, sizeof(conn));
  memset(&ctx, 0, sizeof(ctx));
  ctx.on_token = on_token;
  ctx.user = user;

  st = pai_proto_tcp_connect(host, port, &transport);
  if (st != PAI_OK) {
    return st;
  }

  memset(&cb, 0, sizeof(cb));
  cb.on_message = remote_on_message;
  cb.on_stream_data = remote_on_stream_data;
  st = pai_proto_conn_init(&conn, PAI_PROTO_ROLE_CLIENT, PAI_PROTO_CAP_KNOWN,
                           &transport, &cb, &ctx);
  if (st != PAI_OK) {
    pai_proto_tcp_transport_destroy(&transport);
    return st;
  }

  deadline = pai_proto_now_ns() + timeout_ms * 1000000ull;

  /* Negotiate (HELLO -> HELLO_ACK). */
  st = pai_proto_conn_start(&conn);
  while (st == PAI_OK &&
         pai_proto_conn_state(&conn) == PAI_PROTO_STATE_NEGOTIATING) {
    if (pai_proto_now_ns() >= deadline) {
      st = PAI_ERR_TIMEOUT;
      break;
    }
    st = pai_proto_conn_poll(&conn, &frames);
  }
  if (st == PAI_OK &&
      pai_proto_conn_state(&conn) != PAI_PROTO_STATE_OPEN) {
    /* Refused (HELLO_NACK) or peer vanished mid-negotiation. */
    st = conn.nack_reason != 0 ? PAI_ERR_CAPABILITY : PAI_ERR_IO;
  }
  if (st != PAI_OK) {
    goto out;
  }

  if (!(pai_proto_conn_negotiated_caps(&conn) & PAI_PROTO_CAP_GENERATE)) {
    st = PAI_ERR_UNSUPPORTED;
    goto out;
  }

  /* GENERATE(prompt). */
  st = pai_proto_msg_encode_generate(pay, sizeof(pay), prompt, &plen);
  if (st != PAI_OK) {
    goto out;
  }
  req = pai_proto_conn_new_request_id(&conn);
  st = pai_proto_conn_send_raw(&conn, PAI_PROTO_MSG_GENERATE, 0, req, 0, pay,
                               plen);
  if (st != PAI_OK) {
    goto out;
  }

  /* Relay TOKEN* until COMPLETE / ERROR / deadline / disconnect. */
  while (!ctx.complete && !ctx.error) {
    if (pai_proto_now_ns() >= deadline) {
      st = PAI_ERR_TIMEOUT;
      break;
    }
    st = pai_proto_conn_poll(&conn, &frames);
    if (st != PAI_OK) {
      break;
    }
    if (pai_proto_conn_state(&conn) == PAI_PROTO_STATE_CLOSED) {
      st = PAI_ERR_IO; /* peer closed before COMPLETE */
      break;
    }
  }
  if (ctx.error) {
    st = ctx.error_status; /* ERROR frame overrides any clean poll */
  } else if (!ctx.complete && st == PAI_OK) {
    st = PAI_ERR_IO; /* loop exited without terminal frame */
  }

out:
  if (out_tokens != NULL) {
    *out_tokens = ctx.tokens;
  }
  pai_proto_conn_destroy(&conn);
  pai_proto_tcp_transport_destroy(&transport);
  return st;
}
