/*
 * ProsperoAI — gateway remote bridge (whitepaper §24/§26) — impl
 *
 * Client side of the §24 exchanges over the TCP transport (§25):
 *   GENERATE #r -> ACCEPTED #r -> TOKEN #r* -> COMPLETE #r
 *   EMBED #r -> EMBEDDING #r (or ERROR #r)
 * Each gateway request opens its own connection (the gateway already
 * serializes work per model entry, so a per-request connection is the
 * simplest correct v0 lifecycle).
 */

#include "remote.h"

#include <stdlib.h>
#include <string.h>

#define REMOTE_DEFAULT_TIMEOUT_MS 30000ull
#define REMOTE_TOKEN_BUF 2048u

/* ------------------------------------------------------------------ */
/* shared connect + negotiate                                          */
/* ------------------------------------------------------------------ */

/*
 * Connect to host:port and negotiate HELLO -> HELLO_ACK. On success
 * the connection is OPEN and *deadline is set; on failure both the
 * connection and the transport are torn down here and the mapped
 * status is returned (the caller must not touch them again):
 *   PAI_ERR_IO           connect failure or peer vanished
 *   PAI_ERR_CAPABILITY   the server refused negotiation (HELLO_NACK)
 *   PAI_ERR_TIMEOUT      negotiation exceeded timeout_ms
 *   PAI_ERR_UNSUPPORTED  negotiated caps lack `needed_cap`
 */
static pai_status_t
remote_connect_and_negotiate(const char *host, uint16_t port,
                             uint32_t needed_cap, uint64_t timeout_ms,
                             const pai_proto_callbacks_t *callbacks,
                             void *user, pai_proto_transport_t *transport,
                             pai_proto_conn_t *conn, uint64_t *deadline) {
  pai_status_t st;
  uint32_t frames = 0;

  memset(transport, 0, sizeof(*transport));
  memset(conn, 0, sizeof(*conn));

  st = pai_proto_tcp_connect(host, port, transport);
  if (st != PAI_OK) {
    return st;
  }
  st = pai_proto_conn_init(conn, PAI_PROTO_ROLE_CLIENT, PAI_PROTO_CAP_KNOWN,
                           transport, callbacks, user);
  if (st != PAI_OK) {
    pai_proto_tcp_transport_destroy(transport);
    return st;
  }

  *deadline = pai_proto_now_ns() + timeout_ms * 1000000ull;

  st = pai_proto_conn_start(conn);
  while (st == PAI_OK &&
         pai_proto_conn_state(conn) == PAI_PROTO_STATE_NEGOTIATING) {
    if (pai_proto_now_ns() >= *deadline) {
      st = PAI_ERR_TIMEOUT;
      break;
    }
    st = pai_proto_conn_poll(conn, &frames);
  }
  if (st == PAI_OK && pai_proto_conn_state(conn) != PAI_PROTO_STATE_OPEN) {
    /* Refused (HELLO_NACK) or peer vanished mid-negotiation. */
    st = conn->nack_reason != 0 ? PAI_ERR_CAPABILITY : PAI_ERR_IO;
  }
  if (st != PAI_OK) {
    pai_proto_conn_destroy(conn);
    pai_proto_tcp_transport_destroy(transport);
    return st;
  }
  if (!(pai_proto_conn_negotiated_caps(conn) & needed_cap)) {
    pai_proto_conn_destroy(conn);
    pai_proto_tcp_transport_destroy(transport);
    return PAI_ERR_UNSUPPORTED;
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Generation                                                          */
/* ------------------------------------------------------------------ */

typedef struct remote_ctx {
  void (*on_token)(const char *token, void *user);
  void *user;
  uint64_t request_id; /* expected reply correlation                   */
  uint32_t tokens;     /* relayed token count                           */
  int complete;        /* COMPLETE seen                                 */
  int error;           /* ERROR frame seen (aborts the stream)          */
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

  if (frame->request_id != c->request_id) {
    return PAI_OK; /* unrelated pipelined reply */
  }
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
                       const pai_proto_sampler_t *sampler,
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
  /* ~64 KB stack (u16 prompt + sampler trailer); the HTTP server runs
   * this on threads with a 1 MB default stack, so this is safe. */
  uint8_t pay[65535 + 2 + 4 + (uint32_t)sizeof(pai_proto_sampler_t)];

  if (host == NULL || port == 0 || prompt == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (timeout_ms == 0) {
    timeout_ms = REMOTE_DEFAULT_TIMEOUT_MS;
  }
  if (strlen(prompt) > 65535) {
    return PAI_ERR_INVALID_ARG; /* v0 GENERATE prompt length is u16 */
  }

  memset(&ctx, 0, sizeof(ctx));
  ctx.on_token = on_token;
  ctx.user = user;

  memset(&cb, 0, sizeof(cb));
  cb.on_message = remote_on_message;
  cb.on_stream_data = remote_on_stream_data;
  st = remote_connect_and_negotiate(host, port, PAI_PROTO_CAP_GENERATE,
                                    timeout_ms, &cb, &ctx, &transport, &conn,
                                    &deadline);
  if (st != PAI_OK) {
    return st;
  }

  /* GENERATE(prompt[, sampler]). */
  st = pai_proto_msg_encode_generate2(pay, sizeof(pay), prompt, sampler,
                                      &plen);
  if (st != PAI_OK) {
    goto out;
  }
  req = pai_proto_conn_new_request_id(&conn);
  ctx.request_id = req;
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

/* ------------------------------------------------------------------ */
/* Embeddings                                                          */
/* ------------------------------------------------------------------ */

typedef struct embed_ctx {
  float *vec;           /* malloc'd; set on a matching EMBEDDING     */
  uint32_t dim;
  uint64_t request_id;  /* expected reply correlation                 */
  int got;
  int error;
  pai_status_t error_status;
} embed_ctx_t;

static pai_status_t
embed_on_message(void *user, const pai_proto_frame_t *frame) {
  embed_ctx_t *c = (embed_ctx_t *)user;
  const float *values;
  uint32_t dim;

  if (frame->request_id != c->request_id) {
    return PAI_OK; /* unrelated pipelined reply */
  }
  switch (frame->msg_type) {
  case PAI_PROTO_MSG_EMBEDDING:
    /* Decode + copy now: the payload pointer is only valid until the
     * next poll. The codec already bounds dim by MAX_EMBED_DIM. */
    if (pai_proto_msg_decode_embedding(frame->payload, frame->payload_len,
                                       &values, &dim) != PAI_OK) {
      c->error = 1;
      c->error_status = PAI_ERR_PROTOCOL;
      break;
    }
    {
      float *vec = (float *)malloc((size_t)dim * sizeof(float));
      if (vec == NULL) {
        c->error = 1;
        c->error_status = PAI_ERR_NOMEM;
        break;
      }
      memcpy(vec, values, (size_t)dim * sizeof(float));
      c->vec = vec;
      c->dim = dim;
      c->got = 1;
    }
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
pai_gw_remote_embed(const char *host, uint16_t port, const char *text,
                    uint32_t text_len, float **out_vec, uint32_t *out_dim,
                    uint64_t timeout_ms) {
  pai_proto_transport_t transport;
  pai_proto_conn_t conn;
  pai_proto_callbacks_t cb;
  embed_ctx_t ctx;
  pai_status_t st;
  uint64_t deadline;
  uint64_t req;
  uint32_t frames = 0;
  uint32_t plen = 0;
  /* ~64 KB stack (u16 text); the HTTP server thread stack is 1 MB. */
  uint8_t pay[65535u + 2u];

  if (host == NULL || port == 0 || text == NULL || text_len == 0 ||
      out_vec == NULL || out_dim == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (text_len > 0xFFFFu) {
    return PAI_ERR_INVALID_ARG; /* v0 EMBED text length is u16 */
  }
  if (timeout_ms == 0) {
    timeout_ms = REMOTE_DEFAULT_TIMEOUT_MS;
  }

  memset(&ctx, 0, sizeof(ctx));

  memset(&cb, 0, sizeof(cb));
  cb.on_message = embed_on_message;
  st = remote_connect_and_negotiate(host, port, PAI_PROTO_CAP_EMBED,
                                    timeout_ms, &cb, &ctx, &transport, &conn,
                                    &deadline);
  if (st != PAI_OK) {
    return st;
  }

  /* EMBED(text). */
  st = pai_proto_msg_encode_embed(pay, sizeof(pay), text, text_len, &plen);
  if (st != PAI_OK) {
    goto out;
  }
  req = pai_proto_conn_new_request_id(&conn);
  ctx.request_id = req;
  st = pai_proto_conn_send_raw(&conn, PAI_PROTO_MSG_EMBED, 0, req, 0, pay,
                               plen);
  if (st != PAI_OK) {
    goto out;
  }

  /* Wait for EMBEDDING / ERROR / deadline / disconnect. */
  while (!ctx.got && !ctx.error) {
    if (pai_proto_now_ns() >= deadline) {
      st = PAI_ERR_TIMEOUT;
      break;
    }
    st = pai_proto_conn_poll(&conn, &frames);
    if (st != PAI_OK) {
      break;
    }
    if (pai_proto_conn_state(&conn) == PAI_PROTO_STATE_CLOSED) {
      st = PAI_ERR_IO; /* peer closed before the reply */
      break;
    }
  }
  if (ctx.error) {
    st = ctx.error_status;
  } else if (!ctx.got && st == PAI_OK) {
    st = PAI_ERR_IO; /* loop exited without a reply */
  }

out:
  if (st == PAI_OK) {
    *out_vec = ctx.vec;
    *out_dim = ctx.dim;
  } else {
    free(ctx.vec);
    if (out_vec != NULL) {
      *out_vec = NULL;
    }
    if (out_dim != NULL) {
      *out_dim = 0;
    }
  }
  pai_proto_conn_destroy(&conn);
  pai_proto_tcp_transport_destroy(&transport);
  return st;
}
