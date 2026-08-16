/*
 * ProsperoAI — gateway remote bridge (whitepaper §24/§26) — impl
 *
 * Client side of the §24 exchanges over the TCP transport (§25):
 *   GENERATE #r -> ACCEPTED #r -> TOKEN #r* -> COMPLETE #r
 *   EMBED #r -> EMBEDDING #r (or ERROR #r)
 *
 * Connections are pooled per gateway entry (§24 advertises persistent
 * connections): the first exchange establishes one connection +
 * negotiated session which later exchanges reuse, with a transparent
 * reconnect once when the peer has closed it (idle timeout, payload
 * restart). The gateway serializes exchanges under the per-entry
 * lock, so a pool never runs two exchanges concurrently.
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
  if (request_id != c->request_id || c->complete || c->error) {
    return; /* unrelated stream, or already terminal */
  }
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

  if (frame->request_id != c->request_id || c->complete || c->error) {
    return PAI_OK; /* unrelated pipelined reply, or already terminal */
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

  if (frame->request_id != c->request_id || c->got || c->error) {
    return PAI_OK; /* unrelated pipelined reply, or already answered */
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

/* ------------------------------------------------------------------ */
/* Connection pool                                                     */
/* ------------------------------------------------------------------ */

typedef enum pai_remote_kind {
  PAI_REMOTE_KIND_GENERATE = 0,
  PAI_REMOTE_KIND_EMBED = 1
} pai_remote_kind_t;

struct pai_remote_pool {
  char host[256];
  uint16_t port;
  pai_proto_transport_t transport;
  pai_proto_conn_t conn;
  int valid;            /* transport + conn initialized               */
  int busy;             /* an exchange is in flight                   */
  pai_remote_kind_t kind; /* kind of the in-flight exchange           */
  remote_ctx_t gen;     /* state for the generate exchange            */
  embed_ctx_t emb;      /* state for the embedding exchange           */
};

pai_remote_pool_t *
pai_remote_pool_create(const char *host, uint16_t port) {
  pai_remote_pool_t *pool;

  if (host == NULL || host[0] == '\0' || port == 0) {
    return NULL;
  }
  {
    size_t n = strlen(host);
    if (n >= sizeof(pool->host)) {
      return NULL;
    }
    pool = (pai_remote_pool_t *)malloc(sizeof(*pool));
    if (pool == NULL) {
      return NULL;
    }
    memset(pool, 0, sizeof(*pool));
    memcpy(pool->host, host, n);
    pool->port = port;
  }
  return pool;
}

void
pai_remote_pool_destroy(pai_remote_pool_t *pool) {
  if (pool == NULL) {
    return;
  }
  if (pool->valid) {
    pai_proto_conn_destroy(&pool->conn);
    pai_proto_tcp_transport_destroy(&pool->transport);
    pool->valid = 0;
  }
  free(pool);
}

/* The pooled connection was initialized with these callbacks once;
 * they route inbound frames to the exchange state of the current
 * kind (exchanges are serialized, so at most one is live at a time). */
static pai_status_t
pool_on_message(void *user, const pai_proto_frame_t *frame) {
  pai_remote_pool_t *pool = (pai_remote_pool_t *)user;
  if (pool->kind == PAI_REMOTE_KIND_EMBED) {
    return embed_on_message(&pool->emb, frame);
  }
  return remote_on_message(&pool->gen, frame);
}

static void
pool_on_stream_data(void *user, uint64_t request_id, uint64_t session_id,
                    uint8_t kind, uint32_t seq, const uint8_t *data,
                    uint32_t len) {
  pai_remote_pool_t *pool = (pai_remote_pool_t *)user;
  if (pool->kind == PAI_REMOTE_KIND_GENERATE) {
    remote_on_stream_data(&pool->gen, request_id, session_id, kind, seq,
                          data, len);
  }
}

/*
 * Take the pooled connection for one exchange of `kind`. Reuses the
 * OPEN connection when available; otherwise (first use, or the peer
 * closed it between exchanges) reconnects and negotiates with the
 * capability the exchange needs. *deadline bounds the whole acquire
 * + exchange. The pool is marked busy on success.
 */
static pai_status_t
pool_acquire(pai_remote_pool_t *pool, pai_remote_kind_t kind,
             uint64_t timeout_ms, uint64_t *deadline) {
  pai_status_t st;
  uint32_t needed_cap = kind == PAI_REMOTE_KIND_GENERATE
                            ? PAI_PROTO_CAP_GENERATE
                            : PAI_PROTO_CAP_EMBED;

  if (pool->valid &&
      pai_proto_conn_state(&pool->conn) != PAI_PROTO_STATE_OPEN) {
    /* Stale: the peer closed between exchanges. Reconnect below. */
    pai_proto_conn_destroy(&pool->conn);
    pai_proto_tcp_transport_destroy(&pool->transport);
    pool->valid = 0;
  }
  if (!pool->valid) {
    pai_proto_callbacks_t cb;
    memset(&cb, 0, sizeof(cb));
    cb.on_message = pool_on_message;
    cb.on_stream_data = pool_on_stream_data;
    st = remote_connect_and_negotiate(pool->host, pool->port, needed_cap,
                                      timeout_ms, &cb, pool, &pool->transport,
                                      &pool->conn, deadline);
    if (st != PAI_OK) {
      return st;
    }
    pool->valid = 1;
  } else {
    /* Reused connection: caps were checked at negotiate time, but the
     * entry may now request a capability the peer never advertised. */
    *deadline = pai_proto_now_ns() + timeout_ms * 1000000ull;
    if (!(pai_proto_conn_negotiated_caps(&pool->conn) & needed_cap)) {
      return PAI_ERR_UNSUPPORTED;
    }
  }
  pool->kind = kind;
  pool->busy = 1;
  return PAI_OK;
}

static void
pool_release(pai_remote_pool_t *pool) {
  pool->busy = 0;
}

/*
 * One pooled generation attempt. `attempt` bounds the transparent
 * reconnect: a connection-level failure with nothing delivered (peer
 * closed before answering, or the send hit a dead socket) tears the
 * connection down and retries once on a fresh one. Deliveries that
 * already reached the response (relayed tokens) never retry — that
 * would duplicate output — and explicit ERROR frames and deadlines
 * are respected.
 */
static pai_status_t
pool_generate_attempt(pai_remote_pool_t *pool, const char *prompt,
                      const pai_proto_sampler_t *sampler,
                      void (*on_token)(const char *, void *), void *user,
                      uint32_t *out_tokens, uint64_t timeout_ms,
                      int attempt) {
  pai_status_t st;
  uint64_t deadline;
  uint64_t req;
  uint32_t frames = 0;
  uint32_t plen = 0;
  /* ~64 KB stack (u16 prompt + sampler trailer); the HTTP server runs
   * this on threads with a 1 MB default stack, so this is safe. */
  uint8_t pay[65535 + 2 + 4 + (uint32_t)sizeof(pai_proto_sampler_t)];

  st = pool_acquire(pool, PAI_REMOTE_KIND_GENERATE, timeout_ms, &deadline);
  if (st != PAI_OK) {
    return st;
  }
  memset(&pool->gen, 0, sizeof(pool->gen));
  pool->gen.on_token = on_token;
  pool->gen.user = user;

  st = pai_proto_msg_encode_generate2(pay, sizeof(pay), prompt, sampler,
                                      &plen);
  if (st != PAI_OK) {
    goto fail;
  }
  req = pai_proto_conn_new_request_id(&pool->conn);
  pool->gen.request_id = req;
  st = pai_proto_conn_send_raw(&pool->conn, PAI_PROTO_MSG_GENERATE, 0, req,
                               0, pay, plen);
  if (st != PAI_OK) {
    goto fail; /* dead socket (peer closed between exchanges) */
  }

  /* Relay TOKEN* until COMPLETE / ERROR / deadline / disconnect. The
   * terminal-frame check runs before the closed-state check because a
   * single poll can deliver COMPLETE and then consume the peer's EOF
   * (a payload that closes right after answering). */
  while (!pool->gen.complete && !pool->gen.error) {
    if (pai_proto_now_ns() >= deadline) {
      st = PAI_ERR_TIMEOUT;
      goto fail;
    }
    st = pai_proto_conn_poll(&pool->conn, &frames);
    if (st != PAI_OK) {
      goto fail;
    }
    if (pool->gen.complete || pool->gen.error) {
      break; /* terminal frame seen; the close is not a failure */
    }
    if (pai_proto_conn_state(&pool->conn) == PAI_PROTO_STATE_CLOSED) {
      st = PAI_ERR_IO; /* peer closed before COMPLETE */
      goto fail;
    }
  }
  if (pool->gen.error) {
    st = pool->gen.error_status; /* ERROR frame overrides any clean poll */
  } else if (!pool->gen.complete && st == PAI_OK) {
    st = PAI_ERR_IO; /* loop exited without terminal frame */
  }

fail:
  if (out_tokens != NULL) {
    *out_tokens = pool->gen.tokens;
  }
  pool_release(pool);

  /* Transparent reconnect (bounded: one retry per call). Nothing was
   * delivered, so the exchange is redone on a fresh connection. */
  if (st == PAI_ERR_IO && attempt < 2 && !pool->gen.error &&
      !pool->gen.complete && pool->gen.tokens == 0) {
    pai_proto_conn_destroy(&pool->conn);
    pai_proto_tcp_transport_destroy(&pool->transport);
    pool->valid = 0;
    return pool_generate_attempt(pool, prompt, sampler, on_token, user,
                                 out_tokens, timeout_ms, attempt + 1);
  }
  return st;
}

pai_status_t
pai_remote_pool_generate(pai_remote_pool_t *pool, const char *prompt,
                         const pai_proto_sampler_t *sampler,
                         void (*on_token)(const char *, void *), void *user,
                         uint32_t *out_tokens, uint64_t timeout_ms) {
  if (pool == NULL || prompt == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (pool->busy) {
    return PAI_ERR_INVALID_ARG; /* exchanges are serialized by the caller */
  }
  if (timeout_ms == 0) {
    timeout_ms = REMOTE_DEFAULT_TIMEOUT_MS;
  }
  if (strlen(prompt) > 65535) {
    return PAI_ERR_INVALID_ARG; /* v0 GENERATE prompt length is u16 */
  }
  return pool_generate_attempt(pool, prompt, sampler, on_token, user,
                               out_tokens, timeout_ms, 1);
}

/* One pooled embedding attempt — see pool_generate_attempt for the
 * retry contract (bounded, nothing-delivered only). */
static pai_status_t
pool_embed_attempt(pai_remote_pool_t *pool, const char *text,
                   uint32_t text_len, float **out_vec, uint32_t *out_dim,
                   uint64_t timeout_ms, int attempt) {
  pai_status_t st;
  uint64_t deadline;
  uint64_t req;
  uint32_t frames = 0;
  uint32_t plen = 0;
  /* ~64 KB stack (u16 text); the HTTP server thread stack is 1 MB. */
  uint8_t pay[65535u + 2u];

  st = pool_acquire(pool, PAI_REMOTE_KIND_EMBED, timeout_ms, &deadline);
  if (st != PAI_OK) {
    return st;
  }
  memset(&pool->emb, 0, sizeof(pool->emb));

  st = pai_proto_msg_encode_embed(pay, sizeof(pay), text, text_len, &plen);
  if (st != PAI_OK) {
    goto fail;
  }
  req = pai_proto_conn_new_request_id(&pool->conn);
  pool->emb.request_id = req;
  st = pai_proto_conn_send_raw(&pool->conn, PAI_PROTO_MSG_EMBED, 0, req, 0,
                               pay, plen);
  if (st != PAI_OK) {
    goto fail; /* dead socket (peer closed between exchanges) */
  }

  /* Wait for EMBEDDING / ERROR / deadline / disconnect. As with
   * generation, the reply can arrive in the same poll that consumes
   * the peer's EOF (a payload that closes right after answering). */
  while (!pool->emb.got && !pool->emb.error) {
    if (pai_proto_now_ns() >= deadline) {
      st = PAI_ERR_TIMEOUT;
      goto fail;
    }
    st = pai_proto_conn_poll(&pool->conn, &frames);
    if (st != PAI_OK) {
      goto fail;
    }
    if (pool->emb.got || pool->emb.error) {
      break; /* reply seen; the close is not a failure */
    }
    if (pai_proto_conn_state(&pool->conn) == PAI_PROTO_STATE_CLOSED) {
      st = PAI_ERR_IO; /* peer closed before the reply */
      goto fail;
    }
  }
  if (pool->emb.error) {
    st = pool->emb.error_status;
  } else if (!pool->emb.got && st == PAI_OK) {
    st = PAI_ERR_IO; /* loop exited without a reply */
  }

fail:
  if (st == PAI_OK) {
    *out_vec = pool->emb.vec;
    *out_dim = pool->emb.dim;
  } else {
    free(pool->emb.vec);
    pool->emb.vec = NULL;
    if (out_vec != NULL) {
      *out_vec = NULL;
    }
    if (out_dim != NULL) {
      *out_dim = 0;
    }
  }
  pool_release(pool);

  /* Transparent reconnect (bounded: one retry per call). */
  if (st == PAI_ERR_IO && attempt < 2 && !pool->emb.error &&
      !pool->emb.got) {
    pai_proto_conn_destroy(&pool->conn);
    pai_proto_tcp_transport_destroy(&pool->transport);
    pool->valid = 0;
    return pool_embed_attempt(pool, text, text_len, out_vec, out_dim,
                              timeout_ms, attempt + 1);
  }
  return st;
}

pai_status_t
pai_remote_pool_embed(pai_remote_pool_t *pool, const char *text,
                      uint32_t text_len, float **out_vec, uint32_t *out_dim,
                      uint64_t timeout_ms) {
  if (pool == NULL || text == NULL || text_len == 0 || out_vec == NULL ||
      out_dim == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (pool->busy) {
    return PAI_ERR_INVALID_ARG; /* exchanges are serialized by the caller */
  }
  if (text_len > 0xFFFFu) {
    return PAI_ERR_INVALID_ARG; /* v0 EMBED text length is u16 */
  }
  if (timeout_ms == 0) {
    timeout_ms = REMOTE_DEFAULT_TIMEOUT_MS;
  }
  return pool_embed_attempt(pool, text, text_len, out_vec, out_dim,
                            timeout_ms, 1);
}

/* ------------------------------------------------------------------ */
/* One-shot bridges (thin wrappers over a transient pool)              */
/* ------------------------------------------------------------------ */

pai_status_t
pai_gw_remote_generate(const char *host, uint16_t port, const char *prompt,
                       const pai_proto_sampler_t *sampler,
                       void (*on_token)(const char *, void *), void *user,
                       uint32_t *out_tokens, uint64_t timeout_ms) {
  pai_remote_pool_t *pool;
  pai_status_t st;

  if (host == NULL || port == 0 || prompt == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  pool = pai_remote_pool_create(host, port);
  if (pool == NULL) {
    return PAI_ERR_NOMEM;
  }
  st = pai_remote_pool_generate(pool, prompt, sampler, on_token, user,
                                out_tokens, timeout_ms);
  pai_remote_pool_destroy(pool);
  return st;
}

pai_status_t
pai_gw_remote_embed(const char *host, uint16_t port, const char *text,
                    uint32_t text_len, float **out_vec, uint32_t *out_dim,
                    uint64_t timeout_ms) {
  pai_remote_pool_t *pool;
  pai_status_t st;

  if (host == NULL || port == 0 || text == NULL || text_len == 0 ||
      out_vec == NULL || out_dim == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  pool = pai_remote_pool_create(host, port);
  if (pool == NULL) {
    return PAI_ERR_NOMEM;
  }
  st = pai_remote_pool_embed(pool, text, text_len, out_vec, out_dim,
                             timeout_ms);
  pai_remote_pool_destroy(pool);
  return st;
}
