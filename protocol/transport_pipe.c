/*
 * ProsperoAI — Prospero Protocol
 *
 * In-memory pipe pair transport: two growable FIFO directions with
 * endpoint transport vtables. Side A's sends land in a_to_b, which is
 * what side B's recv drains; the reverse for b_to_a.
 *
 * This is the loopback transport for host tests and local development
 * (whitepaper §25: TCP and Local/Internal are initial transports; the
 * protocol itself is transport-independent). A TCP transport plugs in
 * with the same pai_proto_transport_t interface.
 */

#include <protocol/protocol.h>

#include <stdlib.h>
#include <string.h>

typedef struct pai_proto_pipe {
  uint8_t *buf;
  uint32_t cap;
  uint32_t head; /* read offset */
  uint32_t len;  /* bytes buffered */
  int closed;
} pai_proto_pipe_t;

typedef struct pai_proto_pipe_endpoint {
  pai_proto_pipe_t *tx; /* this endpoint sends into tx */
  pai_proto_pipe_t *rx; /* this endpoint receives from rx */
} pai_proto_pipe_endpoint_t;

struct pai_proto_pipe_pair {
  pai_proto_pipe_t *a_to_b;
  pai_proto_pipe_t *b_to_a;
  pai_proto_pipe_endpoint_t ep_a;
  pai_proto_pipe_endpoint_t ep_b;
};

/* ------------------------------------------------------------------ */
/* pipe internals                                                      */
/* ------------------------------------------------------------------ */

static pai_status_t
pipe_push(pai_proto_pipe_t *pipe, const void *data, uint32_t nbytes) {
  const uint8_t *p = (const uint8_t *)data;

  if (pipe->closed) {
    return PAI_ERR_IO;
  }
  if (nbytes == 0) {
    return PAI_OK;
  }

  /* Compact when empty to avoid unbounded growth. */
  if (pipe->len == 0) {
    pipe->head = 0;
  }

  if (pipe->len + nbytes > pipe->cap) {
    uint32_t need = pipe->len + nbytes;
    uint32_t newcap = pipe->cap ? pipe->cap : 256u;
    uint8_t *nbuf;

    while (newcap < need) {
      newcap *= 2;
    }
    nbuf = (uint8_t *)malloc(newcap);
    if (!nbuf) {
      return PAI_ERR_NOMEM;
    }
    memcpy(nbuf, pipe->buf + pipe->head, pipe->len);
    free(pipe->buf);
    pipe->buf = nbuf;
    pipe->cap = newcap;
    pipe->head = 0;
  }

  memcpy(pipe->buf + pipe->head + pipe->len, p, nbytes);
  pipe->len += nbytes;
  return PAI_OK;
}

static pai_status_t
pipe_pull(pai_proto_pipe_t *pipe, void *data, uint32_t nbytes,
          uint32_t *out_read) {
  uint32_t n;

  *out_read = 0;
  if (pipe->len == 0) {
    if (pipe->closed) {
      return PAI_ERR_IO; /* EOF */
    }
    return PAI_OK; /* nothing available yet */
  }

  n = nbytes < pipe->len ? nbytes : pipe->len;
  memcpy(data, pipe->buf + pipe->head, n);
  pipe->head += n;
  pipe->len -= n;
  if (pipe->len == 0) {
    pipe->head = 0;
  }
  *out_read = n;
  return PAI_OK;
}

static pai_status_t
endpoint_send(void *ctx, const void *data, uint32_t nbytes) {
  pai_proto_pipe_endpoint_t *ep = (pai_proto_pipe_endpoint_t *)ctx;
  return pipe_push(ep->tx, data, nbytes);
}

static pai_status_t
endpoint_recv(void *ctx, void *data, uint32_t nbytes, uint32_t *out_read) {
  pai_proto_pipe_endpoint_t *ep = (pai_proto_pipe_endpoint_t *)ctx;
  return pipe_pull(ep->rx, data, nbytes, out_read);
}

static void
endpoint_close(void *ctx) {
  pai_proto_pipe_endpoint_t *ep = (pai_proto_pipe_endpoint_t *)ctx;
  ep->tx->closed = 1;
  ep->rx->closed = 1;
}

/* ------------------------------------------------------------------ */
/* pair API                                                            */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_pipe_pair_create(pai_proto_pipe_pair_t **out_pair) {
  pai_proto_pipe_pair_t *pair;

  if (!out_pair) {
    return PAI_ERR_INVALID_ARG;
  }

  pair = (pai_proto_pipe_pair_t *)calloc(1, sizeof(*pair));
  if (!pair) {
    return PAI_ERR_NOMEM;
  }
  pair->a_to_b = (pai_proto_pipe_t *)calloc(1, sizeof(*pair->a_to_b));
  pair->b_to_a = (pai_proto_pipe_t *)calloc(1, sizeof(*pair->b_to_a));
  if (!pair->a_to_b || !pair->b_to_a) {
    pai_proto_pipe_pair_destroy(pair);
    return PAI_ERR_NOMEM;
  }

  pair->ep_a.tx = pair->a_to_b;
  pair->ep_a.rx = pair->b_to_a;
  pair->ep_b.tx = pair->b_to_a;
  pair->ep_b.rx = pair->a_to_b;

  *out_pair = pair;
  return PAI_OK;
}

void
pai_proto_pipe_pair_destroy(pai_proto_pipe_pair_t *pair) {
  if (!pair) {
    return;
  }
  free(pair->a_to_b->buf);
  free(pair->a_to_b);
  free(pair->b_to_a->buf);
  free(pair->b_to_a);
  free(pair);
}

void
pai_proto_pipe_endpoint(const pai_proto_pipe_pair_t *pair, int which,
                        pai_proto_transport_t *out) {
  pai_proto_pipe_endpoint_t *ep;

  if (!pair || !out) {
    return;
  }
  ep = which == 0 ? (pai_proto_pipe_endpoint_t *)&pair->ep_a
                  : (pai_proto_pipe_endpoint_t *)&pair->ep_b;

  out->ctx = ep;
  out->send = endpoint_send;
  out->recv = endpoint_recv;
  out->close = endpoint_close;
}
