/*
 * ProsperoAI — Prospero Protocol
 *
 * pai_proto_ping — connectivity health check for desktop tooling
 * (whitepaper §24/§25). Connects over the TCP transport, negotiates,
 * then measures `count` PING -> PONG round trips. The protocol core
 * auto-replies PONG to PING, so no application callbacks are needed on
 * the server side; this is purely a client-side loop.
 *
 * Host-side only (like transport_tcp.c). Carries its own monotonic
 * clock to stay dependency-free inside the core.
 */

#include <protocol/protocol.h>

#include <pai/log.h>

#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static uint64_t
proto_now_ns(void) {
  static LARGE_INTEGER freq;
  static int have_freq = 0;
  LARGE_INTEGER c;
  if (!have_freq) {
    QueryPerformanceFrequency(&freq);
    have_freq = 1;
  }
  QueryPerformanceCounter(&c);
  return (uint64_t)((double)c.QuadPart * 1e9 / (double)freq.QuadPart);
}
#else
#include <time.h>
static uint64_t
proto_now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#endif

/* ------------------------------------------------------------------ */
/* client-side PONG tracking                                           */
/* ------------------------------------------------------------------ */

typedef struct ping_ctx {
  uint64_t expected_req; /* request id of the in-flight ping          */
  int got_pong;
  uint64_t arrival_ns;   /* timestamp at PONG dispatch                */
} ping_ctx_t;

static pai_status_t
ping_on_message(void *user, const pai_proto_frame_t *frame) {
  ping_ctx_t *c = (ping_ctx_t *)user;

  if (frame->msg_type == PAI_PROTO_MSG_PONG &&
      (frame->flags & PAI_PROTO_FLAG_REPLY) &&
      frame->request_id == c->expected_req) {
    c->got_pong = 1;
    c->arrival_ns = proto_now_ns();
  }
  return PAI_OK;
}

/* Pump the connection until OPEN, CLOSED (refused/EOF) or timeout. */
static pai_status_t
wait_open(pai_proto_conn_t *conn, uint64_t timeout_ms) {
  uint64_t deadline = proto_now_ns() + timeout_ms * 1000000ull;
  uint32_t frames = 0;

  for (;;) {
    pai_status_t st = pai_proto_conn_poll(conn, &frames);
    if (st != PAI_OK) {
      return st;
    }
    if (pai_proto_conn_state(conn) == PAI_PROTO_STATE_OPEN) {
      return PAI_OK;
    }
    if (pai_proto_conn_state(conn) == PAI_PROTO_STATE_CLOSED) {
      return PAI_ERR_IO; /* refused or peer vanished mid-negotiation */
    }
    if (proto_now_ns() >= deadline) {
      return PAI_ERR_TIMEOUT;
    }
  }
}

/* ------------------------------------------------------------------ */
/* ping                                                                */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_ping(const char *host, uint16_t port, uint32_t count,
               pai_proto_ping_result_t *results, uint32_t *out_caps,
               uint64_t timeout_ms) {
  pai_proto_transport_t transport;
  pai_proto_conn_t conn;
  pai_proto_callbacks_t cb;
  ping_ctx_t ctx;
  pai_status_t st;
  uint32_t i;

  if (host == NULL || count == 0 || results == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (timeout_ms == 0) {
    timeout_ms = 2000;
  }

  memset(&transport, 0, sizeof(transport));
  memset(&conn, 0, sizeof(conn));
  memset(&ctx, 0, sizeof(ctx));

  st = pai_proto_tcp_connect(host, port, &transport);
  if (st != PAI_OK) {
    PAI_LOG_DEBUG_(PAI_SUB_PROTO, "ping: connect %s:%u failed (%s)\n", host,
                   (unsigned)port, pai_status_str(st));
    return st;
  }

  memset(&cb, 0, sizeof(cb));
  cb.on_message = ping_on_message;
  st = pai_proto_conn_init(&conn, PAI_PROTO_ROLE_CLIENT, PAI_PROTO_CAP_KNOWN,
                           &transport, &cb, &ctx);
  if (st != PAI_OK) {
    pai_proto_tcp_transport_destroy(&transport);
    return st;
  }

  st = pai_proto_conn_start(&conn);
  if (st == PAI_OK) {
    st = wait_open(&conn, timeout_ms);
  }
  if (st != PAI_OK || pai_proto_conn_state(&conn) != PAI_PROTO_STATE_OPEN) {
    if (st == PAI_OK || conn.nack_reason != 0) {
      /* Negotiation refused (HELLO_NACK): the core closed the conn and
       * recorded the reason (wait_open surfaces it as PAI_ERR_IO). */
      st = PAI_ERR_CAPABILITY;
    }
    pai_proto_conn_destroy(&conn);
    pai_proto_tcp_transport_destroy(&transport);
    return st;
  }
  if (out_caps != NULL) {
    *out_caps = pai_proto_conn_negotiated_caps(&conn);
  }

  for (i = 0; i < count; i++) {
    uint64_t req = pai_proto_conn_new_request_id(&conn);
    uint64_t t0 = proto_now_ns();
    uint64_t deadline = t0 + timeout_ms * 1000000ull;
    uint32_t frames = 0;

    results[i].rtt_ns = 0;
    results[i].lost = 1;
    ctx.expected_req = req;
    ctx.got_pong = 0;

    st = pai_proto_conn_send_raw(&conn, PAI_PROTO_MSG_PING, 0, req, 0, NULL,
                                 0);
    if (st != PAI_OK) {
      break; /* connection died mid-loop; remaining pings stay lost */
    }

    for (;;) {
      if (ctx.got_pong) {
        results[i].lost = 0;
        results[i].rtt_ns = ctx.arrival_ns - t0;
        break;
      }
      if (proto_now_ns() >= deadline) {
        break; /* lost */
      }
      st = pai_proto_conn_poll(&conn, &frames);
      if (st != PAI_OK || pai_proto_conn_state(&conn) == PAI_PROTO_STATE_CLOSED) {
        break;
      }
    }
  }

  pai_proto_conn_destroy(&conn);
  pai_proto_tcp_transport_destroy(&transport);
  return PAI_OK;
}
