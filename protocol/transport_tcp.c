/*
 * ProsperoAI — Prospero Protocol
 *
 * TCP transport (whitepaper §25 — the desktop <-> PS5 link). Blocking
 * sockets with SO_RCVTIMEO (100 ms) keep the poll loop moving:
 *
 *   bytes > 0            -> PAI_OK, *out_read = bytes
 *   timeout / wouldblock -> PAI_OK, *out_read = 0    (nothing yet)
 *   peer closed          -> PAI_ERR_IO, *out_read = 0 (EOF)
 *
 * send() is a blocking full-write loop. Host-side only; gateway/gwsys.c
 * has a separate socket layer because its recv contract differs
 * (blocking, thread-per-connection).
 */

#include <protocol/protocol.h>

#include <pai/log.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET tcp_sock_t;
#define TCP_SOCK_INVALID INVALID_SOCKET
#define TCP_SOCK_ERR(e) WSAGetLastError()
#define TCP_ERR_TIMEOUT WSAETIMEDOUT
#define TCP_ERR_INTR WSAEINTR
#define TCP_ERR_BLOCK WSAEWOULDBLOCK
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int tcp_sock_t;
#define TCP_SOCK_INVALID (-1)
#define TCP_SOCK_ERR(e) errno
#define TCP_ERR_TIMEOUT EAGAIN
#define TCP_ERR_INTR EINTR
#define TCP_ERR_BLOCK EWOULDBLOCK
#endif

/* How long recv() waits for bytes before returning "nothing yet". */
#define PAI_TCP_RECV_TIMEOUT_MS 100

/* ------------------------------------------------------------------ */
/* socket helpers                                                      */
/* ------------------------------------------------------------------ */

static void
tcp_sock_close(tcp_sock_t s) {
  if (s == TCP_SOCK_INVALID) {
    return;
  }
#ifdef _WIN32
  closesocket(s);
#else
  close(s);
#endif
}

static int
tcp_sys_ready(void) {
#ifdef _WIN32
  static int done = 0;
  static int ok = 0;
  if (!done) {
    WSADATA wsa;
    ok = WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
    done = 1;
  }
  return ok;
#else
  return 1;
#endif
}

static void
tcp_set_rcv_timeout(tcp_sock_t s) {
#ifdef _WIN32
  DWORD tv = PAI_TCP_RECV_TIMEOUT_MS;
  (void)setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
#else
  struct timeval tv;
  tv.tv_sec = PAI_TCP_RECV_TIMEOUT_MS / 1000;
  tv.tv_usec = (PAI_TCP_RECV_TIMEOUT_MS % 1000) * 1000;
  (void)setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
}

/* Resolve + open a connected stream socket. */
static tcp_sock_t
tcp_socket_connect(const char *host, uint16_t port) {
  tcp_sock_t s = TCP_SOCK_INVALID;
  struct addrinfo hints;
  struct addrinfo *res = NULL;
  struct addrinfo *ai;
  char portstr[16];
  int ok = 0;

  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);

  if (getaddrinfo(host, portstr, &hints, &res) != 0 || res == NULL) {
    if (res) {
      freeaddrinfo(res);
    }
    return TCP_SOCK_INVALID;
  }

  for (ai = res; ai != NULL; ai = ai->ai_next) {
    s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (s == TCP_SOCK_INVALID) {
      continue;
    }
    if (connect(s, ai->ai_addr, (int)ai->ai_addrlen) == 0) {
      ok = 1;
      break;
    }
    tcp_sock_close(s);
  }
  freeaddrinfo(res);
  return ok ? s : TCP_SOCK_INVALID;
}

/* ------------------------------------------------------------------ */
/* endpoint transport vtable                                           */
/* ------------------------------------------------------------------ */

typedef struct pai_proto_tcp_ep {
  tcp_sock_t sock;
  int closed;
} pai_proto_tcp_ep_t;

struct pai_proto_tcp_listener {
  tcp_sock_t sock;
  int closed;
};

static pai_status_t
tcp_ep_send(void *ctx, const void *data, uint32_t nbytes) {
  pai_proto_tcp_ep_t *ep = (pai_proto_tcp_ep_t *)ctx;
  const char *p = (const char *)data;
  uint32_t off = 0;

  if (ep == NULL || ep->closed || ep->sock == TCP_SOCK_INVALID) {
    return PAI_ERR_IO;
  }
  while (off < nbytes) {
    int n = (int)send(ep->sock, p + off, (int)(nbytes - off), 0);
    if (n > 0) {
      off += (uint32_t)n;
      continue;
    }
    if (TCP_SOCK_ERR(errno) == TCP_ERR_INTR) {
      continue;
    }
    return PAI_ERR_IO;
  }
  return PAI_OK;
}

static pai_status_t
tcp_ep_recv(void *ctx, void *data, uint32_t nbytes, uint32_t *out_read) {
  pai_proto_tcp_ep_t *ep = (pai_proto_tcp_ep_t *)ctx;
  int n;

  *out_read = 0;
  if (ep == NULL || ep->closed || ep->sock == TCP_SOCK_INVALID) {
    return PAI_ERR_IO;
  }
  for (;;) {
    int e;
    n = (int)recv(ep->sock, (char *)data, (int)nbytes, 0);
    if (n > 0) {
      *out_read = (uint32_t)n;
      return PAI_OK;
    }
    if (n == 0) {
      return PAI_ERR_IO; /* peer closed: EOF */
    }
    e = TCP_SOCK_ERR(errno);
    if (e == TCP_ERR_INTR) {
      continue;
    }
    if (e == TCP_ERR_TIMEOUT || e == TCP_ERR_BLOCK) {
      return PAI_OK; /* nothing available yet */
    }
    return PAI_ERR_IO;
  }
}

static void
tcp_ep_close(void *ctx) {
  pai_proto_tcp_ep_t *ep = (pai_proto_tcp_ep_t *)ctx;
  if (ep != NULL && !ep->closed) {
    ep->closed = 1;
    tcp_sock_close(ep->sock);
    ep->sock = TCP_SOCK_INVALID;
  }
}

/* Build the transport vtable over a heap endpoint owning `sock`. */
static pai_status_t
tcp_ep_wrap(tcp_sock_t sock, pai_proto_transport_t *out) {
  pai_proto_tcp_ep_t *ep;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  ep = (pai_proto_tcp_ep_t *)calloc(1, sizeof(*ep));
  if (ep == NULL) {
    tcp_sock_close(sock);
    return PAI_ERR_NOMEM;
  }
  ep->sock = sock;
  tcp_set_rcv_timeout(sock);

  out->ctx = ep;
  out->send = tcp_ep_send;
  out->recv = tcp_ep_recv;
  out->close = tcp_ep_close;
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* client + listener APIs                                              */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_tcp_connect(const char *host, uint16_t port,
                      pai_proto_transport_t *out) {
  tcp_sock_t s;

  if (host == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!tcp_sys_ready()) {
    return PAI_ERR_INIT;
  }
  s = tcp_socket_connect(host, port);
  if (s == TCP_SOCK_INVALID) {
    PAI_LOG_DEBUG_(PAI_SUB_PROTO, "tcp connect to %s:%u failed\n", host,
                   (unsigned)port);
    return PAI_ERR_IO;
  }
  return tcp_ep_wrap(s, out);
}

pai_status_t
pai_proto_tcp_listen(pai_proto_tcp_listener_t **out, const char *host,
                     uint16_t port) {
  pai_proto_tcp_listener_t *l;
  struct addrinfo hints;
  struct addrinfo *res = NULL;
  struct addrinfo *ai;
  char portstr[16];
  int one = 1;
  tcp_sock_t s = TCP_SOCK_INVALID;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (!tcp_sys_ready()) {
    return PAI_ERR_INIT;
  }

  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = host == NULL ? AI_PASSIVE : 0;
  snprintf(portstr, sizeof(portstr), "%u", (unsigned)port);

  if (getaddrinfo(host, portstr, &hints, &res) != 0 || res == NULL) {
    if (res) {
      freeaddrinfo(res);
    }
    return PAI_ERR_IO;
  }

  for (ai = res; ai != NULL; ai = ai->ai_next) {
    s = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (s == TCP_SOCK_INVALID) {
      continue;
    }
    (void)setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one,
                     sizeof(one));
    if (bind(s, ai->ai_addr, (int)ai->ai_addrlen) == 0 &&
        listen(s, 8) == 0) {
      break;
    }
    tcp_sock_close(s);
    s = TCP_SOCK_INVALID;
  }
  freeaddrinfo(res);

  if (s == TCP_SOCK_INVALID) {
    PAI_LOG_DEBUG_(PAI_SUB_PROTO, "tcp listen on %s:%u failed\n",
                   host ? host : "*", (unsigned)port);
    return PAI_ERR_IO;
  }

  l = (pai_proto_tcp_listener_t *)calloc(1, sizeof(*l));
  if (l == NULL) {
    tcp_sock_close(s);
    return PAI_ERR_NOMEM;
  }
  l->sock = s;
  *out = l;
  return PAI_OK;
}

pai_status_t
pai_proto_tcp_listener_port(const pai_proto_tcp_listener_t *l,
                            uint16_t *out_port) {
#ifdef _WIN32
  int alen = (int)sizeof(struct sockaddr_in);
#else
  socklen_t alen = (socklen_t)sizeof(struct sockaddr_in);
#endif
  struct sockaddr_in a;

  if (l == NULL || l->closed || out_port == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(&a, 0, sizeof(a));
  if (getsockname(l->sock, (struct sockaddr *)&a, &alen) != 0) {
    return PAI_ERR_IO;
  }
  *out_port = ntohs(a.sin_port);
  return PAI_OK;
}

pai_status_t
pai_proto_tcp_accept(pai_proto_tcp_listener_t *l, pai_proto_transport_t *out) {
  tcp_sock_t s;

  if (l == NULL || l->closed || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  s = accept(l->sock, NULL, NULL);
  if (s == TCP_SOCK_INVALID) {
    return PAI_ERR_IO;
  }
  return tcp_ep_wrap(s, out);
}

void
pai_proto_tcp_listener_destroy(pai_proto_tcp_listener_t *l) {
  if (l == NULL) {
    return;
  }
  if (!l->closed) {
    l->closed = 1;
    tcp_sock_close(l->sock);
    l->sock = TCP_SOCK_INVALID;
  }
  free(l);
}

void
pai_proto_tcp_transport_destroy(pai_proto_transport_t *t) {
  if (t == NULL || t->ctx == NULL) {
    return;
  }
  tcp_ep_close(t->ctx);
  free(t->ctx);
  t->ctx = NULL;
  t->send = NULL;
  t->recv = NULL;
  t->close = NULL;
}
