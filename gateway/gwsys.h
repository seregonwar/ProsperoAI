/*
 * ProsperoAI — gateway system layer (whitepaper §25/§26)
 *
 * Minimal portability wrapper: TCP sockets, detached threads
 * (thread-per-connection) and mutexes (per-model serialization).
 * Windows: winsock2 + _beginthreadex + CRITICAL_SECTION; POSIX: BSD
 * sockets + pthread. Host-side only (not in PS5 payload builds).
 */

#ifndef PAI_GATEWAY_GWSYS_H
#define PAI_GATEWAY_GWSYS_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Sockets                                                             */
/* ------------------------------------------------------------------ */

typedef intptr_t pai_gw_sock_t;
#define PAI_GW_SOCK_INVALID ((pai_gw_sock_t)-1)

/* One-time subsystem init (WSAStartup on Windows). Idempotent. */
pai_status_t pai_gw_sys_init(void);
void pai_gw_sys_shutdown(void);

/* Create a listening socket bound to `host` ("127.0.0.1", "0.0.0.0",
 * NULL = any) on `port`. Returns PAI_ERR_IO when bind/listen fails. */
pai_status_t pai_gw_sock_listen(pai_gw_sock_t *out, const char *host,
                                uint16_t port);

/* Accept one connection (blocking). */
pai_status_t pai_gw_sock_accept(pai_gw_sock_t listener, pai_gw_sock_t *out);

/* Dial host:port (blocking connect). */
pai_status_t pai_gw_sock_connect(pai_gw_sock_t *out, const char *host,
                                 uint16_t port);

/* Send all nbytes; PAI_ERR_IO on failure/closed. */
pai_status_t pai_gw_sock_send_all(pai_gw_sock_t s, const void *data,
                                  uint32_t nbytes);

/* Receive up to cap bytes; *out_read == 0 signals EOF. */
pai_status_t pai_gw_sock_recv(pai_gw_sock_t s, void *data, uint32_t cap,
                              uint32_t *out_read);

/*
 * Bound a blocking recv() with SO_RCVTIMEO (`ms`). After the timeout
 * fires without data, pai_gw_sock_recv returns PAI_ERR_IO — callers
 * treat that as "idle, close". 0 clears the timeout (infinite).
 */
pai_status_t pai_gw_sock_set_recv_timeout(pai_gw_sock_t s, uint32_t ms);

void pai_gw_sock_close(pai_gw_sock_t s);

/* ------------------------------------------------------------------ */
/* Threads                                                             */
/* ------------------------------------------------------------------ */

/* Start a detached thread running fn(arg). */
pai_status_t pai_gw_thread_create(void (*fn)(void *), void *arg);

/* ------------------------------------------------------------------ */
/* Mutexes                                                             */
/* ------------------------------------------------------------------ */

typedef struct pai_gw_mutex pai_gw_mutex_t; /* opaque                  */

pai_status_t pai_gw_mutex_init(pai_gw_mutex_t **out);
void pai_gw_mutex_destroy(pai_gw_mutex_t *m);
void pai_gw_mutex_lock(pai_gw_mutex_t *m);
void pai_gw_mutex_unlock(pai_gw_mutex_t *m);

#ifdef __cplusplus
}
#endif

#endif /* PAI_GATEWAY_GWSYS_H */
