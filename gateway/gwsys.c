/*
 * ProsperoAI — gateway system layer — implementation
 */

#include "gwsys.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <process.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

/* ------------------------------------------------------------------ */
/* Sockets                                                             */
/* ------------------------------------------------------------------ */

static int sys_ready = 0;

pai_status_t
pai_gw_sys_init(void) {
#ifdef _WIN32
  WSADATA wsa;
  if (sys_ready) {
    return PAI_OK;
  }
  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
    return PAI_ERR_IO;
  }
  sys_ready = 1;
#else
  sys_ready = 1;
#endif
  return PAI_OK;
}

void
pai_gw_sys_shutdown(void) {
#ifdef _WIN32
  if (sys_ready) {
    WSACleanup();
    sys_ready = 0;
  }
#else
  sys_ready = 0;
#endif
}

pai_status_t
pai_gw_sock_listen(pai_gw_sock_t *out, const char *host, uint16_t port) {
#ifdef _WIN32
  SOCKET s;
#else
  int s;
#endif
  struct sockaddr_in addr;
  int one = 1;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out = PAI_GW_SOCK_INVALID;

#ifdef _WIN32
  s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) {
    return PAI_ERR_IO;
  }
#else
  s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    return PAI_ERR_IO;
  }
#endif

  setsockopt((pai_gw_sock_t)s, SOL_SOCKET, SO_REUSEADDR, (const char *)&one,
             (int)sizeof(one));

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (host != NULL && host[0] != '\0' && strcmp(host, "0.0.0.0") != 0) {
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
      /* Try "localhost" -> loopback. */
      if (strcmp(host, "localhost") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      } else {
        pai_gw_sock_close((pai_gw_sock_t)s);
        return PAI_ERR_INVALID_ARG;
      }
    }
  } else {
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
  }

  if (bind((pai_gw_sock_t)s, (const struct sockaddr *)&addr,
           (int)sizeof(addr)) != 0 ||
      listen((pai_gw_sock_t)s, 16) != 0) {
    pai_gw_sock_close((pai_gw_sock_t)s);
    return PAI_ERR_IO;
  }

  *out = (pai_gw_sock_t)s;
  return PAI_OK;
}

pai_status_t
pai_gw_sock_accept(pai_gw_sock_t listener, pai_gw_sock_t *out) {
#ifdef _WIN32
  SOCKET c;
#else
  int c;
#endif

  if (out == NULL || listener == PAI_GW_SOCK_INVALID) {
    return PAI_ERR_INVALID_ARG;
  }
  *out = PAI_GW_SOCK_INVALID;

#ifdef _WIN32
  c = accept((SOCKET)listener, NULL, NULL);
  if (c == INVALID_SOCKET) {
    return PAI_ERR_IO;
  }
#else
  c = accept((int)listener, NULL, NULL);
  if (c < 0) {
    return PAI_ERR_IO;
  }
#endif
  *out = (pai_gw_sock_t)c;
  return PAI_OK;
}

pai_status_t
pai_gw_sock_connect(pai_gw_sock_t *out, const char *host, uint16_t port) {
#ifdef _WIN32
  SOCKET s;
#else
  int s;
#endif
  struct sockaddr_in addr;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out = PAI_GW_SOCK_INVALID;

#ifdef _WIN32
  s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == INVALID_SOCKET) {
    return PAI_ERR_IO;
  }
#else
  s = socket(AF_INET, SOCK_STREAM, 0);
  if (s < 0) {
    return PAI_ERR_IO;
  }
#endif

  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (host != NULL && host[0] != '\0' && strcmp(host, "0.0.0.0") != 0) {
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) {
      if (strcmp(host, "localhost") == 0) {
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      } else {
        pai_gw_sock_close((pai_gw_sock_t)s);
        return PAI_ERR_INVALID_ARG;
      }
    }
  } else {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  }

  if (connect((pai_gw_sock_t)s, (const struct sockaddr *)&addr,
              (int)sizeof(addr)) != 0) {
    pai_gw_sock_close((pai_gw_sock_t)s);
    return PAI_ERR_IO;
  }

  *out = (pai_gw_sock_t)s;
  return PAI_OK;
}

pai_status_t
pai_gw_sock_send_all(pai_gw_sock_t s, const void *data, uint32_t nbytes) {
  const char *p = (const char *)data;
  uint32_t sent = 0;

  while (sent < nbytes) {
#ifdef _WIN32
    int n = send((SOCKET)s, p + sent, (int)(nbytes - sent), 0);
    if (n <= 0) {
      return PAI_ERR_IO;
    }
#else
    ssize_t n = send((int)s, p + sent, (size_t)(nbytes - sent), 0);
    if (n <= 0) {
      return PAI_ERR_IO;
    }
#endif
    sent += (uint32_t)n;
  }
  return PAI_OK;
}

pai_status_t
pai_gw_sock_recv(pai_gw_sock_t s, void *data, uint32_t cap,
                 uint32_t *out_read) {
#ifdef _WIN32
  int n;
#else
  ssize_t n;
#endif

  if (out_read == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  *out_read = 0;
  if (s == PAI_GW_SOCK_INVALID || (data == NULL && cap != 0)) {
    return PAI_ERR_INVALID_ARG;
  }

#ifdef _WIN32
  n = recv((SOCKET)s, (char *)data, (int)cap, 0);
  if (n == SOCKET_ERROR) {
    return PAI_ERR_IO;
  }
#else
  n = recv((int)s, data, (size_t)cap, 0);
  if (n < 0) {
    return PAI_ERR_IO;
  }
#endif
  *out_read = (uint32_t)n; /* 0 = EOF */
  return PAI_OK;
}

pai_status_t
pai_gw_sock_set_recv_timeout(pai_gw_sock_t s, uint32_t ms) {
  if (s == PAI_GW_SOCK_INVALID) {
    return PAI_ERR_INVALID_ARG;
  }
#ifdef _WIN32
  {
    DWORD tv = ms;
    if (setsockopt((SOCKET)s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
                   sizeof(tv)) != 0) {
      return PAI_ERR_IO;
    }
  }
#else
  {
    struct timeval tv;
    tv.tv_sec = ms / 1000;
    tv.tv_usec = (ms % 1000) * 1000;
    if (setsockopt((int)s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
      return PAI_ERR_IO;
    }
  }
#endif
  return PAI_OK;
}

void
pai_gw_sock_close(pai_gw_sock_t s) {
  if (s == PAI_GW_SOCK_INVALID) {
    return;
  }
#ifdef _WIN32
  closesocket((SOCKET)s);
#else
  close((int)s);
#endif
}

/* ------------------------------------------------------------------ */
/* Threads                                                             */
/* ------------------------------------------------------------------ */

/* Adapter: the caller's entry is void(*)(void*); the OS wants a
 * different signature. The wrapper owns its argument copy. */
typedef struct gw_thread_arg {
  void (*fn)(void *);
  void *arg;
} gw_thread_arg_t;

#ifdef _WIN32
static unsigned __stdcall
gw_thread_entry(void *p) {
  gw_thread_arg_t *a = (gw_thread_arg_t *)p;
  a->fn(a->arg);
  free(a);
  return 0;
}
#else
static void *
gw_thread_entry(void *p) {
  gw_thread_arg_t *a = (gw_thread_arg_t *)p;
  a->fn(a->arg);
  free(a);
  return NULL;
}
#endif

pai_status_t
pai_gw_thread_create(void (*fn)(void *), void *arg) {
  gw_thread_arg_t *a;
#ifdef _WIN32
  uintptr_t h;
#else
  pthread_t tid;
  int rc;
#endif

  a = (gw_thread_arg_t *)malloc(sizeof(*a));
  if (a == NULL) {
    return PAI_ERR_NOMEM;
  }
  a->fn = fn;
  a->arg = arg;
#ifdef _WIN32
  h = _beginthreadex(NULL, 0, gw_thread_entry, a, 0, NULL);
  if (h == 0) {
    free(a);
    return PAI_ERR_IO;
  }
  CloseHandle((HANDLE)h); /* detached */
  return PAI_OK;
#else
  rc = pthread_create(&tid, NULL, gw_thread_entry, a);
  if (rc != 0) {
    free(a);
    return PAI_ERR_IO;
  }
  pthread_detach(tid);
  return PAI_OK;
#endif
}

/* ------------------------------------------------------------------ */
/* Mutexes                                                             */
/* ------------------------------------------------------------------ */

struct pai_gw_mutex {
#ifdef _WIN32
  CRITICAL_SECTION cs;
#else
  pthread_mutex_t m;
#endif
};

pai_status_t
pai_gw_mutex_init(pai_gw_mutex_t **out) {
  pai_gw_mutex_t *m;

  if (out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  m = (pai_gw_mutex_t *)malloc(sizeof(*m));
  if (m == NULL) {
    return PAI_ERR_NOMEM;
  }
#ifdef _WIN32
  InitializeCriticalSection(&m->cs);
#else
  pthread_mutex_init(&m->m, NULL);
#endif
  *out = m;
  return PAI_OK;
}

void
pai_gw_mutex_destroy(pai_gw_mutex_t *m) {
  if (m == NULL) {
    return;
  }
#ifdef _WIN32
  DeleteCriticalSection(&m->cs);
#else
  pthread_mutex_destroy(&m->m);
#endif
  free(m);
}

void
pai_gw_mutex_lock(pai_gw_mutex_t *m) {
  if (m == NULL) {
    return;
  }
#ifdef _WIN32
  EnterCriticalSection(&m->cs);
#else
  pthread_mutex_lock(&m->m);
#endif
}

void
pai_gw_mutex_unlock(pai_gw_mutex_t *m) {
  if (m == NULL) {
    return;
  }
#ifdef _WIN32
  LeaveCriticalSection(&m->cs);
#else
  pthread_mutex_unlock(&m->m);
#endif
}
