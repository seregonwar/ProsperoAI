/*
 * ProsperoAI — PS5 deployment lifecycle
 *
 * Automates consecutive deploys: the first thing a fresh payload does is
 * ask any previous instance listening on PAI_LIFECYCLE_PORT to terminate
 * itself (same pattern as MemDBG's --replace-existing), then it binds the
 * port and serves the same request for the next deploy.
 *
 * The stop path uses payload_exit(): the hijacked host process continues
 * from its hijack point instead of being killed, so the console shell
 * never notices the payload went away.
 *
 * Wire protocol (8-byte frames):
 *   client -> "PAISTOP" : terminate this instance
 *   server -> "PAIBYE"  : acknowledged, exiting
 */

#include "platform.h"

#include <pai/log.h>

#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#ifdef PAI_PS5

#include <ps5/payload.h>

#define PAI_LIFECYCLE_STOP_MAGIC "PAISTOP"
#define PAI_LIFECYCLE_BYE_MAGIC  "PAIBYE"

static int g_pai_lifecycle_fd = -1;

static void
pai_lifecycle_sleep_ms(long ms) {
  struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
  nanosleep(&ts, NULL);
}

static int
pai_lifecycle_socket(void) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;

  if (fd < 0) {
    return -1;
  }
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  return fd;
}

/* Binds the lifecycle port. Returns 0 on success, -1 on failure with
 * errno preserved (EADDRINUSE = a previous instance is alive). */
static int
pai_lifecycle_try_bind(uint16_t port, int *out_fd) {
  struct sockaddr_in sin;
  int fd;

  fd = pai_lifecycle_socket();
  if (fd < 0) {
    return -1;
  }

  memset(&sin, 0, sizeof(sin));
  sin.sin_family = AF_INET;
  sin.sin_port = htons(port);
  sin.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(fd, (struct sockaddr *)&sin, sizeof(sin)) != 0) {
    int e = errno;
    close(fd);
    errno = e;
    return -1;
  }

  *out_fd = fd;
  return 0;
}

static void *
pai_lifecycle_listener(void *arg) {
  uint16_t port = (uint16_t)(uintptr_t)arg;
  int fd = g_pai_lifecycle_fd;

  if (fd < 0 || listen(fd, 4) != 0) {
    PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                  "lifecycle: listen failed, stop probes disabled\n");
    return NULL;
  }

  PAI_LOG_INFO_(PAI_SUB_PLATFORM,
                "lifecycle: listening on port %u for stop probes\n", port);

  for (;;) {
    char frame[8];
    int client = accept(fd, NULL, NULL);
    ssize_t n;

    if (client < 0) {
      continue;
    }

    n = recv(client, frame, sizeof(frame), 0);
    if (n >= 7 && memcmp(frame, PAI_LIFECYCLE_STOP_MAGIC, 7) == 0) {
      PAI_LOG_INFO_(PAI_SUB_PLATFORM,
                    "lifecycle: stop probe received; terminating instance\n");
      send(client, PAI_LIFECYCLE_BYE_MAGIC, 6, 0);
      close(client);
      payload_exit(0);
      /* not reached */
    }
    close(client);
  }
}

pai_status_t
pai_lifecycle_stop_previous(uint16_t port) {
  struct sockaddr_in dst;
  int fd;
  int r;

  fd = pai_lifecycle_socket();
  if (fd < 0) {
    return PAI_OK; /* no socket support: nothing to stop */
  }

  if (pai_lifecycle_try_bind(port, &r) == 0) {
    close(r);
    return PAI_OK; /* port free: no previous instance */
  }

  if (errno != EADDRINUSE) {
    close(fd);
    return PAI_OK; /* cannot tell; proceed anyway */
  }

  /* A previous instance holds the port: ask it to stop. */
  memset(&dst, 0, sizeof(dst));
  dst.sin_family = AF_INET;
  dst.sin_port = htons(port);
  dst.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

  if (connect(fd, (struct sockaddr *)&dst, sizeof(dst)) == 0) {
    char ack[7] = {0};
    PAI_LOG_INFO_(PAI_SUB_PLATFORM,
                  "lifecycle: previous instance found, sending stop probe\n");
    send(fd, PAI_LIFECYCLE_STOP_MAGIC, 7, 0);
    recv(fd, ack, sizeof(ack), 0);
    PAI_LOG_INFO_(PAI_SUB_PLATFORM, "lifecycle: previous instance: %s\n", ack);
  } else {
    PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                  "lifecycle: port busy but probe failed; continuing\n");
  }
  close(fd);

  /* Wait for the previous instance to release the port. */
  for (int i = 0; i < 30; i++) {
    int probe_fd;
    pai_lifecycle_sleep_ms(100);
    if (pai_lifecycle_try_bind(port, &probe_fd) == 0) {
      close(probe_fd);
      PAI_LOG_INFO_(PAI_SUB_PLATFORM,
                    "lifecycle: previous instance terminated\n");
      return PAI_OK;
    }
    if (errno != EADDRINUSE) {
      break;
    }
  }

  PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                "lifecycle: previous instance did not terminate in time\n");
  return PAI_ERR_TIMEOUT;
}

pai_status_t
pai_lifecycle_start(uint16_t port) {
  pthread_t thread;

  if (pai_lifecycle_try_bind(port, &g_pai_lifecycle_fd) != 0) {
    if (errno == EADDRINUSE) {
      PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                    "lifecycle: port %u already bound; stop probes disabled\n",
                    port);
    }
    return PAI_ERR_IO;
  }

  if (pthread_create(&thread, NULL, pai_lifecycle_listener,
                     (void *)(uintptr_t)port) != 0) {
    PAI_LOG_WARN_(PAI_SUB_PLATFORM,
                  "lifecycle: listener thread creation failed\n");
  }

  return PAI_OK;
}

#endif /* PAI_PS5 */
