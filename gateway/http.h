/*
 * ProsperoAI — minimal HTTP/1.1 (gateway §26)
 *
 * A small, dependency-free HTTP/1.1 server for the OpenAI-compatible
 * gateway: request parsing (method/target/headers/Content-Length body),
 * a streaming response path (Transfer-Encoding: chunked for SSE) and a
 * thread-per-connection TCP listener over the gwsys portability layer.
 *
 * Connection: close only (v0): every response closes the connection.
 * Request bodies are Content-Length only; chunked request encoding is
 * rejected (501). Headers are capped (PAI_HTTP_MAX_HEADER_BYTES) and
 * bodies at PAI_HTTP_MAX_BODY to bound memory.
 *
 * The response object is also usable in-process (unit tests): the
 * `send` callback can capture wire bytes into a buffer instead of a
 * socket, which lets the OpenAI layer be tested without networking.
 */

#ifndef PAI_GATEWAY_HTTP_H
#define PAI_GATEWAY_HTTP_H

#include "gwsys.h"

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PAI_HTTP_MAX_HEADER_BYTES (1u << 20) /* total request header cap */
#define PAI_HTTP_MAX_BODY (16u << 20)        /* Content-Length cap       */
#define PAI_HTTP_MAX_HEADERS 32u

typedef struct pai_http_req {
  char method[16];
  char target[1024];   /* raw request target (path + query)          */
  char path[1024];     /* target up to '?'                           */
  char query[512];     /* target after '?' (empty when none)         */
  char version[16];
  struct {
    char name[64];
    char value[511];
  } headers[PAI_HTTP_MAX_HEADERS];
  uint32_t num_headers;
  const uint8_t *body; /* points into the parsed buffer (not owned)  */
  uint32_t body_len;
} pai_http_req_t;

/*
 * Parse a complete HTTP/1.1 request from `data` (which must stay alive
 * while the request is used). Returns PAI_ERR_MISMATCH when more bytes
 * are needed (incomplete), PAI_ERR_PROTOCOL on malformed input and
 * PAI_ERR_NOMEM when a size cap is exceeded.
 */
pai_status_t pai_http_parse_request(const uint8_t *data, uint32_t len,
                                    pai_http_req_t *out);

/* Case-insensitive header lookup; NULL when absent. */
const char *pai_http_header(const pai_http_req_t *req, const char *name);

/* ------------------------------------------------------------------ */
/* Response                                                            */
/* ------------------------------------------------------------------ */

typedef struct pai_http_resp {
  int status;       /* HTTP status code (0 = not begun)               */
  int streaming;    /* 1 = chunked transfer (SSE)                     */
  int begun;
  void *ctx;        /* opaque, passed to send                         */
  int (*send)(void *ctx, const void *data, uint32_t nbytes); /* 1 = ok */
} pai_http_resp_t;

/* Initialize a response over the given wire sink. */
void pai_http_resp_init(pai_http_resp_t *resp, void *ctx,
                        int (*send)(void *, const void *, uint32_t));

/*
 * Begin the response: writes the status line + headers. When
 * `streaming` is set the body uses chunked transfer encoding and each
 * pai_http_resp_write call becomes one chunk (SSE events flush
 * naturally). content_type may be NULL (defaults to text/plain).
 */
pai_status_t pai_http_resp_begin(pai_http_resp_t *resp, int status,
                                 const char *content_type, int streaming);

/* Write body bytes (chunk-encoded when streaming). */
pai_status_t pai_http_resp_write(pai_http_resp_t *resp, const void *data,
                                 uint32_t nbytes);

/* Finish the response (terminal chunk when streaming). */
pai_status_t pai_http_resp_end(pai_http_resp_t *resp);

/* HTTP status phrase ("OK", "Not Found", ...). */
const char *pai_http_status_text(int status);

/* ------------------------------------------------------------------ */
/* Server                                                              */
/* ------------------------------------------------------------------ */

typedef struct pai_http_server {
  pai_gw_sock_t listener;
  void *user;
  pai_status_t (*handler)(void *user, const pai_http_req_t *req,
                          pai_http_resp_t *resp);
} pai_http_server_t;

/*
 * Bind and listen on host:port. The handler receives every request and
 * writes a response through the resp object; a non-OK return from the
 * handler produces a 500. Returns PAI_ERR_IO when the port is busy.
 */
pai_status_t pai_http_server_init(pai_http_server_t *server,
                                  const char *host, uint16_t port,
                                  pai_status_t (*handler)(
                                      void *, const pai_http_req_t *,
                                      pai_http_resp_t *),
                                  void *user);

/* Accept loop: blocks serving connections (thread-per-connection). */
pai_status_t pai_http_server_run(pai_http_server_t *server);

void pai_http_server_close(pai_http_server_t *server);

#ifdef __cplusplus
}
#endif

#endif /* PAI_GATEWAY_HTTP_H */
