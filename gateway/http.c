/*
 * ProsperoAI — minimal HTTP/1.1 (gateway §26) — implementation
 */

#include "http.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Status phrases                                                      */
/* ------------------------------------------------------------------ */

const char *
pai_http_status_text(int status) {
  switch (status) {
  case 200: return "OK";
  case 400: return "Bad Request";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 411: return "Length Required";
  case 413: return "Payload Too Large";
  case 500: return "Internal Server Error";
  case 501: return "Not Implemented";
  default:  return "Unknown";
  }
}

/* ------------------------------------------------------------------ */
/* Request parsing                                                     */
/* ------------------------------------------------------------------ */

static int
ascii_ieq(const char *a, const char *b) {
  while (*a != '\0' && *b != '\0') {
    char ca = *a;
    char cb = *b;
    if (ca >= 'A' && ca <= 'Z') {
      ca = (char)(ca - 'A' + 'a');
    }
    if (cb >= 'A' && cb <= 'Z') {
      cb = (char)(cb - 'A' + 'a');
    }
    if (ca != cb) {
      return 0;
    }
    a++;
    b++;
  }
  return *a == '\0' && *b == '\0';
}

const char *
pai_http_header(const pai_http_req_t *req, const char *name) {
  uint32_t i;
  for (i = 0; i < req->num_headers; i++) {
    if (ascii_ieq(req->headers[i].name, name)) {
      return req->headers[i].value;
    }
  }
  return NULL;
}

/* Advance to the next header line; returns 0 at the blank line. */
static const char *
next_line(const uint8_t *data, uint32_t len, uint32_t *pos, char *out,
          uint32_t out_cap) {
  uint32_t i = *pos;
  uint32_t n = 0;

  while (i < len && n + 1 < out_cap) {
    if (data[i] == '\r' && i + 1 < len && data[i + 1] == '\n') {
      out[n] = '\0';
      *pos = i + 2;
      return out;
    }
    if (data[i] == '\n') { /* tolerate bare LF */
      out[n] = '\0';
      *pos = i + 1;
      return out;
    }
    out[n++] = (char)data[i++];
  }
  return NULL; /* line too long or truncated */
}

static int
parse_u64(const char *s, uint64_t *out) {
  uint64_t v = 0;
  if (s == NULL || *s == '\0') {
    return -1;
  }
  for (; *s != '\0'; s++) {
    if (*s < '0' || *s > '9') {
      return -1;
    }
    v = v * 10 + (uint64_t)(*s - '0');
    if (v > (uint64_t)PAI_HTTP_MAX_BODY) {
      return -1;
    }
  }
  *out = v;
  return 0;
}

pai_status_t
pai_http_parse_request(const uint8_t *data, uint32_t len,
                       pai_http_req_t *out) {
  uint32_t pos = 0;
  char line[1024];
  char *sp;
  char *target;
  char *qmark;
  const char *cl;
  uint64_t body_len;
  uint32_t nheaders = 0;

  if (data == NULL || out == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(out, 0, sizeof(*out));

  if (next_line(data, len, &pos, line, sizeof(line)) == NULL) {
    return len >= PAI_HTTP_MAX_HEADER_BYTES ? PAI_ERR_NOMEM
                                            : PAI_ERR_MISMATCH;
  }
  /* "METHOD SP TARGET SP HTTP/1.x" */
  sp = strchr(line, ' ');
  if (sp == NULL) {
    return PAI_ERR_PROTOCOL;
  }
  *sp = '\0';
  if (sp - line == 0 || (uint32_t)(sp - line) >= sizeof(out->method)) {
    return PAI_ERR_PROTOCOL;
  }
  memcpy(out->method, line, (size_t)(sp - line));
  target = sp + 1;
  sp = strchr(target, ' ');
  if (sp == NULL) {
    return PAI_ERR_PROTOCOL;
  }
  *sp = '\0';
  if (target[0] == '\0' || (uint32_t)strlen(target) >= sizeof(out->target)) {
    return PAI_ERR_PROTOCOL;
  }
  strcpy(out->target, target);
  sp++;
  if ((uint32_t)strlen(sp) >= sizeof(out->version)) {
    return PAI_ERR_PROTOCOL;
  }
  strcpy(out->version, sp);
  if (strncmp(out->version, "HTTP/1.", 7) != 0) {
    return PAI_ERR_PROTOCOL;
  }

  /* Split target into path + query. */
  qmark = strchr(out->target, '?');
  if (qmark != NULL) {
    *qmark = '\0';
    if ((uint32_t)strlen(qmark + 1) >= sizeof(out->query)) {
      return PAI_ERR_PROTOCOL;
    }
    strcpy(out->query, qmark + 1);
  }
  if (out->target[0] == '\0' || strlen(out->target) >= sizeof(out->path)) {
    return PAI_ERR_PROTOCOL;
  }
  strcpy(out->path, out->target);

  /* Header lines. */
  for (;;) {
    char *colon;
    if (next_line(data, len, &pos, line, sizeof(line)) == NULL) {
      return len >= PAI_HTTP_MAX_HEADER_BYTES ? PAI_ERR_NOMEM
                                              : PAI_ERR_MISMATCH;
    }
    if (line[0] == '\0') {
      break; /* end of headers */
    }
    if (nheaders >= PAI_HTTP_MAX_HEADERS) {
      return PAI_ERR_NOMEM;
    }
    colon = strchr(line, ':');
    if (colon == NULL) {
      return PAI_ERR_PROTOCOL;
    }
    *colon = '\0';
    if (colon == line || (uint32_t)(colon - line) >=
                             sizeof(out->headers[nheaders].name)) {
      return PAI_ERR_PROTOCOL;
    }
    memcpy(out->headers[nheaders].name, line, (size_t)(colon - line));
    out->headers[nheaders].name[colon - line] = '\0';
    {
      const char *v = colon + 1;
      while (*v == ' ' || *v == '\t') {
        v++;
      }
      if ((uint32_t)strlen(v) >= sizeof(out->headers[nheaders].value)) {
        return PAI_ERR_PROTOCOL;
      }
      strcpy(out->headers[nheaders].value, v);
    }
    nheaders++;
  }
  out->num_headers = nheaders;

  /* Content-Length body (chunked requests are rejected: v0). */
  cl = pai_http_header(out, "Content-Length");
  if (cl != NULL) {
    if (parse_u64(cl, &body_len) != 0) {
      return PAI_ERR_PROTOCOL;
    }
    if (body_len > (uint64_t)PAI_HTTP_MAX_BODY) {
      return PAI_ERR_NOMEM;
    }
    if (pos + (uint32_t)body_len > len) {
      return PAI_ERR_MISMATCH; /* need more body bytes */
    }
    out->body = data + pos;
    out->body_len = (uint32_t)body_len;
  } else if (pai_http_header(out, "Transfer-Encoding") != NULL) {
    return PAI_ERR_UNSUPPORTED;
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Response                                                            */
/* ------------------------------------------------------------------ */

void
pai_http_resp_init(pai_http_resp_t *resp, void *ctx,
                   int (*send)(void *, const void *, uint32_t)) {
  resp->status = 0;
  resp->streaming = 0;
  resp->begun = 0;
  resp->ctx = ctx;
  resp->send = send;
}

static int
sock_send(void *ctx, const void *data, uint32_t nbytes) {
  return pai_gw_sock_send_all((pai_gw_sock_t)(intptr_t)ctx, data, nbytes) ==
         PAI_OK;
}

pai_status_t
pai_http_resp_begin(pai_http_resp_t *resp, int status,
                    const char *content_type, int streaming) {
  char hdr[512];
  int n;

  if (resp == NULL || resp->send == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  resp->status = status;
  resp->streaming = streaming != 0;
  resp->begun = 1;

  n = snprintf(hdr, sizeof(hdr),
               "HTTP/1.1 %d %s\r\n"
               "Content-Type: %s\r\n"
               "%s"
               "Connection: close\r\n"
               "Server: ProsperoAI\r\n"
               "\r\n",
               status, pai_http_status_text(status),
               content_type != NULL ? content_type : "text/plain",
               streaming ? "Transfer-Encoding: chunked\r\n" : "");
  if (n <= 0) {
    return PAI_ERR_INVALID_ARG;
  }
  return resp->send(resp->ctx, hdr, (uint32_t)n) ? PAI_OK : PAI_ERR_IO;
}

pai_status_t
pai_http_resp_write(pai_http_resp_t *resp, const void *data, uint32_t nbytes) {
  char chunk_hdr[16];
  int n;

  if (resp == NULL || resp->send == NULL || !resp->begun) {
    return PAI_ERR_INVALID_ARG;
  }
  if (nbytes == 0) {
    return PAI_OK;
  }
  if (resp->streaming) {
    n = snprintf(chunk_hdr, sizeof(chunk_hdr), "%x\r\n", nbytes);
    if (n <= 0 ||
        !resp->send(resp->ctx, chunk_hdr, (uint32_t)n) ||
        !resp->send(resp->ctx, data, nbytes) ||
        !resp->send(resp->ctx, "\r\n", 2)) {
      return PAI_ERR_IO;
    }
    return PAI_OK;
  }
  return resp->send(resp->ctx, data, nbytes) ? PAI_OK : PAI_ERR_IO;
}

pai_status_t
pai_http_resp_end(pai_http_resp_t *resp) {
  if (resp == NULL || resp->send == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (resp->streaming) {
    if (!resp->send(resp->ctx, "0\r\n\r\n", 5)) {
      return PAI_ERR_IO;
    }
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Server                                                              */
/* ------------------------------------------------------------------ */

typedef struct gw_conn {
  pai_gw_sock_t sock;
  pai_http_server_t *server;
} gw_conn_t;

static void
gw_conn_thread(void *p) {
  gw_conn_t *c = (gw_conn_t *)p;
  pai_http_server_t *server = c->server;
  uint8_t *buf = NULL;
  uint32_t len = 0;
  uint32_t cap = 0;
  pai_http_req_t req;
  pai_http_resp_t resp;
  pai_status_t st;
  int parsed = 0;

  /* Read until the request parses (or fails hard). */
  while (!parsed) {
    uint8_t tmp[4096];
    uint32_t got = 0;
    st = pai_gw_sock_recv(c->sock, tmp, sizeof(tmp), &got);
    if (st != PAI_OK || got == 0) {
      break; /* EOF / error: nothing to answer */
    }
    if (len + got > cap) {
      uint32_t ncap = cap == 0 ? 8192 : cap * 2;
      uint8_t *nb;
      while (ncap < len + got) {
        ncap *= 2;
      }
      nb = (uint8_t *)realloc(buf, ncap);
      if (nb == NULL) {
        break; /* cannot buffer: drop */
      }
      buf = nb;
      cap = ncap;
    }
    memcpy(buf + len, tmp, got);
    len += got;

    st = pai_http_parse_request(buf, len, &req);
    if (st == PAI_OK) {
      parsed = 1;
      break;
    }
    if (st == PAI_ERR_MISMATCH) {
      if (len >= PAI_HTTP_MAX_HEADER_BYTES + PAI_HTTP_MAX_BODY) {
        pai_http_resp_init(&resp, (void *)(intptr_t)c->sock, sock_send);
        pai_http_resp_begin(&resp, 413, "text/plain", 0);
        pai_http_resp_write(&resp, "request too large\n", 18);
        pai_http_resp_end(&resp);
        break;
      }
      continue;
    }
    /* Malformed / oversized / unsupported encoding. */
    pai_http_resp_init(&resp, (void *)(intptr_t)c->sock, sock_send);
    if (st == PAI_ERR_UNSUPPORTED) {
      pai_http_resp_begin(&resp, 501, "text/plain", 0);
    } else if (st == PAI_ERR_NOMEM) {
      pai_http_resp_begin(&resp, 413, "text/plain", 0);
    } else {
      pai_http_resp_begin(&resp, 400, "text/plain", 0);
    }
    pai_http_resp_write(&resp, "bad request\n", 12);
    pai_http_resp_end(&resp);
    break;
  }

  if (parsed) {
    pai_http_resp_init(&resp, (void *)(intptr_t)c->sock, sock_send);
    st = server->handler(server->user, &req, &resp);
    if (st != PAI_OK && !resp.begun) {
      pai_http_resp_begin(&resp, 500, "application/json", 0);
      pai_http_resp_write(&resp, "{\"error\":{\"message\":\"internal "
                                 "error\",\"type\":\"server_error\"}}",
                          57);
    }
    pai_http_resp_end(&resp);
  }

  pai_gw_sock_close(c->sock);
  free(buf);
  free(c);
}

pai_status_t
pai_http_server_init(pai_http_server_t *server, const char *host,
                     uint16_t port,
                     pai_status_t (*handler)(void *, const pai_http_req_t *,
                                             pai_http_resp_t *),
                     void *user) {
  pai_status_t st;

  if (server == NULL || handler == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(server, 0, sizeof(*server));
  server->user = user;
  server->handler = handler;
  server->listener = PAI_GW_SOCK_INVALID;

  st = pai_gw_sys_init();
  if (st != PAI_OK) {
    return st;
  }
  return pai_gw_sock_listen(&server->listener, host, port);
}

pai_status_t
pai_http_server_run(pai_http_server_t *server) {
  if (server == NULL || server->listener == PAI_GW_SOCK_INVALID) {
    return PAI_ERR_INVALID_ARG;
  }
  for (;;) {
    pai_gw_sock_t c;
    gw_conn_t *conn;
    pai_status_t st;

    st = pai_gw_sock_accept(server->listener, &c);
    if (st != PAI_OK) {
      return st;
    }
    conn = (gw_conn_t *)malloc(sizeof(*conn));
    if (conn == NULL) {
      pai_gw_sock_close(c);
      continue;
    }
    conn->sock = c;
    conn->server = server;
    if (pai_gw_thread_create(gw_conn_thread, conn) != PAI_OK) {
      pai_gw_sock_close(c);
      free(conn);
    }
  }
}

void
pai_http_server_close(pai_http_server_t *server) {
  if (server == NULL) {
    return;
  }
  if (server->listener != PAI_GW_SOCK_INVALID) {
    pai_gw_sock_close(server->listener);
    server->listener = PAI_GW_SOCK_INVALID;
  }
}
