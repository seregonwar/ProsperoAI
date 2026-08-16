#include "test.h"

#include <gateway/gateway.h>
#include <gateway/remote.h>

#include <importer.h>
#include <model.h>
#include <protocol/protocol.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define TEST_SLEEP_MS(ms) Sleep(ms)
#else
#include <sys/stat.h>
#include <unistd.h>
#define TEST_SLEEP_MS(ms) usleep((ms) * 1000)
#endif

static void
make_dir(const char *path) {
#ifdef _WIN32
  _mkdir(path);
#else
  mkdir(path, 0755);
#endif
}

/* ---- tiny deterministic model fixture (mirrors test_importer) ---- */

#define V 4
#define H 8

static const char k_desc[] =
    "name tiny-lm\n"
    "family llm\n"
    "context 16\n"
    "layers 1\n"
    "kv_bytes 8\n"
    "vocab 4\n"
    "\n"
    "token 0 a\n"
    "token 1 b\n"
    "token 2 c\n"
    "token 3 d\n"
    "\n"
    "value 1 f32 [4] input\n"
    "value 2 f32 [8,4] param weights=tok_embed.bin\n"
    "value 3 f32 [8] activation\n"
    "value 4 f32 [4,8] param weights=w_out.bin\n"
    "value 5 f32 [4] activation\n"
    "value 6 f32 [4] output\n"
    "\n"
    "op 1 gemv 2, 1 -> 3\n"
    "op 2 gemv 4, 3 -> 5\n"
    "op 3 softmax 5 -> 6\n";

static int
write_file(const char *path, const void *data, size_t nbytes) {
  FILE *f = fopen(path, "wb");
  if (f == NULL) {
    return -1;
  }
  if (fwrite(data, 1, nbytes, f) != nbytes) {
    fclose(f);
    return -1;
  }
  fclose(f);
  return 0;
}

static int
build_fixture(void) {
  float w_embed[H * V];
  float w_out[V * H];
  pai_import_options_t opts;
  FILE *f;

  memset(w_embed, 0, sizeof(w_embed));
  for (uint32_t i = 0; i < V && i < H; i++) {
    w_embed[i * V + i] = 2.0f;
  }
  memset(w_out, 0, sizeof(w_out));
  for (uint32_t i = 0; i < V; i++) {
    w_out[((i + 1) % V) * H + i] = 100.0f;
  }

  CHECK(write_file("gw_desc.txt", k_desc, sizeof(k_desc) - 1) == 0);
  f = fopen("tok_embed.bin", "wb");
  CHECK(f != NULL);
  fwrite(w_embed, sizeof(float), H * V, f);
  fclose(f);
  f = fopen("w_out.bin", "wb");
  CHECK(f != NULL);
  fwrite(w_out, sizeof(float), V * H, f);
  fclose(f);

  memset(&opts, 0, sizeof(opts));
  opts.quant = NULL;
  return pai_import_model_to_file("gw_desc.txt", "gw_tiny.pai", &opts) ==
         PAI_OK;
}

/* ---- in-process request capture ---- */

typedef struct cap {
  char buf[1 << 16];
  uint32_t len;
} cap_t;

static int
test_send(void *ctx, const void *data, uint32_t n) {
  cap_t *c = (cap_t *)ctx;
  if (c->len + n <= sizeof(c->buf)) {
    memcpy(c->buf + c->len, data, n);
    c->len += n;
    return 1;
  }
  return 0;
}

static pai_status_t
gw_call(pai_gw_t *gw, const char *method, const char *target,
        const char *body, cap_t *cap) {
  pai_http_resp_t resp;
  cap->len = 0;
  pai_http_resp_init(&resp, cap, test_send);
  return pai_gw_handle_request(gw, method, target,
                               (const uint8_t *)body,
                               body != NULL ? (uint32_t)strlen(body) : 0,
                               &resp);
}

static int
resp_status(const cap_t *cap) {
  if (cap->len < 12 || strncmp(cap->buf, "HTTP/1.1 ", 9) != 0) {
    return -1;
  }
  return atoi(cap->buf + 9);
}

/* Body start offset (after \r\n\r\n); -1 when absent. */
static int
resp_body(const cap_t *cap, const char **out_body, uint32_t *out_len) {
  int i;
  for (i = 3; i + 3 < (int)cap->len; i++) {
    if (cap->buf[i] == '\r' && cap->buf[i + 1] == '\n' &&
        cap->buf[i + 2] == '\r' && cap->buf[i + 3] == '\n') {
      *out_body = cap->buf + i + 4;
      *out_len = cap->len - (uint32_t)(i + 4);
      return 0;
    }
  }
  return -1;
}

static int
count_substr(const cap_t *cap, const char *needle) {
  int n = 0;
  uint32_t nl = (uint32_t)strlen(needle);
  uint32_t i;
  for (i = 0; i + nl <= cap->len; i++) {
    if (memcmp(cap->buf + i, needle, nl) == 0) {
      n++;
    }
  }
  return n;
}

static int
count_substr_in(const char *buf, uint32_t len, const char *needle) {
  int n = 0;
  uint32_t nl = (uint32_t)strlen(needle);
  uint32_t i;
  for (i = 0; i + nl <= len; i++) {
    if (memcmp(buf + i, needle, nl) == 0) {
      n++;
    }
  }
  return n;
}

/* Reassemble the chunked-transfer body of a captured streaming
 * response (SSE frames are split across chunk boundaries by design). */
static uint32_t
dechunk_body(const cap_t *cap, char *out, uint32_t out_cap) {
  const char *body;
  uint32_t blen;
  const char *p;
  const char *end;
  uint32_t olen = 0;

  if (resp_body(cap, &body, &blen) != 0) {
    return 0;
  }
  p = body;
  end = body + blen;
  while (p < end && olen < out_cap) {
    char *nx;
    unsigned long sz = strtoul(p, &nx, 16);
    if (nx == p) {
      break;
    }
    if (*nx == '\r' && nx + 1 < end && nx[1] == '\n') {
      nx += 2;
    } else if (*nx == '\n') {
      nx += 1;
    }
    if (sz == 0) {
      break; /* terminal chunk */
    }
    if (nx + sz > end) {
      break;
    }
    if (olen + (uint32_t)sz > out_cap) {
      sz = out_cap - olen;
    }
    memcpy(out + olen, nx, sz);
    olen += (uint32_t)sz;
    p = nx + sz;
    if (p < end && p[0] == '\r' && p + 1 < end && p[1] == '\n') {
      p += 2;
    } else if (p < end && p[0] == '\n') {
      p += 1;
    }
  }
  return olen;
}

/* Parse the JSON body of a non-streamed response. */
static int
parse_resp_json(const cap_t *cap, pai_json_doc_t *doc) {
  const char *body;
  uint32_t blen;
  if (resp_body(cap, &body, &blen) != 0) {
    return -1;
  }
  return pai_json_parse(doc, body, blen) == PAI_OK ? 0 : -1;
}

/* ---- HTTP idle-timeout helpers (slowloris guard) ---- */

static pai_status_t
idle_handler(void *user, const pai_http_req_t *req, pai_http_resp_t *resp) {
  /* Never reached: the client sends nothing and is disconnected first. */
  (void)user;
  (void)req;
  return pai_http_resp_begin(resp, 200, "text/plain", 0);
}

static void
server_run_thread(void *p) {
  /* Returns when the listener is closed from another thread. */
  (void)pai_http_server_run((pai_http_server_t *)p);
}

/* ---- remote payload probe (Prospero protocol server thread) ---- */

typedef struct remote_probe {
  pai_proto_tcp_listener_t *lst;
  volatile int stop;
  volatile int finished;
  volatile int refuse;     /* 1 = HELLO_NACK on negotiation            */
  volatile int no_answer;  /* 1 = accept GENERATE but never reply       */
  volatile int n_served;
} remote_probe_t;

typedef struct probe_gen {
  pai_proto_conn_t *conn;
  remote_probe_t *probe;
} probe_gen_t;

static uint32_t
probe_hello(void *user, uint32_t remote_caps) {
  probe_gen_t *g = (probe_gen_t *)user;
  (void)remote_caps;
  return g->probe->refuse ? 0 : PAI_PROTO_CAP_KNOWN;
}

/* Echo the prompt back as 2-char TOKEN chunks (§24 example). */
static pai_status_t
probe_on_message(void *user, const pai_proto_frame_t *frame) {
  probe_gen_t *g = (probe_gen_t *)user;
  const char *prompt;
  uint32_t plen;
  uint32_t off = 0;
  uint64_t sid = 0;

  if (frame->msg_type != PAI_PROTO_MSG_GENERATE) {
    return PAI_OK;
  }
  CHECK(pai_proto_msg_decode_generate(frame->payload, frame->payload_len,
                                      &prompt, &plen) == PAI_OK);
  if (g->probe->no_answer) {
    return PAI_OK; /* negotiates, then goes silent */
  }
  CHECK(pai_proto_conn_session_open(g->conn, &sid) == PAI_OK);
  CHECK(pai_proto_conn_send_raw(g->conn, PAI_PROTO_MSG_ACCEPTED,
                                PAI_PROTO_FLAG_REPLY, frame->request_id, sid,
                                NULL, 0) == PAI_OK);
  while (off < plen) {
    uint32_t n = plen - off;
    uint32_t flags = 0;
    uint8_t pay[128];
    uint32_t tlen = 0;
    if (n > 2) {
      n = 2;
    }
    if (off == 0) {
      flags |= PAI_PROTO_FLAG_STREAM_START;
    }
    if (off + n >= plen) {
      flags |= PAI_PROTO_FLAG_STREAM_END;
    }
    CHECK(pai_proto_msg_encode_token(pay, sizeof(pay),
                                     (const uint8_t *)prompt + off, n,
                                     &tlen) == PAI_OK);
    CHECK(pai_proto_conn_send_raw(g->conn, PAI_PROTO_MSG_TOKEN, flags,
                                  frame->request_id, sid, pay, tlen) ==
          PAI_OK);
    off += n;
  }
  CHECK(pai_proto_conn_send_raw(g->conn, PAI_PROTO_MSG_COMPLETE, 0,
                                frame->request_id, sid, NULL, 0) == PAI_OK);
  return PAI_OK;
}

/* Accept connections and pump a server conn until closed. */
static void
remote_probe_thread(void *arg) {
  remote_probe_t *p = (remote_probe_t *)arg;

  while (!p->stop) {
    pai_proto_transport_t t;
    pai_proto_conn_t conn;
    pai_proto_callbacks_t cb;
    probe_gen_t gen;
    uint32_t frames = 0;

    if (pai_proto_tcp_accept(p->lst, &t) != PAI_OK) {
      break; /* listener closed */
    }
    memset(&gen, 0, sizeof(gen));
    gen.conn = &conn;
    gen.probe = p;
    memset(&cb, 0, sizeof(cb));
    cb.on_hello = probe_hello;
    cb.on_message = probe_on_message;
    if (pai_proto_conn_init(&conn, PAI_PROTO_ROLE_SERVER, PAI_PROTO_CAP_KNOWN,
                            &t, &cb, &gen) != PAI_OK) {
      pai_proto_tcp_transport_destroy(&t);
      break;
    }
    while (!p->stop &&
           pai_proto_conn_state(&conn) != PAI_PROTO_STATE_CLOSED) {
      (void)pai_proto_conn_poll(&conn, &frames);
    }
    if (p->stop) {
      pai_proto_conn_destroy(&conn);
      pai_proto_tcp_transport_destroy(&t);
      break;
    }
    p->n_served++;
    pai_proto_conn_destroy(&conn);
    pai_proto_tcp_transport_destroy(&t);
  }
  p->finished = 1;
}

/* Start a probe on an ephemeral port; *out_port receives it. */
static int
probe_start(remote_probe_t *p, pai_proto_tcp_listener_t **out_lst,
            uint16_t *out_port) {
  pai_proto_tcp_listener_t *lst = NULL;
  uint16_t port = 0;
  if (pai_proto_tcp_listen(&lst, "127.0.0.1", 0) != PAI_OK) {
    return -1;
  }
  if (pai_proto_tcp_listener_port(lst, &port) != PAI_OK) {
    pai_proto_tcp_listener_destroy(lst);
    return -1;
  }
  memset(p, 0, sizeof(*p));
  p->lst = lst;
  if (pai_gw_thread_create(remote_probe_thread, p) != PAI_OK) {
    pai_proto_tcp_listener_destroy(lst);
    return -1;
  }
  *out_lst = lst;
  *out_port = port;
  return 0;
}

static void
probe_stop(remote_probe_t *p, pai_proto_tcp_listener_t *lst) {
  int waited = 0;
  p->stop = 1;
  pai_proto_tcp_listener_destroy(lst); /* unblock accept */
  while (!p->finished && waited < 200) {
    TEST_SLEEP_MS(10);
    waited++;
  }
}

TEST_MAIN_BEGIN()

/* Build the fixture model. */
CHECK(build_fixture());

/* ---- raw HTTP request parsing ---- */
{
  pai_http_req_t req;
  static const char r1[] = "GET /v1/models?x=1 HTTP/1.1\r\n"
                           "Host: localhost\r\n"
                           "User-Agent: test\r\n\r\n";
  static const char r2[] =
      "POST /v1/completions HTTP/1.1\r\n"
      "Content-Length: 2\r\n"
      "\r\n"
      "{}";
  CHECK(pai_http_parse_request((const uint8_t *)r1, sizeof(r1) - 1, &req) ==
        PAI_OK);
  CHECK(strcmp(req.method, "GET") == 0);
  CHECK(strcmp(req.path, "/v1/models") == 0);
  CHECK(strcmp(req.query, "x=1") == 0);
  CHECK_EQ_UINT(req.num_headers, 2);
  CHECK(strcmp(pai_http_header(&req, "host"), "localhost") == 0);
  CHECK(pai_http_header(&req, "Missing") == NULL);
  CHECK_EQ_UINT(req.body_len, 0);

  CHECK(pai_http_parse_request((const uint8_t *)r2, sizeof(r2) - 1, &req) ==
        PAI_OK);
  CHECK(strcmp(req.method, "POST") == 0);
  CHECK_EQ_UINT(req.body_len, 2);
  CHECK(memcmp(req.body, "{}", 2) == 0);

  /* incomplete / malformed */
  CHECK(pai_http_parse_request((const uint8_t *)r1, 12, &req) ==
        PAI_ERR_MISMATCH);
  CHECK(pai_http_parse_request((const uint8_t *)"BOGUS\r\n\r\n",
                               sizeof("BOGUS\r\n\r\n") - 1, &req) ==
        PAI_ERR_PROTOCOL);
  /* declared length beyond buffer */
  CHECK(pai_http_parse_request(
            (const uint8_t *)"POST / HTTP/1.1\r\nContent-Length: 99\r\n\r\nab",
            39, &req) == PAI_ERR_MISMATCH);
}

/* ---- gateway registry ---- */
{
  pai_gw_t gw;
  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);
  CHECK(pai_gw_find(&gw, "gw_tiny") != NULL);
  CHECK(pai_gw_find(&gw, "nope") == NULL);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_ERR_MISMATCH);
  CHECK(pai_gw_add_file(&gw, "missing.pai") == PAI_OK); /* lazy open */
  pai_gw_destroy(&gw);
}

/* ---- GET endpoints ---- */
{
  pai_gw_t gw;
  cap_t cap;
  pai_json_doc_t doc;
  int32_t root, data;

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);

  CHECK(gw_call(&gw, "GET", "/healthz", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  {
    const char *b;
    uint32_t bl;
    CHECK(resp_body(&cap, &b, &bl) == 0);
    CHECK_EQ_UINT(bl, 3);
    CHECK(memcmp(b, "ok\n", 3) == 0);
  }

  CHECK(gw_call(&gw, "GET", "/v1/models", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  data = pai_json_member(&doc, root, "data");
  CHECK(pai_json_type(&doc, data) == PAI_JSON_ARRAY);
  CHECK_EQ_INT(pai_json_array_len(&doc, data), 1);
  {
    int32_t m0 = pai_json_array_at(&doc, data, 0);
    CHECK(strcmp(pai_json_str_member(&doc, m0, "id"), "gw_tiny") == 0);
    CHECK(strcmp(pai_json_str_member(&doc, m0, "object"), "model") == 0);
  }
  pai_json_destroy(&doc);

  CHECK(gw_call(&gw, "GET", "/v1/models/gw_tiny", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(gw_call(&gw, "GET", "/v1/models/nope", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 404);
  CHECK(gw_call(&gw, "GET", "/v1/unknown", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 404);

  /* unsupported method */
  CHECK(gw_call(&gw, "DELETE", "/v1/models", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 405);

  pai_gw_destroy(&gw);
}

/* ---- text completions ---- */
{
  pai_gw_t gw;
  cap_t cap;
  pai_json_doc_t doc;
  int32_t root, choices, c0;

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);

  /* missing body / malformed / unknown model */
  CHECK(gw_call(&gw, "POST", "/v1/completions", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);
  CHECK(gw_call(&gw, "POST", "/v1/completions", "{oops", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"nope\",\"prompt\":\"a\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 404);
  CHECK(gw_call(&gw, "POST", "/v1/completions", "{}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400); /* missing prompt */

  /* deterministic greedy chain: a -> bcdabcda */
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"gw_tiny\",\"prompt\":\"a\",\"temperature\":0,"
                "\"max_tokens\":8}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  CHECK(strcmp(pai_json_str_member(&doc, root, "object"), "text_completion") ==
        0);
  CHECK(strcmp(pai_json_str_member(&doc, root, "model"), "gw_tiny") == 0);
  choices = pai_json_member(&doc, root, "choices");
  c0 = pai_json_array_at(&doc, choices, 0);
  CHECK(strcmp(pai_json_str_member(&doc, c0, "text"), "bcdabcda") == 0);
  {
    int32_t usage = pai_json_member(&doc, root, "usage");
    CHECK_EQ_INT((int)pai_json_num(&doc, pai_json_member(&doc, usage,
                                                         "completion_tokens")),
                 8);
  }
  pai_json_destroy(&doc);

  /* streaming: one SSE data chunk per token + [DONE] */
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"gw_tiny\",\"prompt\":\"a\",\"temperature\":0,"
                "\"max_tokens\":8,\"stream\":true}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "data: [DONE]") == 1);
  {
    char sse[8192];
    uint32_t sse_len = dechunk_body(&cap, sse, sizeof(sse));
    CHECK_EQ_INT(count_substr_in(sse, sse_len, "data: {"), 8);
    CHECK_EQ_INT(count_substr_in(sse, sse_len, "data: [DONE]"), 1);
    /* deterministic stream: b c d a b c d a */
    CHECK(count_substr_in(sse, sse_len, "\"text\":\"b\"") > 0);
  }

  pai_gw_destroy(&gw);
}

/* ---- chat completions ---- */
{
  pai_gw_t gw;
  cap_t cap;
  pai_json_doc_t doc;
  int32_t root, c0, msg;

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);

  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"model\":\"gw_tiny\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"a\"}],\"temperature\":0,\"max_tokens\":8}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  CHECK(strcmp(pai_json_str_member(&doc, root, "object"), "chat.completion") ==
        0);
  c0 = pai_json_array_at(&doc, pai_json_member(&doc, root, "choices"), 0);
  msg = pai_json_member(&doc, c0, "message");
  CHECK(strcmp(pai_json_str_member(&doc, msg, "role"), "assistant") == 0);
  CHECK(strcmp(pai_json_str_member(&doc, msg, "content"), "bcdabcda") == 0);
  pai_json_destroy(&doc);

  /* invalid messages */
  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"model\":\"gw_tiny\",\"messages\":[]}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);

  /* streaming chat: role chunk + 8 token chunks */
  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"model\":\"gw_tiny\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"a\"}],\"temperature\":0,\"max_tokens\":8,"
                "\"stream\":true}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "chat.completion.chunk") > 0);
  {
    char sse[8192];
    uint32_t sse_len = dechunk_body(&cap, sse, sizeof(sse));
    CHECK_EQ_INT(count_substr_in(sse, sse_len, "data: {"), 9);
    CHECK(count_substr_in(sse, sse_len, "\"delta\":{\"role\":\"assistant\"")
          > 0);
  }

  pai_gw_destroy(&gw);
}

/* ---- embeddings ---- */
{
  pai_gw_t gw;
  cap_t cap;
  pai_json_doc_t doc;
  int32_t root, data, e0, emb;

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);

  CHECK(gw_call(&gw, "POST", "/v1/embeddings",
                "{\"model\":\"gw_tiny\",\"input\":\"a\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  data = pai_json_member(&doc, root, "data");
  CHECK_EQ_INT(pai_json_array_len(&doc, data), 1);
  e0 = pai_json_array_at(&doc, data, 0);
  CHECK(strcmp(pai_json_str_member(&doc, e0, "object"), "embedding") == 0);
  emb = pai_json_member(&doc, e0, "embedding");
  CHECK_EQ_INT(pai_json_array_len(&doc, emb), H);
  /* token "a" -> embedding column 0: [2,0,0,0,0,0,0,0] */
  CHECK(fabs(pai_json_num(&doc, pai_json_array_at(&doc, emb, 0)) - 2.0) <
        1e-4);
  CHECK(fabs(pai_json_num(&doc, pai_json_array_at(&doc, emb, 1)) - 0.0) <
        1e-4);
  pai_json_destroy(&doc);

  /* array input: two vectors */
  CHECK(gw_call(&gw, "POST", "/v1/embeddings",
                "{\"model\":\"gw_tiny\",\"input\":[\"a\",\"b\"]}", &cap) ==
        PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  data = pai_json_member(&doc, root, "data");
  CHECK_EQ_INT(pai_json_array_len(&doc, data), 2);
  pai_json_destroy(&doc);

  /* missing input / unknown model */
  CHECK(gw_call(&gw, "POST", "/v1/embeddings", "{\"model\":\"gw_tiny\"}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);
  CHECK(gw_call(&gw, "POST", "/v1/embeddings",
                "{\"model\":\"nope\",\"input\":\"a\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 404);

  pai_gw_destroy(&gw);
}

/* ---- SDK embed bridge ---- */
{
  pai_model_t *model = NULL;
  float vec[H];
  uint32_t n = 0;
  CHECK(pai_model_open(NULL, "gw_tiny.pai", &model) == PAI_OK);
  CHECK(pai_model_embed_dim(model, &n) == PAI_OK);
  CHECK_EQ_UINT(n, H);
  CHECK(pai_embed(model, "a", vec, H) == PAI_OK);
  CHECK(fabs(vec[0] - 2.0f) < 1e-5f);
  CHECK(pai_embed(model, "ab", vec, H) == PAI_OK);
  CHECK(fabs(vec[0] - 1.0f) < 1e-5f); /* mean of cols 0 and 1 */
  CHECK(fabs(vec[1] - 1.0f) < 1e-5f);
  /* buffer too small */
  CHECK(pai_embed(model, "a", vec, 4) == PAI_ERR_NOMEM);
  pai_model_close(model);
}

/* ---- directory scan ---- */
{
  pai_gw_t gw;
  cap_t cap;
  make_dir("gwdir");
  {
    /* copy the built container into a scanned directory */
    FILE *in = fopen("gw_tiny.pai", "rb");
    FILE *out;
    char buf[8192];
    size_t got;
    CHECK(in != NULL);
    out = fopen("gwdir/other.pai", "wb");
    CHECK(out != NULL);
    while ((got = fread(buf, 1, sizeof(buf), in)) > 0) {
      fwrite(buf, 1, got, out);
    }
    fclose(out);
    fclose(in);
  }
  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_dir(&gw, "gwdir") == PAI_OK);
  CHECK(pai_gw_find(&gw, "other") != NULL);
  CHECK(gw_call(&gw, "GET", "/v1/models", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "\"id\":\"other\"") == 1);
  pai_gw_destroy(&gw);
}

/* ---- out-of-vocabulary input fails cleanly (no crash, no 500) ---- */
{
  pai_gw_t gw;
  cap_t cap;
  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);

  /* byte-fallback ids have no embedding row -> 400, not OOB/crash */
  CHECK(gw_call(&gw, "POST", "/v1/embeddings",
                "{\"model\":\"gw_tiny\",\"input\":\"h\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);

  /* chat prompt with out-of-vocab bytes -> friendly 400, not a 500 */
  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);
  CHECK(count_substr(&cap, "outside the model's vocabulary") == 1);

  /* multi-part message content (valid OpenAI shape) rejected cleanly */
  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":"
                "[{\"type\":\"text\",\"text\":\"a\"}]}]}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 400);

  pai_gw_destroy(&gw);
}

/* ---- response ids are unique across model entries (registry seq) ---- */
{
  pai_gw_t gw;
  cap_t cap;
  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gw_tiny.pai") == PAI_OK);
  CHECK(pai_gw_add_file(&gw, "gwdir/other.pai") == PAI_OK);
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"other\",\"prompt\":\"a\",\"max_tokens\":2}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "\"id\":\"cmpl-00000000\"") == 1);
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"gw_tiny\",\"prompt\":\"a\",\"max_tokens\":2}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "\"id\":\"cmpl-00000001\"") == 1);
  pai_gw_destroy(&gw);
}

/* ---- HTTP idle timeout (slowloris guard) over a real socket ---- */
{
  pai_http_server_t srv;
  pai_gw_sock_t cli = PAI_GW_SOCK_INVALID;
  uint8_t tmp[16];
  uint32_t got = 0;
  pai_status_t st;
  time_t t0, dt;

  CHECK(pai_http_server_init(&srv, "127.0.0.1", 18397, idle_handler,
                             NULL) == PAI_OK);
  srv.idle_timeout_ms = 300;
  CHECK(pai_gw_thread_create(server_run_thread, &srv) == PAI_OK);
  TEST_SLEEP_MS(150); /* let the accept loop start */

  /* Connect, then send nothing: the server must disconnect us. */
  CHECK(pai_gw_sock_connect(&cli, "127.0.0.1", 18397) == PAI_OK);
  CHECK(pai_gw_sock_set_recv_timeout(cli, 4000) == PAI_OK);
  TEST_SLEEP_MS(900); /* well past the 300 ms idle timeout */

  t0 = time(NULL);
  st = pai_gw_sock_recv(cli, tmp, sizeof(tmp), &got);
  dt = time(NULL) - t0;
  /* Closed by the server: EOF or error, and promptly (never our own
   * 4 s recv timeout — that would mean the idle guard did not fire). */
  CHECK(st != PAI_OK || got == 0);
  CHECK(dt <= 1);

  pai_gw_sock_close(cli);
  pai_http_server_close(&srv); /* aborts the accept loop */
  TEST_SLEEP_MS(100);
}

/* ---- remote models over the Prospero protocol ---- */
{
  pai_proto_tcp_listener_t *lst = NULL;
  remote_probe_t probe;
  pai_gw_t gw;
  cap_t cap;
  pai_json_doc_t doc;
  int32_t root, choices, c0, msg;
  uint16_t port = 0;

  CHECK(probe_start(&probe, &lst, &port) == 0);

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_remote(&gw, "remote-lm", "127.0.0.1", port) == PAI_OK);
  CHECK(pai_gw_add_remote(&gw, "remote-lm", "127.0.0.1", port) ==
        PAI_ERR_MISMATCH);
  CHECK(pai_gw_add_remote(&gw, "", "127.0.0.1", port) == PAI_ERR_INVALID_ARG);
  CHECK(pai_gw_add_remote(&gw, "x", NULL, port) == PAI_ERR_INVALID_ARG);
  CHECK(pai_gw_add_remote(&gw, "x", "127.0.0.1", 0) == PAI_ERR_INVALID_ARG);

  /* The registry lists remote entries like any other model. */
  CHECK(gw_call(&gw, "GET", "/v1/models", NULL, &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(count_substr(&cap, "\"id\":\"remote-lm\"") == 1);

  /* Non-stream completion: the probe echoes the prompt as 2-char
   * chunks, so the response text is the prompt itself. */
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"remote-lm\",\"prompt\":\"abcde\","
                "\"max_tokens\":8}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  choices = pai_json_member(&doc, root, "choices");
  c0 = pai_json_array_at(&doc, choices, 0);
  CHECK(strcmp(pai_json_str_member(&doc, c0, "text"), "abcde") == 0);
  {
    int32_t usage = pai_json_member(&doc, root, "usage");
    /* 3 relayed chunks; prompt tokens are unknown to the gateway. */
    CHECK_EQ_INT(
        (int)pai_json_num(&doc, pai_json_member(&doc, usage,
                                                "completion_tokens")),
        3);
    CHECK_EQ_INT(
        (int)pai_json_num(&doc, pai_json_member(&doc, usage,
                                                "prompt_tokens")),
        0);
  }
  pai_json_destroy(&doc);

  /* Streaming: one SSE frame per relayed chunk + [DONE]. */
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"remote-lm\",\"prompt\":\"abcdef\","
                "\"max_tokens\":8,\"stream\":true}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  {
    char sse[4096];
    uint32_t sse_len = dechunk_body(&cap, sse, sizeof(sse));
    CHECK_EQ_INT(count_substr_in(sse, sse_len, "data: {"), 3);
    CHECK_EQ_INT(count_substr_in(sse, sse_len, "data: [DONE]"), 1);
    CHECK(count_substr_in(sse, sse_len, "\"text\":\"ab\"") > 0);
    CHECK(count_substr_in(sse, sse_len, "\"text\":\"ef\"") > 0);
  }

  /* Chat completions over remote. */
  CHECK(gw_call(&gw, "POST", "/v1/chat/completions",
                "{\"model\":\"remote-lm\",\"messages\":[{\"role\":\"user\","
                "\"content\":\"hi\"}],\"max_tokens\":8}",
                &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 200);
  CHECK(parse_resp_json(&cap, &doc) == 0);
  root = pai_json_root(&doc);
  choices = pai_json_member(&doc, root, "choices");
  c0 = pai_json_array_at(&doc, choices, 0);
  msg = pai_json_member(&doc, c0, "message");
  /* The gateway builds "role: content" prompts, which the probe
   * echoes verbatim. */
  CHECK(strcmp(pai_json_str_member(&doc, msg, "content"), "user: hi") == 0);
  pai_json_destroy(&doc);

  /* Embeddings are not bridged in v0. */
  CHECK(gw_call(&gw, "POST", "/v1/embeddings",
                "{\"model\":\"remote-lm\",\"input\":\"a\"}", &cap) ==
        PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 501);

  probe_stop(&probe, lst);
  CHECK_EQ_UINT(probe.finished, 1);
  /* Three generation requests: the accept loop must have been reused. */
  CHECK(probe.n_served >= 3);
  pai_gw_destroy(&gw);
}

/* ---- remote: negotiation refusal and unreachable payload -> 502 ---- */
{
  pai_proto_tcp_listener_t *lst = NULL;
  pai_proto_tcp_listener_t *dead = NULL;
  remote_probe_t probe;
  pai_gw_t gw;
  cap_t cap;
  uint16_t port = 0;
  uint16_t dead_port = 0;

  CHECK(probe_start(&probe, &lst, &port) == 0);
  probe.refuse = 1;

  /* Bind an ephemeral port and release it: nothing is listening. */
  CHECK(pai_proto_tcp_listen(&dead, "127.0.0.1", 0) == PAI_OK);
  CHECK(pai_proto_tcp_listener_port(dead, &dead_port) == PAI_OK);
  pai_proto_tcp_listener_destroy(dead);

  CHECK(pai_gw_init(&gw) == PAI_OK);
  CHECK(pai_gw_add_remote(&gw, "refuse", "127.0.0.1", port) == PAI_OK);
  CHECK(pai_gw_add_remote(&gw, "dead", "127.0.0.1", dead_port) == PAI_OK);

  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"refuse\",\"prompt\":\"a\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 502);
  CHECK(gw_call(&gw, "POST", "/v1/completions",
                "{\"model\":\"dead\",\"prompt\":\"a\"}", &cap) == PAI_OK);
  CHECK_EQ_INT(resp_status(&cap), 502);

  probe_stop(&probe, lst);
  pai_gw_destroy(&gw);
}

/* ---- remote bridge: unresponsive payload -> PAI_ERR_TIMEOUT ---- */
{
  pai_proto_tcp_listener_t *lst = NULL;
  remote_probe_t probe;
  uint32_t tokens = 0;
  uint16_t port = 0;

  CHECK(probe_start(&probe, &lst, &port) == 0);
  probe.no_answer = 1;

  CHECK(pai_gw_remote_generate("127.0.0.1", port, "hi", NULL, NULL,
                               &tokens, 300) == PAI_ERR_TIMEOUT);
  CHECK_EQ_UINT(tokens, 0);

  /* Bad arguments. */
  CHECK(pai_gw_remote_generate(NULL, port, "hi", NULL, NULL, NULL, 0) ==
        PAI_ERR_INVALID_ARG);
  CHECK(pai_gw_remote_generate("127.0.0.1", 0, "hi", NULL, NULL, NULL, 0) ==
        PAI_ERR_INVALID_ARG);

  probe_stop(&probe, lst);
  CHECK_EQ_UINT(probe.finished, 1);
}

TEST_MAIN_END()
