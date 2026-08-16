/*
 * ProsperoAI — OpenAI-compatible gateway (whitepaper §26) — impl
 */

#include "gateway.h"
#include "remote.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <sys/stat.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#define GW_PROMPT_MAX PAI_TOK_MAX_INPUT

/* Stop-sequence limits (§26 stop): OpenAI allows up to 4 sequences of
 * at most 64 chars each. The rolling tail keeps the last
 * GW_TAIL_MAX bytes of generated text so a stop sequence can be
 * matched even when it spans token boundaries (tokens decode to at
 * most 255 chars each). */
#define GW_STOP_MAX 4u
#define GW_STOP_MAX_LEN 64u
#define GW_TAIL_MAX (GW_STOP_MAX_LEN + 256u)

/* ------------------------------------------------------------------ */
/* Registry                                                            */
/* ------------------------------------------------------------------ */

const char *
pai_gw_name_from_path(const char *path, char *out, uint32_t cap) {
  const char *base;
  const char *dot;
  size_t n;

  if (path == NULL || out == NULL || cap == 0) {
    return NULL;
  }
  base = strrchr(path, '/');
  {
    const char *bs = strrchr(path, '\\');
    if (bs != NULL && (base == NULL || bs > base)) {
      base = bs;
    }
  }
  base = base != NULL ? base + 1 : path;
  dot = strrchr(base, '.');
  n = dot != NULL ? (size_t)(dot - base) : strlen(base);
  if (n == 0 || n >= cap) {
    return NULL;
  }
  memcpy(out, base, n);
  out[n] = '\0';
  return out;
}

static int
ends_with_pai(const char *name) {
  size_t n = strlen(name);
  return n >= 4 &&
         ((strcmp(name + n - 4, ".pai") == 0) ||
          (strcmp(name + n - 4, ".PAI") == 0));
}

static pai_gw_model_entry_t *
find_entry(pai_gw_t *gw, const char *name) {
  uint32_t i;
  if (gw == NULL || name == NULL) {
    return NULL;
  }
  for (i = 0; i < gw->num_models; i++) {
    if (strcmp(gw->models[i].name, name) == 0) {
      return &gw->models[i];
    }
  }
  return NULL;
}

const pai_gw_model_entry_t *
pai_gw_find(const pai_gw_t *gw, const char *name) {
  return find_entry((pai_gw_t *)gw, name);
}

pai_status_t
pai_gw_init(pai_gw_t *gw) {
  uint32_t i;
  if (gw == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  memset(gw, 0, sizeof(*gw));
  for (i = 0; i < PAI_GW_MAX_MODELS; i++) {
    if (pai_gw_mutex_init(&gw->models[i].lock) != PAI_OK) {
      /* Roll back. */
      uint32_t j;
      for (j = 0; j < i; j++) {
        pai_gw_mutex_destroy(gw->models[j].lock);
      }
      return PAI_ERR_NOMEM;
    }
  }
  if (pai_gw_mutex_init(&gw->seq_lock) != PAI_OK) {
    for (i = 0; i < PAI_GW_MAX_MODELS; i++) {
      pai_gw_mutex_destroy(gw->models[i].lock);
    }
    return PAI_ERR_NOMEM;
  }
  return PAI_OK;
}

void
pai_gw_destroy(pai_gw_t *gw) {
  uint32_t i;
  if (gw == NULL) {
    return;
  }
  for (i = 0; i < PAI_GW_MAX_MODELS; i++) {
    if (gw->models[i].remote_pool != NULL) {
      pai_remote_pool_destroy(gw->models[i].remote_pool);
    }
    if (gw->models[i].session != NULL) {
      pai_session_destroy(gw->models[i].session);
    }
    if (gw->models[i].model != NULL) {
      pai_model_close(gw->models[i].model);
    }
    pai_gw_mutex_destroy(gw->models[i].lock);
  }
  if (gw->seq_lock != NULL) {
    pai_gw_mutex_destroy(gw->seq_lock);
  }
  memset(gw, 0, sizeof(*gw));
}

pai_status_t
pai_gw_add_file(pai_gw_t *gw, const char *path) {
  pai_gw_model_entry_t *e;

  if (gw == NULL || path == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (gw->num_models >= PAI_GW_MAX_MODELS) {
    return PAI_ERR_NOMEM;
  }
  e = &gw->models[gw->num_models];
  if (pai_gw_name_from_path(path, e->name, sizeof(e->name)) == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (find_entry(gw, e->name) != NULL) {
    return PAI_ERR_MISMATCH; /* duplicate registry id */
  }
  if (strlen(path) >= sizeof(e->path)) {
    return PAI_ERR_INVALID_ARG;
  }
  strcpy(e->path, path);
  gw->num_models++;
  return PAI_OK;
}

pai_status_t
pai_gw_add_remote(pai_gw_t *gw, const char *name, const char *host,
                  uint16_t port) {
  pai_gw_model_entry_t *e;

  if (gw == NULL || name == NULL || name[0] == '\0' || host == NULL ||
      host[0] == '\0' || port == 0) {
    return PAI_ERR_INVALID_ARG;
  }
  if (strpbrk(name, "/\\") != NULL) {
    return PAI_ERR_INVALID_ARG; /* registry ids are path-safe */
  }
  if (gw->num_models >= PAI_GW_MAX_MODELS) {
    return PAI_ERR_NOMEM;
  }
  if (strlen(name) >= sizeof(e->name) ||
      strlen(host) >= sizeof(e->remote_host)) {
    return PAI_ERR_INVALID_ARG;
  }
  if (find_entry(gw, name) != NULL) {
    return PAI_ERR_MISMATCH; /* duplicate registry id */
  }
  e = &gw->models[gw->num_models];
  strcpy(e->name, name);
  strcpy(e->remote_host, host);
  e->remote_port = port;
  e->remote = 1;
  gw->num_models++;
  return PAI_OK;
}

pai_status_t
pai_gw_add_dir(pai_gw_t *gw, const char *dir) {
  pai_status_t st = PAI_OK;
  uint32_t added = 0;

  if (gw == NULL || dir == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
#ifdef _WIN32
  {
    char pattern[1100];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(pattern, sizeof(pattern), "%s\\*.pai", dir);
    h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) {
      return PAI_OK; /* empty directory */
    }
    do {
      char full[1100];
      snprintf(full, sizeof(full), "%s\\%s", dir, fd.cFileName);
      if (pai_gw_add_file(gw, full) == PAI_OK) {
        added++;
      }
    } while (FindNextFileA(h, &fd) != 0);
    FindClose(h);
  }
#else
  {
    DIR *d = opendir(dir);
    if (d == NULL) {
      return PAI_ERR_IO;
    }
    for (;;) {
      struct dirent *de = readdir(d);
      char full[1100];
      if (de == NULL) {
        break;
      }
      if (de->d_name[0] == '.' || !ends_with_pai(de->d_name)) {
        continue;
      }
      snprintf(full, sizeof(full), "%s/%s", dir, de->d_name);
      if (pai_gw_add_file(gw, full) == PAI_OK) {
        added++;
      }
    }
    closedir(d);
  }
#endif
  if (added == 0 && gw->num_models == 0) {
    return PAI_ERR_IO; /* nothing to serve */
  }
  return st;
}

/* ------------------------------------------------------------------ */
/* Shared helpers                                                      */
/* ------------------------------------------------------------------ */

static void
send_raw(pai_http_resp_t *resp, int status, const char *content_type,
         const char *bytes, uint32_t nbytes) {
  if (!resp->begun) {
    pai_http_resp_begin(resp, status, content_type, 0);
  }
  pai_http_resp_write(resp, bytes, nbytes);
  pai_http_resp_end(resp);
}

/* OpenAI-style error object. */
static void
send_error(pai_http_resp_t *resp, int status, const char *type,
           const char *message) {
  pai_json_wb_t wb;
  pai_json_wb_init(&wb);
  pai_json_wb_puts(&wb, "{\"error\":{\"message\":");
  pai_json_quote(&wb, message, (uint32_t)strlen(message));
  pai_json_wb_puts(&wb, ",\"type\":");
  pai_json_quote(&wb, type, (uint32_t)strlen(type));
  pai_json_wb_puts(&wb, ",\"param\":null,\"code\":null}}");
  send_raw(resp, status, "application/json", wb.buf, wb.len);
  pai_json_wb_destroy(&wb);
}

static double
num_member(const pai_json_doc_t *doc, int32_t obj, const char *key,
           double def) {
  int32_t m = pai_json_member(doc, obj, key);
  if (m < 0 || pai_json_type(doc, m) != PAI_JSON_NUMBER) {
    return def;
  }
  return pai_json_num(doc, m);
}

static int
bool_member(const pai_json_doc_t *doc, int32_t obj, const char *key,
            int def) {
  int32_t m = pai_json_member(doc, obj, key);
  if (m < 0) {
    return def;
  }
  return pai_json_type(doc, m) == PAI_JSON_BOOL ? pai_json_bool(doc, m) : def;
}

/* Open the model + session behind an entry (under its lock). Remote
 * entries have nothing to open locally: their persistent connection
 * pool is created lazily and exchanges are bridged over the Prospero
 * Protocol (§24/§25). */
static pai_status_t
ensure_ready(pai_gw_model_entry_t *e) {
  pai_status_t st;

  if (e->broken) {
    return PAI_ERR_INVALID_ARG;
  }
  if (e->remote) {
    if (e->remote_pool == NULL) {
      e->remote_pool =
          pai_remote_pool_create(e->remote_host, e->remote_port);
      if (e->remote_pool == NULL) {
        e->broken = 1;
        return PAI_ERR_NOMEM;
      }
    }
    return PAI_OK;
  }
  if (e->model == NULL) {
    st = pai_model_open(NULL, e->path, &e->model);
    if (st != PAI_OK) {
      e->broken = 1;
      return st;
    }
  }
  if (e->session == NULL) {
    st = pai_session_init(e->model, &e->session);
    if (st != PAI_OK) {
      e->broken = 1;
      return st;
    }
  }
  return PAI_OK;
}

/* Build the prompt for a chat request from the messages array:
 * "role: content" lines joined by newlines (no trailing newline). */
static int
chat_prompt(const pai_json_doc_t *doc, int32_t messages, char *out,
            uint32_t cap) {
  int32_t n = pai_json_array_len(doc, messages);
  uint32_t used = 0;
  int32_t i;

  out[0] = '\0';
  if (n <= 0) {
    return -1;
  }
  for (i = 0; i < n; i++) {
    int32_t msg = pai_json_array_at(doc, messages, (uint32_t)i);
    const char *role;
    const char *content;
    uint32_t need;
    if (pai_json_type(doc, msg) != PAI_JSON_OBJECT) {
      return -1;
    }
    role = pai_json_str_member(doc, msg, "role");
    if (role == NULL) {
      role = "user";
    }
    content = pai_json_str_member(doc, msg, "content");
    if (content == NULL) {
      int32_t cm = pai_json_member(doc, msg, "content");
      /* Present-but-non-string content (e.g. multi-part arrays) is
       * rejected; absent content is treated as empty. */
      if (cm >= 0 && pai_json_type(doc, cm) != PAI_JSON_STRING) {
        return -1;
      }
      content = "";
    }
    need = (uint32_t)(strlen(role) + strlen(content) + (used != 0 ? 2 : 1));
    if (used + need >= cap) {
      return -1;
    }
    if (used != 0) {
      out[used++] = '\n';
    }
    used += (uint32_t)snprintf(out + used, (size_t)(cap - used), "%s: %s",
                               role, content);
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* Completions (POST /v1/completions and /v1/chat/completions)         */
/* ------------------------------------------------------------------ */

typedef struct gw_gen_ctx {
  pai_gw_t *gw;
  pai_gw_model_entry_t *entry;
  pai_http_resp_t *resp;
  pai_json_wb_t *out;   /* non-stream: accumulate body before JSON */
  int chat;
  int stream;
  int failed;
  uint64_t created;
  char id[64];
  uint32_t generated;   /* completion tokens (usage reporting)     */
  /* Stop-sequence tracking (§26 stop). */
  uint32_t n_stops;
  char stops[GW_STOP_MAX][GW_STOP_MAX_LEN + 1];
  char tail[GW_TAIL_MAX]; /* rolling suffix of the generated text   */
  uint32_t tail_len;
  uint32_t stop_len;    /* length of the matched stop (0 = none)    */
  int stop_hit;
  /* response_format: 1 = json_object (post-processed extraction).  */
  int json_mode;
  int echo;             /* completions only: prepend the prompt       */
  int include_usage;    /* stream_options.include_usage (§26)         */
} gw_gen_ctx_t;

/* Emit one SSE data chunk for a generated text fragment. */
static void
emit_chunk(gw_gen_ctx_t *c, const char *text) {
  pai_json_wb_t wb;

  pai_json_wb_init(&wb);
  pai_json_wb_puts(&wb, "{\"id\":");
  pai_json_quote(&wb, c->id, (uint32_t)strlen(c->id));
  pai_json_wb_puts(&wb, ",\"object\":");
  pai_json_wb_puts(&wb, c->chat ? "\"chat.completion.chunk\""
                                : "\"text_completion\"");
  pai_json_wb_putf(&wb, ",\"created\":%llu,\"model\":",
                   (unsigned long long)c->created);
  pai_json_quote(&wb, c->entry->name, (uint32_t)strlen(c->entry->name));
  pai_json_wb_puts(&wb, ",\"choices\":[{\"index\":0,\"");
  if (c->chat) {
    pai_json_wb_puts(&wb, "delta\":{\"role\":\"assistant\",\"content\":");
    pai_json_quote(&wb, text, (uint32_t)strlen(text));
    pai_json_wb_puts(&wb, "},\"finish_reason\":null}]}");
  } else {
    pai_json_wb_puts(&wb, "text\":");
    pai_json_quote(&wb, text, (uint32_t)strlen(text));
    pai_json_wb_puts(&wb, ",\"finish_reason\":null}]}");
  }
  if (pai_http_resp_write(c->resp, "data: ", 6) != PAI_OK ||
      pai_http_resp_write(c->resp, wb.buf, wb.len) != PAI_OK ||
      pai_http_resp_write(c->resp, "\n\n", 2) != PAI_OK) {
    c->failed = 1;
  }
  pai_json_wb_destroy(&wb);
}

/* Trim `s` (len bytes) to its outermost balanced JSON object, moving
 * the object to the front of the buffer. Returns the trimmed length,
 * 0 when no complete object is present. Used by response_format
 * json_object (§26): the v0 runtime has no grammar constraint, so the
 * generated text is post-processed to keep only the JSON object. */
static uint32_t
extract_json_object(char *s, uint32_t len) {
  uint32_t i;
  int depth = 0;
  int in_str = 0;
  int esc = 0;
  uint32_t start = 0;

  for (i = 0; i < len; i++) {
    char ch = s[i];
    if (in_str) {
      if (esc) {
        esc = 0;
      } else if (ch == '\\') {
        esc = 1;
      } else if (ch == '"') {
        in_str = 0;
      }
      continue;
    }
    if (ch == '"') {
      in_str = 1;
      continue;
    }
    if (ch == '{') {
      if (depth == 0) {
        start = i; /* the object we are balancing */
      }
      depth++;
    } else if (ch == '}') {
      if (depth > 0) {
        depth--;
      }
      if (depth == 0) {
        uint32_t n = i + 1 - start;
        if (start > 0 && s != NULL) {
          memmove(s, s + start, n);
        }
        return n; /* balanced object spanned [start, i] */
      }
    }
  }
  return 0;
}

static void
gen_on_token(const char *token, void *user) {
  gw_gen_ctx_t *c = (gw_gen_ctx_t *)user;
  uint32_t tl;
  uint32_t i;

  if (c->failed || c->stop_hit) {
    return; /* error, or a stop sequence already terminated the stream */
  }
  tl = (uint32_t)strlen(token);
  if (c->stream) {
    emit_chunk(c, token);
    if (c->failed) {
      return;
    }
  } else if (pai_json_wb_puts(c->out, token) != PAI_OK) {
    c->failed = 1;
    return;
  }

  /* Stop-sequence detection: match against the tail of the
   * accumulated generated text (longest match wins). Non-streaming
   * output is truncated at the stop; a streaming reply keeps the
   * already-emitted stop bytes (the finish_reason still reports
   * "stop"). A local session is cancelled so generation halts; a
   * remote stream simply stops being relayed. */
  if (c->n_stops == 0) {
    return;
  }
  /* Clamp first so the window math below can never overflow the tail
   * buffer (tokens decode to at most 255 bytes in practice). */
  if (tl > GW_TAIL_MAX - 1) {
    tl = GW_TAIL_MAX - 1;
  }
  if (c->tail_len + tl >= GW_TAIL_MAX) {
    uint32_t keep = GW_TAIL_MAX - 1 - tl;
    if (c->tail_len > keep) {
      memmove(c->tail, c->tail + c->tail_len - keep, keep);
      c->tail_len = keep;
    }
  }
  memcpy(c->tail + c->tail_len, token, tl);
  c->tail_len += tl;
  c->tail[c->tail_len] = '\0';

  for (i = 0; i < c->n_stops; i++) {
    uint32_t sl = (uint32_t)strlen(c->stops[i]);
    if (sl > c->stop_len && c->tail_len >= sl &&
        memcmp(c->tail + c->tail_len - sl, c->stops[i], sl) == 0) {
      c->stop_len = sl;
      c->stop_hit = 1;
    }
  }
  if (c->stop_hit) {
    if (!c->stream && c->out->len >= c->stop_len) {
      c->out->len -= c->stop_len; /* drop the stop bytes from the reply */
      if (c->out->buf != NULL) {
        c->out->buf[c->out->len] = '\0';
      }
    }
    if (!c->entry->remote) {
      pai_session_cancel(c->entry->session);
    }
  }
}

static pai_status_t
handle_completions(pai_gw_t *gw, const uint8_t *body, uint32_t body_len,
                   int chat, pai_http_resp_t *resp) {
  pai_json_doc_t doc;
  int32_t root;
  const char *model_name;
  pai_gw_model_entry_t *entry;
  char prompt[GW_PROMPT_MAX];
  gw_gen_ctx_t ctx;
  pai_status_t st;

  if (body == NULL) {
    send_error(resp, 400, "invalid_request_error", "missing JSON body");
    return PAI_OK;
  }
  st = pai_json_parse(&doc, (const char *)body, body_len);
  if (st != PAI_OK) {
    send_error(resp, 400, "invalid_request_error", "malformed JSON body");
    return PAI_OK;
  }
  root = pai_json_root(&doc);
  if (root < 0 || pai_json_type(&doc, root) != PAI_JSON_OBJECT) {
    pai_json_destroy(&doc);
    send_error(resp, 400, "invalid_request_error", "request must be a JSON object");
    return PAI_OK;
  }

  /* Model selection. */
  model_name = pai_json_str_member(&doc, root, "model");
  entry = find_entry(gw, model_name != NULL ? model_name : gw->num_models > 0
                                                           ? gw->models[0].name
                                                           : NULL);
  if (entry == NULL) {
    pai_json_destroy(&doc);
    send_error(resp, 404, "invalid_request_error", "model not found");
    return PAI_OK;
  }

  /* Prompt. */
  if (chat) {
    int32_t messages = pai_json_member(&doc, root, "messages");
    if (messages < 0 || pai_json_type(&doc, messages) != PAI_JSON_ARRAY ||
        chat_prompt(&doc, messages, prompt, sizeof(prompt)) != 0) {
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error",
                 "messages must be a non-empty array of {role, content}");
      return PAI_OK;
    }
  } else {
    int32_t pn = pai_json_member(&doc, root, "prompt");
    if (pn < 0) {
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error", "missing prompt");
      return PAI_OK;
    }
    if (pai_json_type(&doc, pn) == PAI_JSON_STRING) {
      const char *s = pai_json_str(&doc, pn);
      if (strlen(s) >= sizeof(prompt)) {
        pai_json_destroy(&doc);
        send_error(resp, 400, "invalid_request_error", "prompt too long");
        return PAI_OK;
      }
      strcpy(prompt, s);
    } else if (pai_json_type(&doc, pn) == PAI_JSON_ARRAY) {
      /* Join string elements (multi-part prompt). */
      int32_t n = pai_json_array_len(&doc, pn);
      uint32_t used = 0;
      int32_t i;
      prompt[0] = '\0';
      for (i = 0; i < n; i++) {
        int32_t e = pai_json_array_at(&doc, pn, (uint32_t)i);
        const char *s;
        if (pai_json_type(&doc, e) != PAI_JSON_STRING) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "prompt array elements must be strings");
          return PAI_OK;
        }
        s = pai_json_str(&doc, e);
        if (used + strlen(s) + 1 >= sizeof(prompt)) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error", "prompt too long");
          return PAI_OK;
        }
        used += (uint32_t)strlen(s);
        strcat(prompt, s);
      }
    } else {
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error",
                 "prompt must be a string or array of strings");
      return PAI_OK;
    }
  }

  /* Generation parameters. */
  {
    double temp = num_member(&doc, root, "temperature", 0.0);
    double top_p = num_member(&doc, root, "top_p", 1.0);
    double top_k = num_member(&doc, root, "top_k", 0.0);
    double mt = num_member(&doc, root, "max_tokens", 16.0);
    uint32_t max_tokens;
    pai_json_wb_t out_wb;

    if (temp < 0.0 || top_p <= 0.0 || top_p > 1.0 || top_k < 0.0 ||
        mt < 1.0) {
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error", "invalid sampling parameters");
      return PAI_OK;
    }
    max_tokens = mt > 2048.0 ? 2048u : (uint32_t)mt;
    memset(&ctx, 0, sizeof(ctx));
    ctx.gw = gw;
    ctx.entry = entry;
    ctx.resp = resp;
    ctx.chat = chat;
    ctx.stream = bool_member(&doc, root, "stream", 0);
    /* stream_options (§26): OpenAI's streaming extension. v0 honors
     * include_usage: a final chunk with an empty choices array and
     * the request's usage object is emitted right before [DONE].
     * Like OpenAI, stream_options is only valid with stream: true. */
    {
      int32_t so = pai_json_member(&doc, root, "stream_options");
      if (so >= 0) {
        if (pai_json_type(&doc, so) != PAI_JSON_OBJECT) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "stream_options must be an object");
          return PAI_OK;
        }
        if (!ctx.stream) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "stream_options requires stream:true");
          return PAI_OK;
        }
        ctx.include_usage = bool_member(&doc, so, "include_usage", 0);
      }
    }
    /* Stop sequences (§26 stop): a string or an array of up to
     * GW_STOP_MAX non-empty strings of at most GW_STOP_MAX_LEN
     * chars. Parsed before out_wb is initialized (early 400 paths
     * below need no write-buffer cleanup). */
    {
      int32_t stop_node = pai_json_member(&doc, root, "stop");
      if (stop_node >= 0) {
        if (pai_json_type(&doc, stop_node) == PAI_JSON_STRING) {
          const char *s = pai_json_str(&doc, stop_node);
          if (s[0] == '\0' || strlen(s) > GW_STOP_MAX_LEN) {
            pai_json_destroy(&doc);
            send_error(resp, 400, "invalid_request_error",
                       "stop must be a non-empty string of at most 64 chars");
            return PAI_OK;
          }
          strcpy(ctx.stops[0], s);
          ctx.n_stops = 1;
        } else if (pai_json_type(&doc, stop_node) == PAI_JSON_ARRAY) {
          int32_t n = pai_json_array_len(&doc, stop_node);
          int32_t i;
          if (n <= 0 || n > (int32_t)GW_STOP_MAX) {
            pai_json_destroy(&doc);
            send_error(resp, 400, "invalid_request_error",
                       "stop must contain 1..4 strings");
            return PAI_OK;
          }
          for (i = 0; i < n; i++) {
            int32_t e = pai_json_array_at(&doc, stop_node, (uint32_t)i);
            const char *s;
            if (pai_json_type(&doc, e) != PAI_JSON_STRING) {
              pai_json_destroy(&doc);
              send_error(resp, 400, "invalid_request_error",
                         "stop must be a string or array of strings");
              return PAI_OK;
            }
            s = pai_json_str(&doc, e);
            if (s[0] == '\0' || strlen(s) > GW_STOP_MAX_LEN) {
              pai_json_destroy(&doc);
              send_error(resp, 400, "invalid_request_error",
                         "stop must be non-empty strings of at most 64 chars");
              return PAI_OK;
            }
            strcpy(ctx.stops[ctx.n_stops++], s);
          }
        } else {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "stop must be a string or array of strings");
          return PAI_OK;
        }
      }
    }
    /* response_format (§26): {"type":"text"} (default) or
     * {"type":"json_object"}. v0 has no grammar constraint, so
     * json_object post-processes the reply to keep only the first
     * balanced JSON object (non-streaming only). */
    {
      int32_t rf = pai_json_member(&doc, root, "response_format");
      if (rf >= 0) {
        const char *type;
        if (pai_json_type(&doc, rf) != PAI_JSON_OBJECT) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "response_format must be an object");
          return PAI_OK;
        }
        type = pai_json_str_member(&doc, rf, "type");
        if (type == NULL) {
          type = "text";
        }
        if (strcmp(type, "json_object") == 0) {
          ctx.json_mode = 1;
        } else if (strcmp(type, "text") != 0) {
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "response_format type must be text or json_object");
          return PAI_OK;
        }
      }
    }
    /* v0 json_object post-processes the reply, which is impossible on
     * a live stream: refuse the combination instead of silently
     * streaming unconstrained output. */
    if (ctx.json_mode && ctx.stream) {
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error",
                 "response_format json_object is not supported with stream");
      return PAI_OK;
    }
    ctx.created = (uint64_t)time(NULL);
    {
      /* Allocate a unique response id (registry-wide). */
      uint32_t seq;
      pai_gw_mutex_lock(gw->seq_lock);
      seq = gw->seq++;
      pai_gw_mutex_unlock(gw->seq_lock);
      snprintf(ctx.id, sizeof(ctx.id), chat ? "chatcmpl-%08x"
                                            : "cmpl-%08x", seq);
    }
    pai_json_wb_init(&out_wb);
    ctx.out = &out_wb;

    pai_gw_mutex_lock(entry->lock);
    st = ensure_ready(entry);
    if (st != PAI_OK) {
      pai_gw_mutex_unlock(entry->lock);
      pai_json_wb_destroy(&out_wb);
      pai_json_destroy(&doc);
      send_error(resp, 500, "server_error", "model failed to load");
      return PAI_OK;
    }

    if (!entry->remote) {
      /* Apply request sampling config + generation cap (local). */
      entry->session->sampler.temperature = (float)temp;
      entry->session->sampler.top_k = (uint32_t)top_k;
      entry->session->sampler.top_p = (float)top_p;
      pai_session_set_generation(entry->session, max_tokens, 0);
    }
    /* Seed (§31 reproducibility): re-init the local PRNG while
     * preserving the request's sampling config. Remote payloads own
     * their RNG (the wire sampler carries no seed in v0). */
    {
      double seed = num_member(&doc, root, "seed", -1.0);
      if (seed < -1.0 || seed > 18446744073709551615.0) {
        pai_gw_mutex_unlock(entry->lock);
        pai_json_wb_destroy(&out_wb);
        pai_json_destroy(&doc);
        send_error(resp, 400, "invalid_request_error", "seed out of range");
        return PAI_OK;
      }
      if (seed >= 0.0 && !entry->remote) {
        pai_sampler_t saved = entry->session->sampler;
        pai_sampler_init(&entry->session->sampler, (uint64_t)seed);
        entry->session->sampler.temperature = saved.temperature;
        entry->session->sampler.top_k = saved.top_k;
        entry->session->sampler.top_p = saved.top_p;
        entry->session->seed = (uint32_t)(uint64_t)seed;
      }
    }
    /* echo (completions only): prepend the prompt to the reply. */
    ctx.echo = !chat && bool_member(&doc, root, "echo", 0);
    {
      /* Remote: carry the request's effective sampler settings in the
       * GENERATE trailer so the payload applies them (zero/default
       * values fall back to the payload's own defaults). */
      pai_proto_sampler_t wire_sampler;
      const pai_proto_sampler_t *smp = NULL;
      memset(&wire_sampler, 0, sizeof(wire_sampler));
      if (entry->remote) {
        wire_sampler.temperature = (float)temp;
        wire_sampler.top_p = (float)top_p;
        wire_sampler.top_k = (uint32_t)top_k;
        wire_sampler.max_tokens = max_tokens;
        smp = &wire_sampler;
      }

    if (ctx.stream) {
      pai_http_resp_begin(resp, 200, "text/event-stream", 1);
      /* Leading chat chunk carries the role (OpenAI convention). */
      if (chat) {
        pai_json_wb_t wb;
        pai_json_wb_init(&wb);
        pai_json_wb_puts(&wb, "{\"id\":");
        pai_json_quote(&wb, ctx.id, (uint32_t)strlen(ctx.id));
        pai_json_wb_puts(&wb, ",\"object\":\"chat.completion.chunk\",\"created\":");
        pai_json_wb_putf(&wb, "%llu,\"model\":", (unsigned long long)ctx.created);
        pai_json_quote(&wb, entry->name, (uint32_t)strlen(entry->name));
        pai_json_wb_puts(&wb,
                         ",\"choices\":[{\"index\":0,\"delta\":{\"role\":"
                         "\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}");
        pai_http_resp_write(resp, "data: ", 6);
        pai_http_resp_write(resp, wb.buf, wb.len);
        pai_http_resp_write(resp, "\n\n", 2);
        pai_json_wb_destroy(&wb);
      }
      if (ctx.echo) {
        /* echo: the prompt leads the streamed completion. */
        emit_chunk(&ctx, prompt);
      }
      if (entry->remote) {
        st = pai_remote_pool_generate(entry->remote_pool, prompt, smp,
                                      gen_on_token, &ctx, &ctx.generated, 0);
        ctx.failed = ctx.failed || st != PAI_OK;
      } else {
        st = pai_session_generate(entry->session, prompt, gen_on_token, &ctx);
        ctx.generated = (uint32_t)entry->session->generated_tokens;
      }
      if (st != PAI_OK || ctx.failed) {
        /* The 200 status is already on the wire, so a mid-stream
         * failure is surfaced as a final SSE error event (§26). */
        static const char hdr[] = "data: {\"error\":{\"message\":\"";
        static const char tail[] =
            "\",\"type\":\"server_error\",\"param\":null,\"code\":null}}\n\n";
        const char *msg =
            entry->remote ? "remote payload error" : "generation failed";
        pai_http_resp_write(resp, hdr, sizeof(hdr) - 1);
        pai_http_resp_write(resp, msg, (uint32_t)strlen(msg));
        pai_http_resp_write(resp, tail, sizeof(tail) - 1);
      } else {
        /* Final chunk carries the finish_reason (§26 OpenAI shape):
         * "length" when the max_tokens cap was hit, "stop" for stop
         * sequences, EOS, or any other early termination. */
        const char *reason = ctx.stop_hit
                                 ? "stop"
                                 : (ctx.generated >= max_tokens ? "length"
                                                                : "stop");
        pai_json_wb_t wb;
        pai_json_wb_init(&wb);
        pai_json_wb_puts(&wb, "{\"id\":");
        pai_json_quote(&wb, ctx.id, (uint32_t)strlen(ctx.id));
        pai_json_wb_puts(&wb, ",\"object\":");
        pai_json_wb_puts(&wb, chat ? "\"chat.completion.chunk\""
                                   : "\"text_completion\"");
        pai_json_wb_putf(&wb, ",\"created\":%llu,\"model\":",
                         (unsigned long long)ctx.created);
        pai_json_quote(&wb, entry->name, (uint32_t)strlen(entry->name));
        pai_json_wb_puts(&wb, ",\"choices\":[{\"index\":0,\"");
        if (chat) {
          pai_json_wb_puts(&wb, "delta\":{},\"finish_reason\":");
        } else {
          pai_json_wb_puts(&wb, "text\":\"\",\"finish_reason\":");
        }
        pai_json_quote(&wb, reason, (uint32_t)strlen(reason));
        pai_json_wb_puts(&wb, "}]}");
        if (pai_http_resp_write(resp, "data: ", 6) != PAI_OK ||
            pai_http_resp_write(resp, wb.buf, wb.len) != PAI_OK ||
            pai_http_resp_write(resp, "\n\n", 2) != PAI_OK) {
          /* The client is gone; nothing left to do. */
        }
        pai_json_wb_destroy(&wb);
        /* stream_options.include_usage: a final chunk with empty
         * choices carries the usage totals (OpenAI streaming shape).
         * Prompt tokens are counted locally when the tokenizer is
         * available; remote payloads own their tokenizer, so v0
         * reports prompt_tokens as 0 (mirroring the non-stream
         * response). */
        if (ctx.include_usage) {
          uint32_t ptok = 0;
          pai_json_wb_t wb2;
          if (!entry->remote) {
            /* 64 KiB of ids: a u16 prompt can encode to far more than
             * a small fixed cap, so count generously (the HTTP thread
             * has a 1 MB stack). */
            enum { PTOK_MAX = 16384 };
            uint32_t *ids = (uint32_t *)malloc(PTOK_MAX * sizeof(uint32_t));
            if (ids != NULL) {
              uint32_t nids = 0;
              if (pai_tok_encode(&entry->model->tokenizer, prompt,
                                 (uint32_t)strlen(prompt), ids, PTOK_MAX,
                                 &nids) == PAI_OK) {
                ptok = nids;
              }
              free(ids);
            }
          }
          pai_json_wb_init(&wb2);
          pai_json_wb_puts(&wb2, "{\"id\":");
          pai_json_quote(&wb2, ctx.id, (uint32_t)strlen(ctx.id));
          pai_json_wb_puts(&wb2, ",\"object\":");
          pai_json_wb_puts(&wb2, chat ? "\"chat.completion.chunk\""
                                      : "\"text_completion\"");
          pai_json_wb_putf(&wb2, ",\"created\":%llu,\"model\":",
                           (unsigned long long)ctx.created);
          pai_json_quote(&wb2, entry->name, (uint32_t)strlen(entry->name));
          pai_json_wb_puts(&wb2, ",\"choices\":[],\"usage\":"
                                 "{\"prompt_tokens\":");
          pai_json_wb_putf(&wb2, "%u,\"completion_tokens\":%u,"
                                 "\"total_tokens\":%u}}",
                           ptok, ctx.generated, ptok + ctx.generated);
          if (pai_http_resp_write(resp, "data: ", 6) != PAI_OK ||
              pai_http_resp_write(resp, wb2.buf, wb2.len) != PAI_OK ||
              pai_http_resp_write(resp, "\n\n", 2) != PAI_OK) {
            /* The client is gone; nothing left to do. */
          }
          pai_json_wb_destroy(&wb2);
        }
      }
      pai_http_resp_write(resp, "data: [DONE]\n\n", 15);
      pai_http_resp_end(resp);
    } else {
      if (entry->remote) {
        st = pai_remote_pool_generate(entry->remote_pool, prompt, smp,
                                      gen_on_token, &ctx, &ctx.generated, 0);
        ctx.failed = ctx.failed || st != PAI_OK;
      } else {
        st = pai_session_generate(entry->session, prompt, gen_on_token, &ctx);
        ctx.generated = (uint32_t)entry->session->generated_tokens;
      }
      if (st == PAI_OK && !ctx.failed) {
        /* Prompt token count for usage reporting (remote: the payload
         * owns the tokenizer, so v0 reports prompt_tokens as 0). */
        uint32_t ptok = 0;
        const char *reason = ctx.stop_hit
                                 ? "stop"
                                 : (ctx.generated >= max_tokens ? "length"
                                                                : "stop");
        const char *text_buf;
        uint32_t text_n;
        char *combined = NULL;
        if (!entry->remote) {
          /* 64 KiB of ids: a u16 prompt can encode to far more than
           * a small fixed cap, so count generously (the HTTP thread
           * has a 1 MB stack). */
          enum { PTOK_MAX = 16384 };
          uint32_t *ids = (uint32_t *)malloc(PTOK_MAX * sizeof(uint32_t));
          if (ids != NULL) {
            uint32_t nids = 0;
            if (pai_tok_encode(&entry->model->tokenizer, prompt,
                               (uint32_t)strlen(prompt), ids, PTOK_MAX,
                               &nids) == PAI_OK) {
              ptok = nids;
            }
            free(ids);
          }
        }
        /* response_format json_object: keep only the first balanced
         * JSON object of the reply; no object -> 400 (the model did
         * not honor the requested format). */
        if (ctx.json_mode) {
          uint32_t obj_len = extract_json_object(
              ctx.out->buf != NULL ? ctx.out->buf : "", ctx.out->len);
          if (obj_len == 0) {
            pai_gw_mutex_unlock(entry->lock);
            pai_json_wb_destroy(&out_wb);
            pai_json_destroy(&doc);
            send_error(resp, 400, "invalid_request_error",
                       "response did not match response_format json_object");
            return PAI_OK;
          }
          ctx.out->len = obj_len;
          if (ctx.out->buf != NULL) {
            ctx.out->buf[obj_len] = '\0';
          }
        }
        /* echo (completions only): the reply is prompt + completion. */
        if (ctx.echo) {
          uint32_t plen = (uint32_t)strlen(prompt);
          combined = (char *)malloc((size_t)plen + ctx.out->len + 1);
          if (combined == NULL) {
            pai_gw_mutex_unlock(entry->lock);
            pai_json_wb_destroy(&out_wb);
            pai_json_destroy(&doc);
            send_error(resp, 500, "server_error", "out of memory");
            return PAI_OK;
          }
          memcpy(combined, prompt, plen);
          memcpy(combined + plen,
                 ctx.out->buf != NULL ? ctx.out->buf : "", ctx.out->len);
          combined[plen + ctx.out->len] = '\0';
          text_buf = combined;
          text_n = plen + ctx.out->len;
        } else {
          text_buf = ctx.out->buf != NULL ? ctx.out->buf : "";
          text_n = ctx.out->len;
        }
        {
          pai_json_wb_t wb;
          pai_json_wb_init(&wb);
          pai_json_wb_puts(&wb, "{\"id\":");
          pai_json_quote(&wb, ctx.id, (uint32_t)strlen(ctx.id));
          pai_json_wb_puts(&wb, ",\"object\":");
          pai_json_wb_puts(&wb, chat ? "\"chat.completion\"" : "\"text_completion\"");
          pai_json_wb_putf(&wb, ",\"created\":%llu,\"model\":",
                           (unsigned long long)ctx.created);
          pai_json_quote(&wb, entry->name, (uint32_t)strlen(entry->name));
          pai_json_wb_puts(&wb, ",\"choices\":[{\"index\":0,\"");
          if (chat) {
            pai_json_wb_puts(&wb, "message\":{\"role\":\"assistant\",\"content\":");
            pai_json_quote(&wb, text_buf, text_n);
            pai_json_wb_puts(&wb, "},\"finish_reason\":");
            pai_json_quote(&wb, reason, (uint32_t)strlen(reason));
            pai_json_wb_puts(&wb, "}],\"usage\":{\"prompt_tokens\":");
          } else {
            pai_json_wb_puts(&wb, "text\":");
            pai_json_quote(&wb, text_buf, text_n);
            pai_json_wb_puts(&wb, ",\"finish_reason\":");
            pai_json_quote(&wb, reason, (uint32_t)strlen(reason));
            pai_json_wb_puts(&wb, "}],\"usage\":{\"prompt_tokens\":");
          }
          pai_json_wb_putf(&wb, "%u,\"completion_tokens\":%llu,\"total_tokens\":%u}}",
                           ptok, (unsigned long long)ctx.generated,
                           ptok + ctx.generated);
          send_raw(resp, 200, "application/json", wb.buf, wb.len);
          pai_json_wb_destroy(&wb);
        }
        free(combined);
      } else {
        if (st == PAI_ERR_MISMATCH && !entry->remote) {
          send_error(resp, 400, "invalid_request_error",
                     "input contains tokens outside the model's vocabulary");
        } else if (entry->remote && st != PAI_OK) {
          send_error(resp, 502, "server_error", "remote payload error");
        } else {
          send_error(resp, 500, "server_error",
                     st == PAI_OK ? "generation failed" : pai_status_str(st));
        }
      }
    }
    }
    pai_gw_mutex_unlock(entry->lock);
    pai_json_wb_destroy(&out_wb);
  }

  pai_json_destroy(&doc);
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Embeddings (POST /v1/embeddings)                                    */
/* ------------------------------------------------------------------ */

static pai_status_t
handle_embeddings(pai_gw_t *gw, const uint8_t *body, uint32_t body_len,
                  pai_http_resp_t *resp) {
  pai_json_doc_t doc;
  int32_t root;
  const char *model_name;
  pai_gw_model_entry_t *entry;
  pai_status_t st;
  int32_t input_node;
  uint32_t dim = 0;

  if (body == NULL) {
    send_error(resp, 400, "invalid_request_error", "missing JSON body");
    return PAI_OK;
  }
  st = pai_json_parse(&doc, (const char *)body, body_len);
  if (st != PAI_OK) {
    send_error(resp, 400, "invalid_request_error", "malformed JSON body");
    return PAI_OK;
  }
  root = pai_json_root(&doc);
  if (root < 0 || pai_json_type(&doc, root) != PAI_JSON_OBJECT) {
    pai_json_destroy(&doc);
    send_error(resp, 400, "invalid_request_error",
               "request must be a JSON object");
    return PAI_OK;
  }

  model_name = pai_json_str_member(&doc, root, "model");
  entry = find_entry(gw, model_name != NULL ? model_name : gw->num_models > 0
                                                           ? gw->models[0].name
                                                           : NULL);
  if (entry == NULL) {
    pai_json_destroy(&doc);
    send_error(resp, 404, "invalid_request_error", "model not found");
    return PAI_OK;
  }
  input_node = pai_json_member(&doc, root, "input");
  if (input_node < 0) {
    pai_json_destroy(&doc);
    send_error(resp, 400, "invalid_request_error", "missing input");
    return PAI_OK;
  }

  pai_gw_mutex_lock(entry->lock);
  st = ensure_ready(entry);
  if (st == PAI_OK && !entry->remote) {
    /* Remote entries report their dimension with the reply vector. */
    st = pai_model_embed_dim(entry->model, &dim);
  }
  if (st != PAI_OK) {
    pai_gw_mutex_unlock(entry->lock);
    pai_json_destroy(&doc);
    send_error(resp, st == PAI_ERR_UNSUPPORTED ? 501 : 500,
               "server_error",
               st == PAI_ERR_UNSUPPORTED
                   ? "model has no embedding table"
                   : "model failed to load");
    return PAI_OK;
  }

  {
    int32_t count;
    int32_t is_array;
    int32_t i;
    pai_json_wb_t wb;

    if (pai_json_type(&doc, input_node) == PAI_JSON_STRING) {
      count = 1;
      is_array = 0;
    } else if (pai_json_type(&doc, input_node) == PAI_JSON_ARRAY) {
      count = pai_json_array_len(&doc, input_node);
      is_array = 1;
    } else {
      pai_gw_mutex_unlock(entry->lock);
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error",
                 "input must be a string or array of strings");
      return PAI_OK;
    }
    if (count <= 0) {
      pai_gw_mutex_unlock(entry->lock);
      pai_json_destroy(&doc);
      send_error(resp, 400, "invalid_request_error", "input must not be empty");
      return PAI_OK;
    }

    pai_json_wb_init(&wb);
    pai_json_wb_puts(&wb, "{\"object\":\"list\",\"data\":[");
    for (i = 0; i < count; i++) {
      const char *text;
      float *vec;
      uint32_t n = 0;
      uint32_t k;

      if (is_array) {
        int32_t e = pai_json_array_at(&doc, input_node, (uint32_t)i);
        if (pai_json_type(&doc, e) != PAI_JSON_STRING) {
          pai_gw_mutex_unlock(entry->lock);
          pai_json_wb_destroy(&wb);
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error",
                     "input array elements must be strings");
          return PAI_OK;
        }
        text = pai_json_str(&doc, e);
      } else {
        text = pai_json_str(&doc, input_node);
      }
      /* Empty input text is rejected for both local and remote
       * (the local path's mean-pool over zero tokens would fail). */
      if (text[0] == '\0') {
        pai_gw_mutex_unlock(entry->lock);
        pai_json_wb_destroy(&wb);
        pai_json_destroy(&doc);
        send_error(resp, 400, "invalid_request_error",
                   "input must not be empty");
        return PAI_OK;
      }
      if (entry->remote) {
        /* Bridge the request to the payload over the pooled
         * connection (one EMBED exchange per input element). */
        st = pai_remote_pool_embed(entry->remote_pool, text,
                                   (uint32_t)strlen(text), &vec, &n, 0);
        if (st != PAI_OK) {
          pai_gw_mutex_unlock(entry->lock);
          pai_json_wb_destroy(&wb);
          pai_json_destroy(&doc);
          send_error(resp, st == PAI_ERR_UNSUPPORTED
                               ? 501
                               : (st == PAI_ERR_INVALID_ARG ? 400 : 502),
                     "server_error",
                     st == PAI_ERR_UNSUPPORTED
                         ? "remote payload does not expose embeddings"
                         : (st == PAI_ERR_INVALID_ARG
                                ? "invalid input"
                                : "remote payload error"));
          return PAI_OK;
        }
      } else {
        vec = (float *)malloc((size_t)dim * sizeof(float));
        if (vec == NULL) {
          pai_gw_mutex_unlock(entry->lock);
          pai_json_wb_destroy(&wb);
          pai_json_destroy(&doc);
          send_error(resp, 500, "server_error", "out of memory");
          return PAI_OK;
        }
        st = pai_model_embed(entry->model, text, dim, vec, &n);
        if (st != PAI_OK || n < dim) {
          free(vec);
          pai_gw_mutex_unlock(entry->lock);
          pai_json_wb_destroy(&wb);
          pai_json_destroy(&doc);
          send_error(resp, 400, "invalid_request_error", "embedding failed");
          return PAI_OK;
        }
      }
      if (i != 0) {
        pai_json_wb_puts(&wb, ",");
      }
      pai_json_wb_putf(&wb, "{\"object\":\"embedding\",\"index\":%d,\"embedding\":[",
                       i);
      for (k = 0; k < n; k++) {
        if (k != 0) {
          pai_json_wb_puts(&wb, ",");
        }
        pai_json_put_number(&wb, vec[k]);
      }
      pai_json_wb_puts(&wb, "]}");
      free(vec);
    }
    pai_json_wb_puts(&wb, "]}");
    pai_gw_mutex_unlock(entry->lock);
    send_raw(resp, 200, "application/json", wb.buf, wb.len);
    pai_json_wb_destroy(&wb);
  }

  pai_json_destroy(&doc);
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Model listing                                                       */
/* ------------------------------------------------------------------ */

/* Real metadata for the model listing (§26): local entries report the
 * on-disk .pai size, remote entries their Prospero Protocol endpoint;
 * `broken` surfaces entries whose lazy open failed. All fields are
 * derivable without touching the model itself (no forced load). */
static int64_t
file_size_bytes(const char *path) {
  struct stat st;
  if (path == NULL || stat(path, &st) != 0) {
    return 0;
  }
  return (int64_t)st.st_size;
}

static void
send_model_object(pai_json_wb_t *wb, const pai_gw_model_entry_t *e,
                  uint32_t *seq) {
  pai_json_wb_puts(wb, "{\"id\":");
  pai_json_quote(wb, e->name, (uint32_t)strlen(e->name));
  pai_json_wb_puts(wb, ",\"object\":\"model\",\"created\":0,\"owned_by\":");
  pai_json_quote(wb, "prosperoai", 10);
  pai_json_wb_puts(wb, ",\"index\":");
  pai_json_wb_putf(wb, "%u", *seq);
  if (e->remote) {
    char endpoint[sizeof(e->remote_host) + 16];
    pai_json_wb_puts(wb, ",\"kind\":\"remote\",\"endpoint\":");
    snprintf(endpoint, sizeof(endpoint), "%s:%u", e->remote_host,
             (unsigned)e->remote_port);
    pai_json_quote(wb, endpoint, (uint32_t)strlen(endpoint));
  } else {
    pai_json_wb_puts(wb, ",\"kind\":\"local\"");
  }
  pai_json_wb_puts(wb, ",\"size_bytes\":");
  pai_json_wb_putf(wb, "%lld",
                   (long long)(e->remote ? 0 : file_size_bytes(e->path)));
  pai_json_wb_puts(wb, ",\"broken\":");
  pai_json_wb_puts(wb, e->broken ? "true}" : "false}");
  (*seq)++;
}

static pai_status_t
handle_models(pai_gw_t *gw, const char *path, pai_http_resp_t *resp) {
  uint32_t i;
  pai_json_wb_t wb;
  uint32_t seq = 0;

  /* Single-model GET /v1/models/{id}. */
  if (strncmp(path, "/v1/models/", 11) == 0) {
    const char *name = path + 11;
    const pai_gw_model_entry_t *e = pai_gw_find(gw, name);
    if (e == NULL) {
      send_error(resp, 404, "invalid_request_error", "model not found");
      return PAI_OK;
    }
    pai_json_wb_init(&wb);
    send_model_object(&wb, e, &seq);
    send_raw(resp, 200, "application/json", wb.buf, wb.len);
    pai_json_wb_destroy(&wb);
    return PAI_OK;
  }

  if (strcmp(path, "/v1/models") != 0) {
    send_error(resp, 404, "invalid_request_error", "not found");
    return PAI_OK;
  }

  pai_json_wb_init(&wb);
  pai_json_wb_puts(&wb, "{\"object\":\"list\",\"data\":[");
  for (i = 0; i < gw->num_models; i++) {
    if (i != 0) {
      pai_json_wb_puts(&wb, ",");
    }
    send_model_object(&wb, &gw->models[i], &seq);
  }
  pai_json_wb_puts(&wb, "]}");
  send_raw(resp, 200, "application/json", wb.buf, wb.len);
  pai_json_wb_destroy(&wb);
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* Dispatch                                                            */
/* ------------------------------------------------------------------ */

pai_status_t
pai_gw_handle_request(pai_gw_t *gw, const char *method, const char *target,
                      const uint8_t *body, uint32_t body_len,
                      pai_http_resp_t *resp) {
  char path[1024];
  char *q;

  if (gw == NULL || method == NULL || target == NULL || resp == NULL) {
    return PAI_ERR_INVALID_ARG;
  }
  if (strlen(target) >= sizeof(path)) {
    send_error(resp, 404, "invalid_request_error", "not found");
    return PAI_OK;
  }
  strcpy(path, target);
  q = strchr(path, '?');
  if (q != NULL) {
    *q = '\0';
  }

  if (strcmp(method, "GET") == 0) {
    if (strcmp(path, "/healthz") == 0) {
      send_raw(resp, 200, "text/plain", "ok\n", 3);
      return PAI_OK;
    }
    return handle_models(gw, path, resp);
  }
  if (strcmp(method, "POST") == 0) {
    if (strcmp(path, "/v1/completions") == 0) {
      return handle_completions(gw, body, body_len, 0, resp);
    }
    if (strcmp(path, "/v1/chat/completions") == 0) {
      return handle_completions(gw, body, body_len, 1, resp);
    }
    if (strcmp(path, "/v1/embeddings") == 0) {
      return handle_embeddings(gw, body, body_len, resp);
    }
    send_error(resp, 404, "invalid_request_error", "not found");
    return PAI_OK;
  }
  send_error(resp, 405, "invalid_request_error", "method not allowed");
  return PAI_OK;
}
