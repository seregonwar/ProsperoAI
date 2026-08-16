/*
 * ProsperoAI — Prospero Protocol
 *
 * Connection state machine (whitepaper §24): negotiation, pipelining,
 * sessions, streams, structured errors. Transport-independent — the
 * connection pulls bytes through a pai_proto_transport_t and drives a
 * receive state machine that buffers one frame at a time.
 *
 * Core-handled messages: HELLO/HELLO_ACK/HELLO_NACK (negotiation),
 * PING->PONG (keepalive), CLOSE (graceful shutdown) and ERROR
 * auto-replies when a handler fails. Everything else reaches the
 * application through the callbacks.
 */

#include <protocol/protocol.h>

#include <pai/log.h>

#include <stdlib.h>
#include <string.h>

/* Max frames processed per poll() call (fairness bound). */
#define PAI_PROTO_POLL_BATCH 64u

/* ------------------------------------------------------------------ */
/* little-endian writers (frame.c owns the canonical codec; this file  */
/* keeps a private copy to build headers without a full wire buffer)   */
/* ------------------------------------------------------------------ */

static void
put_le16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)(v >> 8);
}

static void
put_le32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)(v & 0xFFu);
  p[1] = (uint8_t)((v >> 8) & 0xFFu);
  p[2] = (uint8_t)((v >> 16) & 0xFFu);
  p[3] = (uint8_t)(v >> 24);
}

static void
put_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) {
    p[i] = (uint8_t)((v >> (8 * i)) & 0xFFu);
  }
}

/* ------------------------------------------------------------------ */
/* lifecycle                                                           */
/* ------------------------------------------------------------------ */

static void
conn_set_closed(pai_proto_conn_t *conn, uint32_t reason) {
  if (conn->state == PAI_PROTO_STATE_CLOSED) {
    return;
  }
  conn->state = PAI_PROTO_STATE_CLOSED;
  conn->transport.close(conn->transport.ctx);
  if (conn->callbacks.on_close) {
    conn->callbacks.on_close(conn->user, reason);
  }
}

static void
conn_reset_phase(pai_proto_conn_t *conn) {
  free(conn->payload_buf);
  conn->payload_buf = NULL;
  conn->payload_filled = 0;
  conn->header_filled = 0;
  memset(&conn->pending, 0, sizeof(conn->pending));
}

pai_status_t
pai_proto_conn_init(pai_proto_conn_t *conn, pai_proto_role_t role,
                    uint32_t caps, const pai_proto_transport_t *transport,
                    const pai_proto_callbacks_t *callbacks, void *user) {
  if (!conn || !transport) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(conn, 0, sizeof(*conn));
  conn->role = role;
  conn->local_caps = caps & PAI_PROTO_CAP_KNOWN;
  conn->state = PAI_PROTO_STATE_NEW;
  conn->next_request_id = 1;
  conn->next_session_id = 1;
  conn->transport = *transport;

  if (callbacks) {
    conn->callbacks = *callbacks;
  }
  conn->user = user;
  return PAI_OK;
}

void
pai_proto_conn_destroy(pai_proto_conn_t *conn) {
  if (!conn) {
    return;
  }
  conn_reset_phase(conn);
  conn->transport.close(conn->transport.ctx);
  conn->state = PAI_PROTO_STATE_CLOSED;
}

/* ------------------------------------------------------------------ */
/* sending                                                             */
/* ------------------------------------------------------------------ */

/* Send a frame without the OPEN-state gate (negotiation, auto-replies). */
static pai_status_t
conn_send_unchecked(pai_proto_conn_t *conn, const pai_proto_frame_t *frame) {
  uint8_t header[PAI_PROTO_HEADER_SIZE];
  uint32_t crc;
  pai_status_t st;

  if (frame->payload_len > PAI_PROTO_MAX_PAYLOAD) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(header, 0, sizeof(header));
  put_le32(header + 0, PAI_PROTO_MAGIC);
  header[4] = PAI_PROTO_VERSION_MAJOR;
  header[5] = PAI_PROTO_VERSION_MINOR;
  put_le16(header + 6, (uint16_t)frame->flags);
  put_le32(header + 8, frame->msg_type);
  put_le32(header + 12, frame->payload_len);
  put_le64(header + 16, frame->request_id);
  put_le64(header + 24, frame->session_id);

  crc = pai_proto_crc32_init();
  crc = pai_proto_crc32_upd(crc, header, PAI_PROTO_HEADER_SIZE - 8u);
  crc = pai_proto_crc32_upd(crc, frame->payload, frame->payload_len);
  put_le32(header + 32, pai_proto_crc32_fin(crc));

  st = conn->transport.send(conn->transport.ctx, header, sizeof(header));
  if (st != PAI_OK) {
    return st;
  }
  if (frame->payload_len > 0) {
    st = conn->transport.send(conn->transport.ctx, frame->payload,
                              frame->payload_len);
    if (st != PAI_OK) {
      return st;
    }
  }
  return PAI_OK;
}

pai_status_t
pai_proto_conn_send(pai_proto_conn_t *conn, const pai_proto_frame_t *frame) {
  if (!conn || !frame) {
    return PAI_ERR_INVALID_ARG;
  }
  if (conn->state != PAI_PROTO_STATE_OPEN) {
    return PAI_ERR_INVALID_ARG;
  }
  return conn_send_unchecked(conn, frame);
}

pai_status_t
pai_proto_conn_send_raw(pai_proto_conn_t *conn, uint32_t msg_type,
                        uint32_t flags, uint64_t request_id,
                        uint64_t session_id, const void *payload,
                        uint32_t payload_len) {
  pai_proto_frame_t frame;

  if (!conn) {
    return PAI_ERR_INVALID_ARG;
  }
  if (conn->state != PAI_PROTO_STATE_OPEN) {
    return PAI_ERR_INVALID_ARG;
  }

  memset(&frame, 0, sizeof(frame));
  frame.msg_type = msg_type;
  frame.flags = flags;
  frame.request_id = request_id;
  frame.session_id = session_id;
  frame.payload = (const uint8_t *)payload;
  frame.payload_len = payload_len;
  return conn_send_unchecked(conn, &frame);
}

uint64_t
pai_proto_conn_new_request_id(pai_proto_conn_t *conn) {
  return conn->next_request_id++;
}

/* Best-effort auto-reply carrying a structured status (whitepaper §24:
 * structured status codes). */
static void
conn_send_error_reply(pai_proto_conn_t *conn, const pai_proto_frame_t *orig,
                      pai_status_t status) {
  uint8_t pay[8u + 256u];
  uint32_t len = 0;
  pai_status_t st;

  st = pai_proto_msg_encode_error(pay, sizeof(pay), status,
                                  pai_status_str(status), &len);
  if (st != PAI_OK) {
    return;
  }
  st = conn_send_unchecked(conn, &(pai_proto_frame_t){
                                   .msg_type = PAI_PROTO_MSG_ERROR,
                                   .flags = PAI_PROTO_FLAG_REPLY,
                                   .request_id = orig->request_id,
                                   .session_id = orig->session_id,
                                   .payload = pay,
                                   .payload_len = len,
                               });
  if (st != PAI_OK) {
    PAI_LOG_WARN_(PAI_SUB_PROTO, "error reply not delivered (%s)\n",
                  pai_status_str(st));
  }
}

/* ------------------------------------------------------------------ */
/* sessions (server side)                                              */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_conn_session_open(pai_proto_conn_t *conn, uint64_t *out_id) {
  uint64_t id;

  if (!conn || !out_id) {
    return PAI_ERR_INVALID_ARG;
  }

  /* Allocate the next free id, skipping 0 (connection-level). */
  for (;;) {
    id = conn->next_session_id++;
    if (id == 0) {
      continue;
    }
    if (pai_proto_conn_session_active(conn, id)) {
      continue; /* wrap-around collision */
    }
    break;
  }

  for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS; i++) {
    if (conn->sessions[i] == 0) {
      conn->sessions[i] = id;
      *out_id = id;
      return PAI_OK;
    }
  }
  return PAI_ERR_NOMEM;
}

void
pai_proto_conn_session_close(pai_proto_conn_t *conn, uint64_t id) {
  if (!conn) {
    return;
  }
  for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS; i++) {
    if (conn->sessions[i] == id) {
      conn->sessions[i] = 0;
      return;
    }
  }
}

int
pai_proto_conn_session_active(const pai_proto_conn_t *conn, uint64_t id) {
  if (!conn || id == 0) {
    return 0;
  }
  for (uint32_t i = 0; i < PAI_PROTO_MAX_SESSIONS; i++) {
    if (conn->sessions[i] == id) {
      return 1;
    }
  }
  return 0;
}

/* ------------------------------------------------------------------ */
/* streams (inbound)                                                   */
/* ------------------------------------------------------------------ */

static int
stream_index(const pai_proto_conn_t *conn, uint64_t request_id) {
  for (uint32_t i = 0; i < PAI_PROTO_MAX_STREAMS; i++) {
    if (conn->streams[i] == request_id) {
      return (int)i;
    }
  }
  return -1;
}

static pai_status_t
stream_begin(pai_proto_conn_t *conn, uint64_t request_id) {
  if (stream_index(conn, request_id) >= 0) {
    return PAI_ERR_PROTOCOL; /* already open */
  }
  for (uint32_t i = 0; i < PAI_PROTO_MAX_STREAMS; i++) {
    if (conn->streams[i] == 0) {
      conn->streams[i] = request_id;
      conn->stream_seqs[i] = 0;
      return PAI_OK;
    }
  }
  return PAI_ERR_NOMEM; /* table full */
}

static void
stream_end(pai_proto_conn_t *conn, uint64_t request_id) {
  int i = stream_index(conn, request_id);
  if (i >= 0) {
    conn->streams[i] = 0;
    conn->stream_seqs[i] = 0;
  }
}

/*
 * Route a stream frame (TOKEN / STREAM_DATA / STREAM_CLOSE) to the
 * on_stream_* callbacks. TOKEN and STREAM_CLOSE frames are also
 * delivered to on_message; STREAM_DATA is consumed by the stream
 * callbacks only (its kind/seq travel in the payload).
 *
 * Stream lifecycle: STREAM_START opens (on_stream_begin), the terminal
 * frame (STREAM_END flag, STREAM_CLOSE or COMPLETE) closes
 * (on_stream_end). TOKEN chunks carry a synthesized ordinal; STREAM_DATA
 * carries its own seq on the wire.
 */
static pai_status_t
conn_handle_stream(pai_proto_conn_t *conn, const pai_proto_frame_t *frame) {
  int idx;
  pai_status_t st;
  int terminal = frame->flags & PAI_PROTO_FLAG_STREAM_END ? 1 : 0;

  if (frame->flags & PAI_PROTO_FLAG_STREAM_START) {
    st = stream_begin(conn, frame->request_id);
    if (st != PAI_OK) {
      return st;
    }
    if (conn->callbacks.on_stream_begin) {
      conn->callbacks.on_stream_begin(conn->user, frame->request_id,
                                      frame->session_id);
    }
  }

  idx = stream_index(conn, frame->request_id);
  if (idx < 0) {
    return PAI_ERR_PROTOCOL; /* data without an open stream */
  }

  switch (frame->msg_type) {
  case PAI_PROTO_MSG_STREAM_DATA: {
    uint8_t kind;
    uint32_t seq;
    const uint8_t *data;
    uint32_t data_len;

    st = pai_proto_msg_decode_stream_data(frame->payload, frame->payload_len,
                                          &kind, &seq, &data, &data_len);
    if (st != PAI_OK) {
      return PAI_ERR_PROTOCOL;
    }
    if (conn->callbacks.on_stream_data) {
      conn->callbacks.on_stream_data(conn->user, frame->request_id,
                                     frame->session_id, kind, seq, data,
                                     data_len);
    }
    break;
  }

  case PAI_PROTO_MSG_TOKEN: {
    const uint8_t *token;
    uint32_t token_len;

    st = pai_proto_msg_decode_token(frame->payload, frame->payload_len, &token,
                                    &token_len);
    if (st != PAI_OK) {
      return PAI_ERR_PROTOCOL;
    }
    if (conn->callbacks.on_stream_data) {
      conn->callbacks.on_stream_data(conn->user, frame->request_id,
                                     frame->session_id,
                                     PAI_PROTO_STREAM_GENERATE,
                                     conn->stream_seqs[idx], token, token_len);
    }
    if (conn->callbacks.on_message) {
      st = conn->callbacks.on_message(conn->user, frame);
      if (st != PAI_OK) {
        conn_send_error_reply(conn, frame, st);
      }
    }
    break;
  }

  case PAI_PROTO_MSG_STREAM_CLOSE:
    /* Informational terminal; no content callback. */
    terminal = 1;
    if (conn->callbacks.on_message) {
      st = conn->callbacks.on_message(conn->user, frame);
      if (st != PAI_OK) {
        conn_send_error_reply(conn, frame, st);
      }
    }
    break;

  default:
    break;
  }

  if (terminal) {
    stream_end(conn, frame->request_id);
    if (conn->callbacks.on_stream_end) {
      conn->callbacks.on_stream_end(conn->user, frame->request_id,
                                    frame->session_id);
    }
  } else if (frame->msg_type == PAI_PROTO_MSG_TOKEN) {
    conn->stream_seqs[idx] += 1u; /* synthesize the next ordinal */
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* negotiation                                                         */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_conn_start(pai_proto_conn_t *conn) {
  uint8_t pay[PAI_PROTO_U32_PAIR_SIZE];
  uint32_t len = 0;
  pai_status_t st;

  if (!conn) {
    return PAI_ERR_INVALID_ARG;
  }
  if (conn->role != PAI_PROTO_ROLE_CLIENT || conn->state != PAI_PROTO_STATE_NEW) {
    return PAI_ERR_INVALID_ARG;
  }

  st = pai_proto_msg_encode_caps(pay, sizeof(pay), conn->local_caps, &len);
  if (st != PAI_OK) {
    return st;
  }

  st = conn_send_unchecked(conn, &(pai_proto_frame_t){
                                   .msg_type = PAI_PROTO_MSG_HELLO,
                                   .request_id = conn->next_request_id++,
                                   .payload = pay,
                                   .payload_len = len,
                               });
  if (st == PAI_OK) {
    conn->state = PAI_PROTO_STATE_NEGOTIATING;
  }
  return st;
}

static pai_status_t
conn_send_negotiation(pai_proto_conn_t *conn, uint32_t msg_type,
                      uint32_t value, uint64_t request_id) {
  uint8_t pay[PAI_PROTO_U32_PAIR_SIZE];
  uint32_t len = 0;
  pai_status_t st;

  st = pai_proto_msg_encode_u32(pay, sizeof(pay), value, &len);
  if (st != PAI_OK) {
    return st;
  }
  return conn_send_unchecked(conn, &(pai_proto_frame_t){
                                   .msg_type = msg_type,
                                   .flags = PAI_PROTO_FLAG_REPLY,
                                   .request_id = request_id,
                                   .payload = pay,
                                   .payload_len = len,
                               });
}

static pai_status_t
conn_handle_hello_server(pai_proto_conn_t *conn,
                         const pai_proto_frame_t *frame) {
  uint32_t remote = 0;
  uint32_t server_caps;
  uint32_t negotiated;

  if (pai_proto_msg_decode_caps(frame->payload, frame->payload_len, &remote) !=
      PAI_OK) {
    return PAI_ERR_PROTOCOL;
  }

  if ((frame->version >> 8) != PAI_PROTO_VERSION_MAJOR) {
    PAI_LOG_WARN_(PAI_SUB_PROTO,
                  "version mismatch: peer %d.%d, local %d.%d\n",
                  frame->version >> 8, frame->version & 0xFFu,
                  PAI_PROTO_VERSION_MAJOR, PAI_PROTO_VERSION_MINOR);
    (void)conn_send_negotiation(conn, PAI_PROTO_MSG_HELLO_NACK,
                                PAI_PROTO_NACK_VERSION, frame->request_id);
    conn_set_closed(conn, PAI_PROTO_CLOSE_PROTOCOL);
    return PAI_OK;
  }

  server_caps = conn->callbacks.on_hello
                    ? conn->callbacks.on_hello(conn->user, remote)
                    : conn->local_caps;
  if (server_caps == 0) {
    PAI_LOG_INFO_(PAI_SUB_PROTO, "negotiation refused: caps 0x%08x\n", remote);
    (void)conn_send_negotiation(conn, PAI_PROTO_MSG_HELLO_NACK,
                                PAI_PROTO_NACK_CAPS, frame->request_id);
    conn_set_closed(conn, PAI_PROTO_CLOSE_NORMAL);
    return PAI_OK;
  }

  negotiated = server_caps & remote & PAI_PROTO_CAP_KNOWN;
  conn->local_caps = server_caps & PAI_PROTO_CAP_KNOWN;
  conn->remote_caps = remote & PAI_PROTO_CAP_KNOWN;
  conn->negotiated_caps = negotiated;
  conn->state = PAI_PROTO_STATE_OPEN;

  PAI_LOG_INFO_(PAI_SUB_PROTO, "negotiated caps 0x%08x (server 0x%08x, "
                               "client 0x%08x)\n",
                negotiated, conn->local_caps, conn->remote_caps);
  return conn_send_negotiation(conn, PAI_PROTO_MSG_HELLO_ACK, negotiated,
                               frame->request_id);
}

static pai_status_t
conn_handle_ack_client(pai_proto_conn_t *conn,
                       const pai_proto_frame_t *frame) {
  uint32_t negotiated = 0;

  if ((frame->version >> 8) != PAI_PROTO_VERSION_MAJOR) {
    return PAI_ERR_PROTOCOL;
  }
  if (pai_proto_msg_decode_caps(frame->payload, frame->payload_len,
                                &negotiated) != PAI_OK) {
    return PAI_ERR_PROTOCOL;
  }

  conn->remote_caps = negotiated;
  conn->negotiated_caps = negotiated & PAI_PROTO_CAP_KNOWN;
  conn->state = PAI_PROTO_STATE_OPEN;
  PAI_LOG_INFO_(PAI_SUB_PROTO, "negotiated caps 0x%08x\n",
                conn->negotiated_caps);
  return PAI_OK;
}

static pai_status_t
conn_handle_nack_client(pai_proto_conn_t *conn,
                        const pai_proto_frame_t *frame) {
  uint32_t reason = 0;

  (void)pai_proto_msg_decode_u32(frame->payload, frame->payload_len, &reason);
  conn->nack_reason = reason;
  conn_set_closed(conn, PAI_PROTO_CLOSE_NORMAL);
  return PAI_OK;
}

static pai_status_t
conn_handle_preopen(pai_proto_conn_t *conn, const pai_proto_frame_t *frame) {
  if (conn->role == PAI_PROTO_ROLE_SERVER) {
    if (frame->msg_type != PAI_PROTO_MSG_HELLO) {
      return PAI_ERR_PROTOCOL; /* first message must be HELLO */
    }
    return conn_handle_hello_server(conn, frame);
  }

  switch (frame->msg_type) {
  case PAI_PROTO_MSG_HELLO_ACK:
    return conn_handle_ack_client(conn, frame);
  case PAI_PROTO_MSG_HELLO_NACK:
    return conn_handle_nack_client(conn, frame);
  default:
    return PAI_ERR_PROTOCOL;
  }
}

/* ------------------------------------------------------------------ */
/* open-state dispatch                                                 */
/* ------------------------------------------------------------------ */

static pai_status_t
conn_handle_open(pai_proto_conn_t *conn, const pai_proto_frame_t *frame) {
  pai_status_t st;

  if ((frame->version >> 8) != PAI_PROTO_VERSION_MAJOR) {
    return PAI_ERR_PROTOCOL;
  }

  switch (frame->msg_type) {
  case PAI_PROTO_MSG_HELLO:
  case PAI_PROTO_MSG_HELLO_ACK:
  case PAI_PROTO_MSG_HELLO_NACK:
    return PAI_ERR_PROTOCOL; /* misplaced negotiation */

  case PAI_PROTO_MSG_PING: {
    /* Keepalive: core answers PONG automatically. */
    pai_proto_frame_t pong;
    memset(&pong, 0, sizeof(pong));
    pong.msg_type = PAI_PROTO_MSG_PONG;
    pong.flags = PAI_PROTO_FLAG_REPLY;
    pong.request_id = frame->request_id;
    pong.session_id = frame->session_id;
    st = conn_send_unchecked(conn, &pong);
    return st == PAI_OK ? PAI_OK : st;
  }

  case PAI_PROTO_MSG_CLOSE: {
    uint32_t reason = PAI_PROTO_CLOSE_NORMAL;
    if (frame->payload_len > 0) {
      (void)pai_proto_msg_decode_u32(frame->payload, frame->payload_len,
                                     &reason);
    }
    PAI_LOG_INFO_(PAI_SUB_PROTO, "peer closed (reason %u)\n", reason);
    conn_set_closed(conn, reason);
    return PAI_OK;
  }

  default:
    break;
  }

  if (frame->msg_type == PAI_PROTO_MSG_STREAM_DATA ||
      frame->msg_type == PAI_PROTO_MSG_TOKEN ||
      frame->msg_type == PAI_PROTO_MSG_STREAM_CLOSE) {
    return conn_handle_stream(conn, frame);
  }

  /* COMPLETE terminates the GENERATE stream (defensive: even without a
   * trailing STREAM_END flag on the last TOKEN). */
  if (frame->msg_type == PAI_PROTO_MSG_COMPLETE) {
    stream_end(conn, frame->request_id);
  }

  if (conn->callbacks.on_message) {
    st = conn->callbacks.on_message(conn->user, frame);
    if (st != PAI_OK && frame->request_id != 0 &&
        frame->msg_type != PAI_PROTO_MSG_ERROR &&
        frame->msg_type != PAI_PROTO_MSG_PONG) {
      conn_send_error_reply(conn, frame, st);
    }
  }
  return PAI_OK;
}

/* ------------------------------------------------------------------ */
/* receive pump                                                        */
/* ------------------------------------------------------------------ */

pai_status_t
pai_proto_conn_poll(pai_proto_conn_t *conn, uint32_t *out_frames) {
  pai_status_t result = PAI_OK;
  uint32_t frames = 0;

  if (!conn) {
    return PAI_ERR_INVALID_ARG;
  }

  while (conn->state != PAI_PROTO_STATE_CLOSED) {
    pai_status_t st;

    /* Phase 1: complete the header. */
    if (conn->header_filled < PAI_PROTO_HEADER_SIZE) {
      uint32_t got = 0;

      st = conn->transport.recv(conn->transport.ctx,
                                conn->header_buf + conn->header_filled,
                                PAI_PROTO_HEADER_SIZE - conn->header_filled,
                                &got);
      if (st == PAI_ERR_IO && got == 0) {
        /* EOF without CLOSE: peer went away. */
        conn_set_closed(conn, PAI_PROTO_CLOSE_PEER_GONE);
        break;
      }
      if (st != PAI_OK) {
        conn_set_closed(conn, PAI_PROTO_CLOSE_PEER_GONE);
        result = st;
        break;
      }
      conn->header_filled += got;
      if (conn->header_filled < PAI_PROTO_HEADER_SIZE) {
        break; /* wait for more bytes */
      }

      st = pai_proto_frame_decode(conn->header_buf, &conn->pending);
      if (st != PAI_OK) {
        PAI_LOG_ERROR_(PAI_SUB_PROTO, "malformed frame header\n");
        conn_set_closed(conn, PAI_PROTO_CLOSE_PROTOCOL);
        result = PAI_ERR_PROTOCOL;
        break;
      }
      if (conn->pending.payload_len > 0) {
        conn->payload_buf =
            (uint8_t *)malloc(conn->pending.payload_len);
        if (!conn->payload_buf) {
          conn_set_closed(conn, PAI_PROTO_CLOSE_APP);
          result = PAI_ERR_NOMEM;
          break;
        }
      }
      conn->payload_filled = 0;
    }

    /* Phase 2: complete the payload. */
    if (conn->pending.payload_len > conn->payload_filled) {
      uint32_t got = 0;

      st = conn->transport.recv(conn->transport.ctx,
                                conn->payload_buf + conn->payload_filled,
                                conn->pending.payload_len - conn->payload_filled,
                                &got);
      if (st == PAI_ERR_IO) {
        PAI_LOG_ERROR_(PAI_SUB_PROTO, "truncated frame payload\n");
        conn_set_closed(conn, PAI_PROTO_CLOSE_PROTOCOL);
        result = PAI_ERR_PROTOCOL;
        break;
      }
      if (st != PAI_OK) {
        conn_set_closed(conn, PAI_PROTO_CLOSE_PEER_GONE);
        result = st;
        break;
      }
      conn->payload_filled += got;
      if (conn->payload_filled < conn->pending.payload_len) {
        break; /* wait for more bytes */
      }

      st = pai_proto_frame_validate(conn->header_buf, conn->payload_buf);
      if (st != PAI_OK) {
        PAI_LOG_ERROR_(PAI_SUB_PROTO, "frame CRC mismatch\n");
        conn_set_closed(conn, PAI_PROTO_CLOSE_PROTOCOL);
        result = PAI_ERR_PROTOCOL;
        break;
      }
    }

    /* Phase 3: dispatch the complete frame. */
    conn->pending.payload = conn->payload_buf;
    st = conn->state == PAI_PROTO_STATE_OPEN
             ? conn_handle_open(conn, &conn->pending)
             : conn_handle_preopen(conn, &conn->pending);
    if (st != PAI_OK) {
      PAI_LOG_ERROR_(PAI_SUB_PROTO, "frame rejected (%s)\n",
                     pai_status_str(st));
      conn_set_closed(conn, PAI_PROTO_CLOSE_PROTOCOL);
      result = PAI_ERR_PROTOCOL;
      break;
    }

    conn_reset_phase(conn);
    frames++;
    if (frames >= PAI_PROTO_POLL_BATCH) {
      break;
    }
  }

  if (out_frames) {
    *out_frames = frames;
  }
  return result;
}

/* ------------------------------------------------------------------ */
/* accessors                                                           */
/* ------------------------------------------------------------------ */

uint32_t
pai_proto_conn_negotiated_caps(const pai_proto_conn_t *conn) {
  return conn->negotiated_caps;
}

pai_proto_state_t
pai_proto_conn_state(const pai_proto_conn_t *conn) {
  return (pai_proto_state_t)conn->state;
}

const char *
pai_proto_conn_state_name(pai_proto_state_t state) {
  switch (state) {
  case PAI_PROTO_STATE_NEW:         return "new";
  case PAI_PROTO_STATE_NEGOTIATING: return "negotiating";
  case PAI_PROTO_STATE_OPEN:        return "open";
  case PAI_PROTO_STATE_CLOSED:      return "closed";
  default:                          return "unknown";
  }
}
