/*
 * ProsperoAI — Prospero Protocol (whitepaper §24/§25)
 *
 * Custom binary protocol for desktop <-> PS5 communication. It is a
 * distinct wire protocol with its own ABI and semantics, influenced by
 * the proven design concepts of MemDBG (compact binary frames, request
 * identifiers, connection roles) while remaining fully independent of
 * it. MemDBG itself is untouched.
 *
 * v0 features implemented here:
 *   - compact binary frames (40-byte header + payload, CRC-32 integrity)
 *   - request identifiers with pipelining (many in-flight requests)
 *   - connection roles (client/server) and persistent connections
 *   - version negotiation (HELLO/HELLO_ACK/HELLO_NACK)
 *   - capability negotiation (offered caps are intersected)
 *   - sessions (server-allocated session ids)
 *   - structured status codes (ERROR frames carry pai_status_t)
 *   - native async streams (whitepaper §24 example:
 *     GENERATE -> ACCEPTED -> TOKEN* -> COMPLETE, plus a generic
 *     STREAM_DATA carrier for telemetry/profiler/transfer/logs)
 *   - compression is reserved (flag defined, rejected by v0 receivers)
 *
 * The protocol is transport-independent (§25): connections operate over
 * a pai_proto_transport_t (send/pull-recv). A TCP transport and a
 * local/internal transport can be added without touching this layer; an
 * in-memory pipe transport (transport_pipe.c) ships for host tests and
 * loopback development.
 *
 * All integers on the wire are little-endian (both target platforms are
 * LE). Frames are not self-delimiting beyond their length fields: the
 * transport must provide byte-stream semantics.
 */

#ifndef PAI_PROTOCOL_H
#define PAI_PROTOCOL_H

#include <pai/error.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Protocol version                                                    */
/* ------------------------------------------------------------------ */

#define PAI_PROTO_VERSION_MAJOR 0
#define PAI_PROTO_VERSION_MINOR 1
#define PAI_PROTO_VERSION \
  ((PAI_PROTO_VERSION_MAJOR << 8) | PAI_PROTO_VERSION_MINOR)

/* ------------------------------------------------------------------ */
/* Wire format                                                         */
/* ------------------------------------------------------------------ */

/* Frame magic: ASCII "PAIP" as a little-endian u32 (bytes 50 41 49 50). */
#define PAI_PROTO_MAGIC 0x50494150u

/* Fixed header size in bytes; the payload starts at offset 40. */
#define PAI_PROTO_HEADER_SIZE 40u

/* Hard cap on payload length accepted by v0 receivers (1 MiB). */
#define PAI_PROTO_MAX_PAYLOAD (1u << 20)

/*
 * Frame layout (little-endian):
 *
 *   [0]    u32 magic            0x50494150 ("PAIP")
 *   [4]    u8  version_major    sender's protocol major
 *   [5]    u8  version_minor    sender's protocol minor
 *   [6]    u16 flags            pai_proto_flag_*
 *   [8]    u32 message_type     pai_proto_msg
 *   [12]   u32 payload_len      bytes following the header
 *   [16]   u64 request_id       0 = none
 *   [24]   u64 session_id       0 = connection-level
 *   [32]   u32 crc32            CRC-32 (ISO-HDLC) over bytes [0,32) and
 *                               the payload; 0 when payload_len == 0
 *                               (still validated)
 *   [36]   u32 reserved         must be 0
 *   [40]   payload              payload_len bytes
 */

/* ------------------------------------------------------------------ */
/* Frame flags                                                         */
/* ------------------------------------------------------------------ */

#define PAI_PROTO_FLAG_REPLY        (1u << 0) /* reply to a prior request   */
#define PAI_PROTO_FLAG_STREAM_START (1u << 1) /* first frame of a stream    */
#define PAI_PROTO_FLAG_STREAM_END   (1u << 2) /* final frame of a stream    */
#define PAI_PROTO_FLAG_COMPRESSED   (1u << 3) /* reserved; rejected by v0   */

/* ------------------------------------------------------------------ */
/* Message types                                                       */
/* ------------------------------------------------------------------ */

enum pai_proto_msg {
  /* Connection lifecycle / negotiation */
  PAI_PROTO_MSG_HELLO      = 0x0001, /* client -> server: offer          */
  PAI_PROTO_MSG_HELLO_ACK  = 0x0002, /* server -> client: accepted       */
  PAI_PROTO_MSG_HELLO_NACK = 0x0003, /* server -> client: rejected       */
  PAI_PROTO_MSG_PING       = 0x0004, /* keepalive; core auto-replies     */
  PAI_PROTO_MSG_PONG       = 0x0005, /* reply to PING                    */
  PAI_PROTO_MSG_ERROR      = 0x0006, /* structured status carrier        */
  PAI_PROTO_MSG_CLOSE      = 0x0007, /* graceful shutdown                */

  /* Sessions */
  PAI_PROTO_MSG_SESSION_CREATE  = 0x0101,
  PAI_PROTO_MSG_SESSION_CREATED = 0x0102,
  PAI_PROTO_MSG_SESSION_CLOSE   = 0x0103,
  PAI_PROTO_MSG_SESSION_CLOSED  = 0x0104,

  /* Inference — the whitepaper §24 example:
   *   GENERATE #r -> ACCEPTED #r -> TOKEN #r* -> COMPLETE #r           */
  PAI_PROTO_MSG_GENERATE = 0x0201,
  PAI_PROTO_MSG_ACCEPTED = 0x0202,
  PAI_PROTO_MSG_TOKEN    = 0x0203,
  PAI_PROTO_MSG_COMPLETE = 0x0204,

  /* Generic async stream carrier (telemetry, profiler output,
   * autotuning, transfer progress, logs, long compilations). */
  PAI_PROTO_MSG_STREAM_DATA  = 0x0301,
  PAI_PROTO_MSG_STREAM_CLOSE = 0x0302,
};

/* Stream kinds carried by STREAM_DATA payloads. TOKEN frames are
 * reported with PAI_PROTO_STREAM_GENERATE (synthesized, not on wire). */
enum pai_proto_stream_kind {
  PAI_PROTO_STREAM_TELEMETRY = 0,
  PAI_PROTO_STREAM_PROFILER,
  PAI_PROTO_STREAM_TRANSFER,
  PAI_PROTO_STREAM_LOG,
  PAI_PROTO_STREAM_COMPILE,
  PAI_PROTO_STREAM_GENERATE,
};

/* HELLO_NACK reasons (payload u32). */
enum pai_proto_nack {
  PAI_PROTO_NACK_VERSION = 1,
  PAI_PROTO_NACK_CAPS    = 2,
};

/* CLOSE / STREAM_CLOSE reasons (payload u32). */
enum pai_proto_close {
  PAI_PROTO_CLOSE_NORMAL   = 0, /* clean shutdown                        */
  PAI_PROTO_CLOSE_PROTOCOL = 1, /* protocol violation, peer misbehaving  */
  PAI_PROTO_CLOSE_APP      = 2, /* application decided to leave          */
  PAI_PROTO_CLOSE_PEER_GONE = 3, /* transport EOF without CLOSE          */
};

/* ------------------------------------------------------------------ */
/* Capabilities                                                        */
/* ------------------------------------------------------------------ */

#define PAI_PROTO_CAP_STREAMS     (1u << 0) /* async streams supported      */
#define PAI_PROTO_CAP_SESSIONS    (1u << 1) /* multiple sessions supported  */
#define PAI_PROTO_CAP_GENERATE    (1u << 2) /* generate/stream tokens       */
#define PAI_PROTO_CAP_EMBED       (1u << 3) /* embeddings endpoint          */
#define PAI_PROTO_CAP_TRANSFER    (1u << 4) /* model transfer engine        */
#define PAI_PROTO_CAP_TELEMETRY   (1u << 5) /* telemetry streams            */
#define PAI_PROTO_CAP_COMPRESSION (1u << 6) /* reserved in v0               */

#define PAI_PROTO_CAP_KNOWN                                                    \
  (PAI_PROTO_CAP_STREAMS | PAI_PROTO_CAP_SESSIONS | PAI_PROTO_CAP_GENERATE |   \
   PAI_PROTO_CAP_EMBED | PAI_PROTO_CAP_TRANSFER | PAI_PROTO_CAP_TELEMETRY |    \
   PAI_PROTO_CAP_COMPRESSION)

/* ------------------------------------------------------------------ */
/* Host-side frame representation                                      */
/* ------------------------------------------------------------------ */

typedef struct pai_proto_frame {
  uint16_t       version;     /* sender protocol version (packed)      */
  uint32_t       msg_type;
  uint32_t       flags;
  uint64_t       request_id;
  uint64_t       session_id;
  const uint8_t *payload;     /* valid until the next poll/frame       */
  uint32_t       payload_len;
} pai_proto_frame_t;

/* HELLO / HELLO_ACK / HELLO_NACK / CLOSE payload size: u32 + u32. */
#define PAI_PROTO_U32_PAIR_SIZE 8u

/* ------------------------------------------------------------------ */
/* CRC-32 (ISO-HDLC, zlib polynomial 0xEDB88320)                       */
/* ------------------------------------------------------------------ */

/* One-shot CRC-32. */
uint32_t pai_proto_crc32(const void *data, uint32_t nbytes);

/* Running variant for chunked hashing (e.g. header + payload split). */
uint32_t pai_proto_crc32_init(void);
uint32_t pai_proto_crc32_upd(uint32_t running, const void *data,
                             uint32_t nbytes);
uint32_t pai_proto_crc32_fin(uint32_t running);

/* ------------------------------------------------------------------ */
/* Frame codec                                                         */
/* ------------------------------------------------------------------ */

/*
 * Serialize a full frame (header + payload) into `out` with capacity
 * `cap`. Fills magic/version/crc32/reserved. Returns PAI_OK and sets
 * *out_nbytes, or PAI_ERR_INVALID_ARG when the frame is malformed.
 */
pai_status_t pai_proto_frame_encode(const pai_proto_frame_t *frame,
                                    uint8_t *out, uint32_t cap,
                                    uint32_t *out_nbytes);

/*
 * Parse a header from `data` (>= PAI_PROTO_HEADER_SIZE bytes).
 * Validates magic, length cap and reserved field. Version compatibility
 * is the connection's decision (it must reach the negotiator for HELLO
 * frames), so the parsed version is returned rather than rejected here.
 * PAI_ERR_PROTOCOL on malformed input.
 */
pai_status_t pai_proto_frame_decode(const uint8_t *data,
                                    pai_proto_frame_t *out);

/*
 * Validate integrity of a complete frame: `header` (>= HEADER_SIZE
 * bytes) plus `payload` (payload_len bytes; may be NULL when 0).
 * Recomputes and compares the CRC-32. PAI_ERR_PROTOCOL on mismatch.
 */
pai_status_t pai_proto_frame_validate(const uint8_t *header,
                                      const uint8_t *payload);

/* Wire size of a frame with the given payload length. */
uint32_t pai_proto_frame_size(uint32_t payload_len);

/* ------------------------------------------------------------------ */
/* Message payload codecs (v0 formats)                                 */
/* ------------------------------------------------------------------ */

/* HELLO / HELLO_ACK: { u32 caps; u32 reserved }. */
pai_status_t pai_proto_msg_encode_caps(uint8_t *out, uint32_t cap, uint32_t caps,
                                       uint32_t *out_len);
pai_status_t pai_proto_msg_decode_caps(const uint8_t *payload, uint32_t len,
                                       uint32_t *out_caps);

/* HELLO_NACK / CLOSE / STREAM_CLOSE: { u32 reason; u32 reserved }. */
pai_status_t pai_proto_msg_encode_u32(uint8_t *out, uint32_t cap, uint32_t value,
                                      uint32_t *out_len);
pai_status_t pai_proto_msg_decode_u32(const uint8_t *payload, uint32_t len,
                                      uint32_t *out_value);

/*
 * ERROR: { i32 status; u32 msg_len; u8 msg[msg_len] }.
 * `msg` may be NULL. msg_len is capped at 255 in v0.
 */
pai_status_t pai_proto_msg_encode_error(uint8_t *out, uint32_t cap,
                                        pai_status_t status, const char *msg,
                                        uint32_t *out_len);
pai_status_t pai_proto_msg_decode_error(const uint8_t *payload, uint32_t len,
                                        pai_status_t *out_status,
                                        char *out_msg, uint32_t msg_cap);

/* GENERATE: { u16 prompt_len; u8 prompt[prompt_len] } (UTF-8). */
pai_status_t pai_proto_msg_encode_generate(uint8_t *out, uint32_t cap,
                                           const char *prompt,
                                           uint32_t *out_len);
pai_status_t pai_proto_msg_decode_generate(const uint8_t *payload, uint32_t len,
                                           const char **out_prompt,
                                           uint32_t *out_prompt_len);

/* TOKEN: { u16 token_len; u8 token[token_len] }. */
pai_status_t pai_proto_msg_encode_token(uint8_t *out, uint32_t cap,
                                        const uint8_t *token, uint32_t token_len,
                                        uint32_t *out_len);
pai_status_t pai_proto_msg_decode_token(const uint8_t *payload, uint32_t len,
                                        const uint8_t **out_token,
                                        uint32_t *out_token_len);

/*
 * STREAM_DATA: { u8 kind; u8 reserved[3]; u32 seq; u32 data_len;
 *                u8 data[data_len] }.
 */
pai_status_t pai_proto_msg_encode_stream_data(uint8_t *out, uint32_t cap,
                                              uint8_t kind, uint32_t seq,
                                              const uint8_t *data,
                                              uint32_t data_len,
                                              uint32_t *out_len);
pai_status_t pai_proto_msg_decode_stream_data(const uint8_t *payload,
                                              uint32_t len, uint8_t *out_kind,
                                              uint32_t *out_seq,
                                              const uint8_t **out_data,
                                              uint32_t *out_data_len);

/* Stable names for diagnostics. */
const char *pai_proto_msg_name(uint32_t msg_type);

/* ------------------------------------------------------------------ */
/* Transport abstraction (§25)                                         */
/* ------------------------------------------------------------------ */

typedef struct pai_proto_transport {
  void *ctx;
  /*
   * Push `nbytes`. PAI_ERR_IO when the transport is closed.
   */
  pai_status_t (*send)(void *ctx, const void *data, uint32_t nbytes);
  /*
   * Pull up to `nbytes` into `data`; sets *out_read. PAI_ERR_IO with
   * *out_read == 0 signals EOF (peer closed). May return fewer bytes
   * than requested; the connection loops.
   */
  pai_status_t (*recv)(void *ctx, void *data, uint32_t nbytes,
                       uint32_t *out_read);
  /* Tear down this endpoint (idempotent). */
  void (*close)(void *ctx);
} pai_proto_transport_t;

/*
 * In-memory pipe pair for host tests / loopback: `a`'s sends reach
 * `b`'s recv and vice versa. Both directions are bounded only by heap.
 */
typedef struct pai_proto_pipe_pair pai_proto_pipe_pair_t;

pai_status_t pai_proto_pipe_pair_create(pai_proto_pipe_pair_t **out_pair);
void pai_proto_pipe_pair_destroy(pai_proto_pipe_pair_t *pair);

/* Endpoint transports; `which` 0 = side A, 1 = side B. */
void pai_proto_pipe_endpoint(const pai_proto_pipe_pair_t *pair, int which,
                             pai_proto_transport_t *out);

/*
 * TCP transports (§25 — the desktop <-> PS5 link). `recv` is
 * time-bounded (SO_RCVTIMEO, 100 ms default) so poll() keeps the pipe
 * semantics: PAI_OK with *out_read == 0 when no data is available yet,
 * PAI_ERR_IO with *out_read == 0 on EOF. `send` blocks until every
 * byte is accepted. `close` is idempotent.
 */

/* Client transport: connect to `host` (IP or name) on `port`. */
pai_status_t pai_proto_tcp_connect(const char *host, uint16_t port,
                                   pai_proto_transport_t *out);

/* Server listener. */
typedef struct pai_proto_tcp_listener pai_proto_tcp_listener_t;

/* Bind + listen on `host` (NULL = any) `port` (0 = ephemeral). */
pai_status_t pai_proto_tcp_listen(pai_proto_tcp_listener_t **out,
                                  const char *host, uint16_t port);

/* The port actually bound (useful with port 0). */
pai_status_t pai_proto_tcp_listener_port(const pai_proto_tcp_listener_t *l,
                                         uint16_t *out_port);

/* Accept one connection (blocking); returns a server-side transport. */
pai_status_t pai_proto_tcp_accept(pai_proto_tcp_listener_t *l,
                                  pai_proto_transport_t *out);

void pai_proto_tcp_listener_destroy(pai_proto_tcp_listener_t *l);

/*
 * Free a TCP transport endpoint. `close` alone only closes the socket
 * (it survives the connection layer's double-close); the endpoint
 * itself is heap-owned and must be released with this call after the
 * connection using it has been destroyed. Idempotent.
 */
void pai_proto_tcp_transport_destroy(pai_proto_transport_t *t);

/*
 * v0 note: pai_proto_tcp_connect uses a blocking connect() — on
 * unreachable hosts it can take the OS default timeout (tens of
 * seconds). Loopback and LAN peers return immediately.
 */

/*
 * Convenience health check (desktop tooling): connect to host:port,
 * negotiate, then send `count` PINGs, waiting up to `timeout_ms` per
 * ping for the core-auto-replied PONG. Per-ping outcome in
 * results[].rtt_ns / results[].lost; the negotiated capability mask is
 * returned through *out_caps (may be NULL). Returns PAI_ERR_IO when
 * the connection cannot be established, PAI_ERR_CAPABILITY when the
 * server refuses negotiation, PAI_ERR_TIMEOUT when negotiation itself
 * times out, PAI_OK otherwise (individual ping losses are reported via
 * results, not the status). `timeout_ms` 0 selects a 2000 ms default.
 */
typedef struct pai_proto_ping_result {
  uint64_t rtt_ns; /* round-trip latency; 0 when lost              */
  int lost;        /* 1 = no PONG within the per-ping timeout      */
} pai_proto_ping_result_t;

pai_status_t pai_proto_ping(const char *host, uint16_t port, uint32_t count,
                            pai_proto_ping_result_t *results,
                            uint32_t *out_caps, uint64_t timeout_ms);

/* ------------------------------------------------------------------ */
/* Connection                                                          */
/* ------------------------------------------------------------------ */

typedef enum pai_proto_role {
  PAI_PROTO_ROLE_CLIENT = 0,
  PAI_PROTO_ROLE_SERVER = 1,
} pai_proto_role_t;

typedef enum pai_proto_state {
  PAI_PROTO_STATE_NEW = 0,       /* not yet negotiated                 */
  PAI_PROTO_STATE_NEGOTIATING,   /* HELLO sent (client) / awaiting      */
  PAI_PROTO_STATE_OPEN,          /* ready to exchange messages          */
  PAI_PROTO_STATE_CLOSED,        /* terminal                            */
} pai_proto_state_t;

/* Fixed-size tracking tables (v0). */
#define PAI_PROTO_MAX_STREAMS 16
#define PAI_PROTO_MAX_SESSIONS 16

typedef struct pai_proto_callbacks {
  /*
   * Server side: a HELLO arrived. Return the server's capability mask;
   * the core intersects it with the client's offer. Returning 0 makes
   * the core refuse the connection (HELLO_NACK, reason CAPS).
   */
  uint32_t (*on_hello)(void *user, uint32_t remote_caps);

  /*
   * Every other received message, including stream terminal frames
   * (COMPLETE, STREAM_CLOSE). Returning a non-OK status makes the core
   * auto-reply an ERROR frame with that status (when the frame is not
   * itself an ERROR/CLOSE/PONG).
   */
  pai_status_t (*on_message)(void *user, const pai_proto_frame_t *frame);

  /*
   * Incoming async stream events (inbound direction). `kind` is the
   * stream kind (PAI_PROTO_STREAM_*; GENERATE for TOKEN frames) and
   * `seq` the per-stream ordinal of this chunk. `data`/`len` is the
   * content bytes (token text for TOKEN, the data field for
   * STREAM_DATA) and is only valid for the duration of the callback.
   */
  void (*on_stream_begin)(void *user, uint64_t request_id,
                          uint64_t session_id);
  void (*on_stream_data)(void *user, uint64_t request_id,
                         uint64_t session_id, uint8_t kind, uint32_t seq,
                         const uint8_t *data, uint32_t len);
  void (*on_stream_end)(void *user, uint64_t request_id,
                        uint64_t session_id);

  /* Peer closed the connection (CLOSE frame or transport EOF). */
  void (*on_close)(void *user, uint32_t reason);
} pai_proto_callbacks_t;

typedef struct pai_proto_conn {
  /* --- public, read-only ----------------------------------------- */
  pai_proto_role_t    role;
  uint32_t            local_caps;
  uint32_t            remote_caps;
  uint32_t            negotiated_caps;
  uint32_t            state;
  uint32_t            nack_reason; /* PAI_PROTO_NACK_* after a refusal */

  /* --- internal --------------------------------------------------- */
  pai_proto_transport_t transport;
  pai_proto_callbacks_t callbacks;
  void                 *user;

  uint64_t  next_request_id;  /* client-side allocator                */
  uint64_t  next_session_id;  /* server-side allocator                */

  uint8_t   header_buf[PAI_PROTO_HEADER_SIZE];
  uint32_t  header_filled;
  uint8_t  *payload_buf;      /* owned; NULL until payload phase      */
  uint32_t  payload_filled;
  pai_proto_frame_t pending;  /* header awaiting its payload          */

  uint64_t  streams[PAI_PROTO_MAX_STREAMS];      /* open request ids  */
  uint32_t  stream_seqs[PAI_PROTO_MAX_STREAMS];  /* per-stream ordinal */
  uint64_t  sessions[PAI_PROTO_MAX_SESSIONS];    /* open session ids  */
} pai_proto_conn_t;

/*
 * Initialize a connection over `transport`. `caps` are the local
 * capability bits offered (client) or returned by on_hello (server).
 * A NULL callback entry is treated as a no-op.
 */
pai_status_t pai_proto_conn_init(pai_proto_conn_t *conn,
                                 pai_proto_role_t role, uint32_t caps,
                                 const pai_proto_transport_t *transport,
                                 const pai_proto_callbacks_t *callbacks,
                                 void *user);
void pai_proto_conn_destroy(pai_proto_conn_t *conn);

/*
 * Client: begin negotiation by sending HELLO. Call poll() afterwards;
 * the connection opens once HELLO_ACK arrives. Refused connections land
 * in CLOSED with nack_reason set.
 */
pai_status_t pai_proto_conn_start(pai_proto_conn_t *conn);

/*
 * Server: call poll() to receive and answer HELLO automatically
 * (on_hello supplies the server caps). Nothing else is required.
 */

/*
 * Pull available transport bytes and dispatch complete frames
 * (handling HELLO, PING/PONG and CLOSE in the core). Sets *out_frames
 * to the number of frames processed. PAI_OK even when zero bytes were
 * available; PAI_ERR_PROTOCOL (connection closed) on wire violations.
 */
pai_status_t pai_proto_conn_poll(pai_proto_conn_t *conn,
                                 uint32_t *out_frames);

/* Send a frame (any message type); PAI_ERR_INVALID_ARG before the
 * connection is OPEN. */
pai_status_t pai_proto_conn_send(pai_proto_conn_t *conn,
                                 const pai_proto_frame_t *frame);
pai_status_t pai_proto_conn_send_raw(pai_proto_conn_t *conn,
                                     uint32_t msg_type, uint32_t flags,
                                     uint64_t request_id, uint64_t session_id,
                                     const void *payload, uint32_t payload_len);

/* Client request-id allocator. */
uint64_t pai_proto_conn_new_request_id(pai_proto_conn_t *conn);

/* Server session registry. */
pai_status_t pai_proto_conn_session_open(pai_proto_conn_t *conn,
                                         uint64_t *out_id);
void pai_proto_conn_session_close(pai_proto_conn_t *conn, uint64_t id);
int  pai_proto_conn_session_active(const pai_proto_conn_t *conn, uint64_t id);

uint32_t       pai_proto_conn_negotiated_caps(const pai_proto_conn_t *conn);
pai_proto_state_t pai_proto_conn_state(const pai_proto_conn_t *conn);
const char    *pai_proto_conn_state_name(pai_proto_state_t state);

#ifdef __cplusplus
}
#endif

#endif /* PAI_PROTOCOL_H */
