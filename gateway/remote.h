/*
 * ProsperoAI — gateway remote bridge (whitepaper §24/§26)
 *
 * The gateway can serve models that run on a remote payload (the PS5
 * console) instead of local .pai files: an entry registered with
 * pai_gw_add_remote points at a host:port speaking the Prospero
 * Protocol. Generation is bridged per request: connect, negotiate,
 * GENERATE(prompt), then relay each TOKEN chunk back through a
 * callback as it arrives.
 *
 * v0 scope: GENERATE carries the prompt plus an optional sampler
 * trailer (§24/§25); zero sampler values fall back to the payload's
 * defaults. Embeddings are bridged with the one-shot EMBED/EMBEDDING
 * exchange guarded by PAI_PROTO_CAP_EMBED.
 */

#ifndef PAI_GATEWAY_REMOTE_H
#define PAI_GATEWAY_REMOTE_H

#include <pai/error.h>
#include <protocol/protocol.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * One-shot remote generation over the Prospero Protocol (v0):
 * connect to host:port, negotiate, send GENERATE(prompt[, sampler]),
 * and relay each TOKEN through `on_token` (synchronous; the token
 * text is only valid for the duration of the call). Finishes when
 * COMPLETE arrives.
 *
 * `sampler` (may be NULL) carries generation overrides; zero/default
 * values mean the payload's own defaults (see pai_proto_sampler_t).
 * `timeout_ms` bounds the whole exchange including negotiation
 * (0 selects a 30000 ms default); on expiry PAI_ERR_TIMEOUT is
 * returned. Other returns:
 *   PAI_ERR_INVALID_ARG  bad arguments or prompt too long (u16 wire)
 *   PAI_ERR_IO           connect/negotiate failure, or the stream
 *                        ended early (peer closed / ERROR frame)
 *   PAI_ERR_CAPABILITY   the server refused negotiation (HELLO_NACK)
 *   PAI_ERR_UNSUPPORTED  the peer negotiated without
 *                        PAI_PROTO_CAP_GENERATE
 *   PAI_OK               COMPLETE received
 *
 * `out_tokens` receives the number of relayed tokens (may be NULL);
 * `on_token` may be NULL to discard tokens.
 */
pai_status_t pai_gw_remote_generate(const char *host, uint16_t port,
                                    const char *prompt,
                                    const pai_proto_sampler_t *sampler,
                                    void (*on_token)(const char *token,
                                                     void *user),
                                    void *user, uint32_t *out_tokens,
                                    uint64_t timeout_ms);

/*
 * One-shot remote embedding (§24/§25): connect to host:port,
 * negotiate, send EMBED(text), and wait for the EMBEDDING reply.
 * The returned vector is heap-allocated with malloc() and must be
 * released by the caller with free(); *out_dim receives its length.
 *
 * `text` must be UTF-8 and at most 65535 bytes (u16 wire length);
 * longer input returns PAI_ERR_INVALID_ARG. `timeout_ms` bounds the
 * whole exchange including negotiation (0 selects the 30000 ms
 * default). Other returns:
 *   PAI_ERR_INVALID_ARG  bad arguments or text too long
 *   PAI_ERR_IO           connect/negotiate failure, or the peer
 *                        closed before the reply
 *   PAI_ERR_CAPABILITY   the server refused negotiation (HELLO_NACK)
 *   PAI_ERR_UNSUPPORTED  the peer negotiated without PAI_PROTO_CAP_EMBED
 *   PAI_ERR_TIMEOUT      no reply within the deadline
 *   PAI_OK               *out_vec / *out_dim set (caller frees)
 */
pai_status_t pai_gw_remote_embed(const char *host, uint16_t port,
                                 const char *text, uint32_t text_len,
                                 float **out_vec, uint32_t *out_dim,
                                 uint64_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* PAI_GATEWAY_REMOTE_H */
