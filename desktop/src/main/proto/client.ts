/*
 * ProsperoAI — Prospero Protocol client connection (whitepaper §24/§25).
 *
 * TypeScript port of the client side of protocol/connection.c over
 * Node's net.Socket: HELLO -> HELLO_ACK / HELLO_NACK negotiation with
 * version + capability checking, core PING -> PONG, CLOSE handling,
 * and a frame pump with CRC validation. Used by the desktop to probe
 * configured PS5 endpoints (the Devices view) and, later, to bridge
 * generation directly.
 *
 * Negotiation rules ported from connection.c:
 *  - HELLO carries { u32 caps; u32 reserved }; the server intersects
 *    its own caps and replies HELLO_ACK { u32 negotiated }.
 *  - Version major mismatch -> HELLO_NACK(VERSION) / protocol error.
 *  - HELLO_NACK { u32 reason } closes with the reason surfaced.
 *  - PING is auto-replied by the peer core with PONG (same request id).
 */

import net from 'node:net';
import {
  CAP_KNOWN,
  CLOSE_REASON,
  FLAG_REPLY,
  HEADER_SIZE,
  MSG,
  NACK,
  PROTO_VERSION,
  PROTO_VERSION_MAJOR,
  ProtocolError,
  capNames,
  decodeCapsPayload,
  decodeHeader,
  decodeU32Payload,
  encodeCapsPayload,
  encodeFrame,
  encodeU32Payload,
  validateFrame,
  type ProtoFrame,
} from './codec';

export type ClientState = 'new' | 'negotiating' | 'open' | 'closed';

export interface ProtoClientOptions {
  /* Local capability offer (default: all known caps). */
  caps?: number;
  /* Overall connect+negotiate budget in ms (default 3000). */
  connectTimeoutMs?: number;
  onFrame?: (frame: ProtoFrame) => void;
  onClose?: (reason: number) => void;
}

export interface ProbePing {
  rttMs: number; /* round-trip latency in ms; 0 when lost */
  lost: boolean;
}

export interface ProbeResult {
  ok: boolean;
  host: string;
  port: number;
  /* Negotiated capability mask (0 when negotiation failed). */
  caps: number;
  capsNames: string[];
  /* Per-ping outcomes, in order. */
  pings: ProbePing[];
  /* Median RTT over non-lost pings; undefined when all lost. */
  latencyMs?: number;
  lost: number;
  nackReason?: string;
  error?: string;
}

export class ProtoClient {
  private socket: net.Socket | null = null;
  private readBuf: Buffer = Buffer.alloc(0);
  private state: ClientState = 'new';
  private negotiatedCaps = 0;
  private nackReason = 0;
  private closeReason = 0;
  private nextRequestId = 1;
  private readonly localCaps: number;
  private readonly connectTimeoutMs: number;
  private readonly onFrame: ((frame: ProtoFrame) => void) | undefined;
  private readonly onCloseCb: ((reason: number) => void) | undefined;
  private pendingPongs = new Map<number, { resolve: (rttMs: number) => void; reject: (err: Error) => void; timer: NodeJS.Timeout; sentAt: number }>();

  constructor(options: ProtoClientOptions = {}) {
    this.localCaps = (options.caps ?? CAP_KNOWN) & CAP_KNOWN;
    this.connectTimeoutMs = options.connectTimeoutMs ?? 3000;
    this.onFrame = options.onFrame;
    this.onCloseCb = options.onClose;
  }

  get stateName(): ClientState {
    return this.state;
  }

  get caps(): number {
    return this.negotiatedCaps;
  }

  newRequestId(): number {
    const id = this.nextRequestId++;
    if (this.nextRequestId === 0) this.nextRequestId = 1; /* 0 is connection-level */
    return id;
  }

  private failPendingPings(err: Error): void {
    for (const [, entry] of this.pendingPongs) {
      clearTimeout(entry.timer);
      entry.reject(err);
    }
    this.pendingPongs.clear();
  }

  private setClosed(reason: number): void {
    if (this.state === 'closed') return;
    this.state = 'closed';
    this.closeReason = reason;
    this.socket?.destroy();
    this.failPendingPings(new Error(`connection closed (reason ${reason})`));
    this.onCloseCb?.(reason);
  }

  private dispatch(frame: ProtoFrame): void {
    /* Version compatibility is enforced on every frame once the major
     * must match (connection.c conn_handle_open / ack path). */
    if ((frame.version >> 8) !== PROTO_VERSION_MAJOR) {
      this.setClosed(CLOSE_REASON.PROTOCOL);
      return;
    }
    if (this.state === 'negotiating') {
      if (frame.msgType === MSG.HELLO_ACK) {
        const negotiated = this.decodeOrClose(() => decodeCapsPayload(frame.payload));
        if (negotiated === null) return;
        this.negotiatedCaps = negotiated & CAP_KNOWN;
        this.state = 'open';
        return;
      }
      if (frame.msgType === MSG.HELLO_NACK) {
        const reason = this.decodeOrClose(() => decodeU32Payload(frame.payload));
        if (reason === null) return;
        this.nackReason = reason;
        this.setClosed(CLOSE_REASON.NORMAL);
        return;
      }
      this.setClosed(CLOSE_REASON.PROTOCOL); /* first message must be the answer */
      return;
    }
    if (this.state !== 'open') return;

    if (frame.msgType === MSG.PONG) {
      const entry = this.pendingPongs.get(frame.requestId);
      if (entry) {
        clearTimeout(entry.timer);
        this.pendingPongs.delete(frame.requestId);
        entry.resolve(Date.now() - entry.sentAt); /* elapsed RTT, not wall clock */
      }
      return;
    }
    if (frame.msgType === MSG.CLOSE) {
      let reason: number = CLOSE_REASON.NORMAL;
      if (frame.payload.length > 0) reason = decodeU32Payload(frame.payload);
      this.setClosed(reason);
      return;
    }
    this.onFrame?.(frame);
  }

  /* Pre-open decode helpers: a hostile or malformed peer must close
   * the connection, never crash the process (mirrors the C core's
   * PAI_ERR_PROTOCOL handling in conn_handle_preopen). */
  private decodeOrClose(decode: () => number): number | null {
    try {
      return decode();
    } catch {
      this.setClosed(CLOSE_REASON.PROTOCOL);
      return null;
    }
  }

  private pump(): void {
    for (;;) {
      if (this.readBuf.length < HEADER_SIZE) return;
      let header: ReturnType<typeof decodeHeader>;
      try {
        header = decodeHeader(this.readBuf.subarray(0, HEADER_SIZE));
      } catch {
        this.setClosed(CLOSE_REASON.PROTOCOL);
        return;
      }
      if (this.readBuf.length < HEADER_SIZE + header.payloadLen) return; /* wait for payload */
      const payload = this.readBuf.subarray(HEADER_SIZE, HEADER_SIZE + header.payloadLen);
      if (!validateFrame(this.readBuf.subarray(0, HEADER_SIZE), payload)) {
        this.setClosed(CLOSE_REASON.PROTOCOL);
        return;
      }
      this.readBuf = this.readBuf.subarray(HEADER_SIZE + header.payloadLen);
      const frame: ProtoFrame = {
        version: header.version,
        msgType: header.msgType,
        flags: header.flags,
        requestId: header.requestId,
        sessionId: header.sessionId,
        payload: Buffer.from(payload), /* owned copy: readBuf moves on */
      };
      this.dispatch(frame);
      if (this.state === 'closed') return;
    }
  }

  /*
   * Connect to host:port, send HELLO and wait for the negotiation
   * answer. Resolves once OPEN; rejects with a descriptive Error on
   * refusal, timeout or wire failure.
   */
  connect(host: string, port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const socket = net.connect({ host, port });
      this.socket = socket;
      this.readBuf = Buffer.alloc(0);

      let settled = false;
      const settle = (fn: () => void): void => {
        if (settled) return;
        settled = true;
        fn();
      };

      const timer = setTimeout(() => {
        settle(() => {
          socket.destroy();
          reject(new Error('negotiation timeout'));
        });
      }, this.connectTimeoutMs);

      socket.on('connect', () => {
        if (this.state !== 'new') return;
        this.state = 'negotiating';
        const hello = encodeFrame(
          MSG.HELLO,
          0,
          this.newRequestId(),
          0,
          encodeCapsPayload(this.localCaps),
        );
        socket.write(hello);
      });

      socket.on('data', (chunk: Buffer) => {
        this.readBuf = this.readBuf.length === 0 ? chunk : Buffer.concat([this.readBuf, chunk]);
        /* A single poll may deliver the whole HELLO_ACK frame, so the
         * state can transition inside pump(); read it back afterwards. */
        this.pump();
        const state = this.state as ClientState;
        if (state === 'open') {
          settle(() => {
            clearTimeout(timer);
            resolve();
          });
        } else if (state === 'closed') {
          settle(() => {
            clearTimeout(timer);
            reject(this.negotiationError());
          });
        }
      });

      socket.on('error', (err: NodeJS.ErrnoException) => {
        settle(() => {
          clearTimeout(timer);
          this.state = 'closed';
          const detail = err.code === 'ECONNREFUSED' ? 'connection refused' : err.message;
          reject(new Error(`${host}:${port} — ${detail}`));
        });
      });

      socket.on('close', () => {
        if (settled) return;
        if (this.state !== 'closed') {
          this.setClosed(CLOSE_REASON.PEER_GONE);
        }
        settle(() => {
          clearTimeout(timer);
          if (this.state === 'closed' && this.nackReason !== 0) {
            reject(this.negotiationError());
          } else {
            reject(new Error(`${host}:${port} — connection closed during negotiation`));
          }
        });
      });

    });
  }

  private negotiationError(): Error {
    if (this.nackReason === NACK.CAPS) {
      return new Error('payload refused the connection (no common capabilities)');
    }
    if (this.nackReason === NACK.VERSION) {
      return new Error('payload refused the connection (protocol version mismatch)');
    }
    return new Error('payload closed the connection during negotiation');
  }

  /* Send one frame; throws when the connection is not OPEN. */
  send(msgType: number, flags: number, requestId: number, sessionId: number, payload: Buffer): void {
    if (this.state !== 'open') throw new ProtocolError('connection is not open');
    this.socket?.write(encodeFrame(msgType, flags, requestId, sessionId, payload));
  }

  /*
   * One keepalive round trip: send PING and wait for the core
   * auto-replied PONG. Resolves with the RTT in milliseconds; rejects
   * on per-ping timeout or connection loss.
   */
  ping(timeoutMs = 1000): Promise<number> {
    if (this.state !== 'open') return Promise.reject(new ProtocolError('connection is not open'));
    return new Promise((resolve, reject) => {
      const requestId = this.newRequestId();
      const timer = setTimeout(() => {
        this.pendingPongs.delete(requestId);
        reject(new Error('ping timeout'));
      }, timeoutMs);
      this.pendingPongs.set(requestId, {
        resolve: (rttMs) => resolve(rttMs),
        reject,
        timer,
        sentAt: Date.now(),
      });
      try {
        this.send(MSG.PING, 0, requestId, 0, Buffer.alloc(0));
      } catch (err) {
        clearTimeout(timer);
        this.pendingPongs.delete(requestId);
        reject(err as Error);
      }
    });
  }

  /* Graceful close: send CLOSE (best effort) and tear down. */
  close(): void {
    if (this.state === 'open') {
      try {
        this.socket?.write(encodeFrame(MSG.CLOSE, 0, 0, 0, encodeU32Payload(CLOSE_REASON.APP)));
      } catch {
        /* best effort */
      }
    }
    this.state = 'closed';
    this.socket?.destroy();
    this.failPendingPings(new Error('client closed'));
  }
}

/* ------------------------------------------------------------------ */
/* Device probe (§24/§25): negotiate, N pings, report.                 */
/* ------------------------------------------------------------------ */

export interface ProbeOptions {
  pings?: number;
  connectTimeoutMs?: number;
  pingTimeoutMs?: number;
}

/*
 * Full protocol probe of one endpoint: connect + negotiate, then
 * `pings` (default 5) keepalive round trips. The median of the
 * non-lost RTTs is reported as latencyMs. Individual ping losses are
 * reported per-ping; the probe fails only when the connection itself
 * cannot be established or negotiated.
 */
export async function probeDevice(host: string, port: number, options: ProbeOptions = {}): Promise<ProbeResult> {
  const pings = Math.max(1, Math.min(16, options.pings ?? 5));
  const connectTimeoutMs = options.connectTimeoutMs ?? 3000;
  const pingTimeoutMs = options.pingTimeoutMs ?? 1500;
  const client = new ProtoClient({ connectTimeoutMs });

  try {
    await client.connect(host, port);
  } catch (err) {
    return {
      ok: false,
      host,
      port,
      caps: 0,
      capsNames: [],
      pings: [],
      lost: 0,
      nackReason: err instanceof Error && /refused the connection/.test(err.message) ? err.message : undefined,
      error: err instanceof Error ? err.message : 'probe failed',
    };
  }

  const results: ProbePing[] = [];
  for (let i = 0; i < pings; i++) {
    try {
      const rttMs = await client.ping(pingTimeoutMs);
      results.push({ rttMs, lost: false });
    } catch {
      results.push({ rttMs: 0, lost: true });
    }
  }
  client.close();

  const rtts = results.filter((ping) => !ping.lost).map((ping) => ping.rttMs).sort((a, b) => a - b);
  const lost = results.filter((ping) => ping.lost).length;
  return {
    ok: true,
    host,
    port,
    caps: client.caps,
    capsNames: capNames(client.caps),
    pings: results,
    latencyMs: rtts.length > 0 ? rtts[Math.floor(rtts.length / 2)] : undefined,
    lost,
  };
}

