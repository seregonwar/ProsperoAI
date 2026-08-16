/*
 * ProsperoAI — Prospero Protocol frame codec (whitepaper §24/§25).
 *
 * Faithful TypeScript port of protocol/frame.c and protocol/crc32.c:
 * 40-byte little-endian header + payload, CRC-32 (ISO-HDLC) over
 * header[0..32) + payload, and the v0 message payload codecs the
 * desktop needs (HELLO caps, PING, CLOSE, TOKEN, GENERATE v1/v2 with
 * sampler trailer, EMBED/EMBEDDING, ERROR, STREAM_DATA).
 *
 * Pure codec: no sockets, no timers — unit-tested against the C golden
 * vectors (tests/test_protocol.c).
 */

export const PROTO_VERSION_MAJOR = 0;
export const PROTO_VERSION_MINOR = 1;
export const PROTO_VERSION = (PROTO_VERSION_MAJOR << 8) | PROTO_VERSION_MINOR;

/* Frame magic: ASCII "PAIP" as a little-endian u32 (bytes 50 41 49 50). */
export const PROTO_MAGIC = 0x50494150;
export const HEADER_SIZE = 40;
export const MAX_PAYLOAD = 1 << 20;

/* Frame flags. */
export const FLAG_REPLY = 1 << 0;
export const FLAG_STREAM_START = 1 << 1;
export const FLAG_STREAM_END = 1 << 2;
export const FLAG_COMPRESSED = 1 << 3; /* reserved; v0 receivers reject it */

/* Message types (enum pai_proto_msg). */
export const MSG = {
  HELLO: 0x0001,
  HELLO_ACK: 0x0002,
  HELLO_NACK: 0x0003,
  PING: 0x0004,
  PONG: 0x0005,
  ERROR: 0x0006,
  CLOSE: 0x0007,
  SESSION_CREATE: 0x0101,
  SESSION_CREATED: 0x0102,
  SESSION_CLOSE: 0x0103,
  SESSION_CLOSED: 0x0104,
  GENERATE: 0x0201,
  ACCEPTED: 0x0202,
  TOKEN: 0x0203,
  COMPLETE: 0x0204,
  EMBED: 0x0205,
  EMBEDDING: 0x0206,
  STREAM_DATA: 0x0301,
  STREAM_CLOSE: 0x0302,
} as const;

/* Stream kinds carried by STREAM_DATA payloads. */
export const STREAM_KIND = {
  TELEMETRY: 0,
  PROFILER: 1,
  TRANSFER: 2,
  LOG: 3,
  COMPILE: 4,
  GENERATE: 5,
} as const;

/* Capability bits. */
export const CAP = {
  STREAMS: 1 << 0,
  SESSIONS: 1 << 1,
  GENERATE: 1 << 2,
  EMBED: 1 << 3,
  TRANSFER: 1 << 4,
  TELEMETRY: 1 << 5,
  COMPRESSION: 1 << 6,
} as const;

export const CAP_KNOWN =
  CAP.STREAMS | CAP.SESSIONS | CAP.GENERATE | CAP.EMBED | CAP.TRANSFER |
  CAP.TELEMETRY | CAP.COMPRESSION;

export const NACK = { VERSION: 1, CAPS: 2 } as const;
export const CLOSE_REASON = {
  NORMAL: 0,
  PROTOCOL: 1,
  APP: 2,
  PEER_GONE: 3,
} as const;

/* GENERATE v2 sampler trailer. */
export const SAMPLER_MAGIC = 0x50494153; /* "PAIS" little-endian */
export const SAMPLER_SIZE = 4 + 4 + 4 + 4 + 8; /* temp, top_p, top_k, max_tokens, seed */

export const MAX_EMBED_DIM = 1 << 16;

export interface ProtoFrame {
  version: number; /* packed major<<8|minor, as read from the wire */
  msgType: number;
  flags: number;
  requestId: number;
  sessionId: number;
  payload: Buffer;
}

export interface ProtoSampler {
  temperature: number; /* float32; 0 = greedy */
  topP: number;        /* float32; 1 = off */
  topK: number;        /* 0 = off */
  maxTokens: number;   /* 0 = model default */
  seed: number;        /* uint64; 0 = default */
}

/* ------------------------------------------------------------------ */
/* CRC-32 (ISO-HDLC, zlib polynomial 0xEDB88320)                       */
/* ------------------------------------------------------------------ */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

export function crc32Init(): number {
  return 0xffffffff;
}

export function crc32Update(running: number, data: Uint8Array, start = 0, end = data.length): number {
  let c = running >>> 0;
  for (let i = start; i < end; i++) {
    c = CRC_TABLE[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  }
  return c >>> 0;
}

export function crc32Finish(running: number): number {
  return (running ^ 0xffffffff) >>> 0;
}

/* One-shot CRC-32. */
export function crc32(data: Uint8Array): number {
  return crc32Finish(crc32Update(crc32Init(), data));
}

/* ------------------------------------------------------------------ */
/* Frame codec                                                         */
/* ------------------------------------------------------------------ */

export class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ProtocolError';
  }
}

/*
 * Serialize a full frame. Throws ProtocolError when the frame is
 * malformed (payload too large, reserved compression flag).
 */
export function encodeFrame(
  msgType: number,
  flags: number,
  requestId: number,
  sessionId: number,
  payload: Buffer,
): Buffer {
  if (payload.length > MAX_PAYLOAD) throw new ProtocolError('payload too large');
  if (flags & FLAG_COMPRESSED) throw new ProtocolError('compression is reserved in v0');

  const buf = Buffer.alloc(HEADER_SIZE + payload.length);
  buf.writeUInt32LE(PROTO_MAGIC, 0);
  buf[4] = PROTO_VERSION_MAJOR;
  buf[5] = PROTO_VERSION_MINOR;
  buf.writeUInt16LE(flags, 6);
  buf.writeUInt32LE(msgType, 8);
  buf.writeUInt32LE(payload.length, 12);
  buf.writeBigUInt64LE(BigInt(requestId), 16);
  buf.writeBigUInt64LE(BigInt(sessionId), 24);
  payload.copy(buf, HEADER_SIZE);
  /* CRC covers header[0..32) (the crc field itself is still zero) and
   * the payload; the reserved field at [36..40) is not covered. */
  const crc = crc32Finish(crc32Update(crc32Update(crc32Init(), buf, 0, 32), payload));
  buf.writeUInt32LE(crc, 32);
  return buf;
}

export interface FrameHeader {
  version: number;
  msgType: number;
  flags: number;
  requestId: number;
  sessionId: number;
  payloadLen: number;
  crc: number;
}

/*
 * Parse a header. Validates magic, payload-length cap and the reserved
 * field. Version compatibility is the connection's decision (it must
 * reach the negotiator for HELLO frames), so the parsed version is
 * returned rather than rejected here. Throws ProtocolError on
 * malformed input.
 */
export function decodeHeader(data: Buffer): FrameHeader {
  if (data.length < HEADER_SIZE) throw new ProtocolError('short header');
  const magic = data.readUInt32LE(0);
  if (magic !== PROTO_MAGIC) throw new ProtocolError('bad magic');
  const reserved = data.readUInt32LE(36);
  if (reserved !== 0) throw new ProtocolError('nonzero reserved field');
  const payloadLen = data.readUInt32LE(12);
  if (payloadLen > MAX_PAYLOAD) throw new ProtocolError('payload too large');
  return {
    version: (data[4] << 8) | data[5],
    msgType: data.readUInt32LE(8),
    flags: data.readUInt16LE(6),
    requestId: Number(data.readBigUInt64LE(16)),
    sessionId: Number(data.readBigUInt64LE(24)),
    payloadLen,
    crc: data.readUInt32LE(32),
  };
}

/* Validate integrity of a complete frame (header + payload). */
export function validateFrame(header: Buffer, payload: Buffer | null): boolean {
  const expected = header.readUInt32LE(32);
  const running = crc32Update(crc32Update(crc32Init(), header, 0, 32), payload ?? Buffer.alloc(0));
  return crc32Finish(running) === expected;
}

/* Assemble a host-side frame from a header buffer + payload buffer. */
export function parseFrame(header: Buffer, payload: Buffer): ProtoFrame {
  const h = decodeHeader(header);
  return {
    version: h.version,
    msgType: h.msgType,
    flags: h.flags,
    requestId: h.requestId,
    sessionId: h.sessionId,
    payload,
  };
}

/* ------------------------------------------------------------------ */
/* Message payload codecs (v0 formats)                                 */
/* ------------------------------------------------------------------ */

/* HELLO / HELLO_ACK: { u32 caps; u32 reserved }. */
export function encodeCapsPayload(caps: number): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeUInt32LE(caps >>> 0, 0);
  return buf;
}

export function decodeCapsPayload(payload: Buffer): number {
  if (payload.length < 8) throw new ProtocolError('short caps payload');
  return payload.readUInt32LE(0) >>> 0;
}

/* HELLO_NACK / CLOSE: { u32 reason; u32 reserved }. */
export function encodeU32Payload(value: number): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeUInt32LE(value >>> 0, 0);
  return buf;
}

export function decodeU32Payload(payload: Buffer): number {
  if (payload.length < 8) throw new ProtocolError('short u32 payload');
  return payload.readUInt32LE(0) >>> 0;
}

/* ERROR: { i32 status; u32 msg_len; u8 msg[msg_len] } (msg_len capped at 255). */
export function encodeErrorPayload(status: number, message?: string): Buffer {
  const msg = Buffer.from(message ?? '', 'utf8').subarray(0, 255);
  const buf = Buffer.alloc(4 + 4 + msg.length);
  buf.writeInt32LE(status | 0, 0);
  buf.writeUInt32LE(msg.length, 4);
  msg.copy(buf, 8);
  return buf;
}

export function decodeErrorPayload(payload: Buffer): { status: number; message: string } {
  if (payload.length < 8) throw new ProtocolError('short error payload');
  const status = payload.readInt32LE(0);
  const msgLen = payload.readUInt32LE(4);
  if (8 + msgLen > payload.length) throw new ProtocolError('truncated error message');
  return { status, message: payload.toString('utf8', 8, 8 + msgLen) };
}

/* TOKEN: { u16 token_len; u8 token[token_len] }. */
export function encodeTokenPayload(token: Buffer | string): Buffer {
  const bytes = typeof token === 'string' ? Buffer.from(token, 'utf8') : token;
  if (bytes.length > 0xffff) throw new ProtocolError('token too long');
  const buf = Buffer.alloc(2 + bytes.length);
  buf.writeUInt16LE(bytes.length, 0);
  bytes.copy(buf, 2);
  return buf;
}

export function decodeTokenPayload(payload: Buffer): Buffer {
  if (payload.length < 2) throw new ProtocolError('short token payload');
  const len = payload.readUInt16LE(0);
  if (2 + len > payload.length) throw new ProtocolError('truncated token');
  return payload.subarray(2, 2 + len);
}

/*
 * GENERATE v1: { u16 prompt_len; u8 prompt[prompt_len] } (UTF-8).
 * GENERATE v2: the same, followed by { u32 sampler_magic; u8
 * sampler[SAMPLER_SIZE] }. NULL sampler = v1 encoding.
 */
export function encodeGeneratePayload(prompt: string, sampler?: ProtoSampler | null): Buffer {
  const promptBytes = Buffer.from(prompt, 'utf8');
  if (promptBytes.length > 0xffff) throw new ProtocolError('prompt too long');
  const trailer = sampler ? 4 + SAMPLER_SIZE : 0;
  const buf = Buffer.alloc(2 + promptBytes.length + trailer);
  buf.writeUInt16LE(promptBytes.length, 0);
  promptBytes.copy(buf, 2);
  if (sampler) {
    let off = 2 + promptBytes.length;
    buf.writeUInt32LE(SAMPLER_MAGIC, off);
    off += 4;
    buf.writeFloatLE(sampler.temperature, off); off += 4;
    buf.writeFloatLE(sampler.topP, off); off += 4;
    buf.writeUInt32LE(sampler.topK >>> 0, off); off += 4;
    buf.writeUInt32LE(sampler.maxTokens >>> 0, off); off += 4;
    /* seed is a u64 on the wire; never truncate to 32 bits. */
    buf.writeBigUInt64LE(BigInt.asUintN(64, BigInt(Math.trunc(sampler.seed))), off); off += 8;
  }
  return buf;
}

export interface DecodedGenerate {
  prompt: string;
  sampler: ProtoSampler | null; /* null = no trailer (v1 or v2-without) */
}

/* Decodes both v1 and v2 payloads. */
export function decodeGeneratePayload(payload: Buffer): DecodedGenerate {
  if (payload.length < 2) throw new ProtocolError('short generate payload');
  const promptLen = payload.readUInt16LE(0);
  if (2 + promptLen > payload.length) throw new ProtocolError('truncated generate prompt');
  const prompt = payload.toString('utf8', 2, 2 + promptLen);
  let sampler: ProtoSampler | null = null;
  const trailer = payload.subarray(2 + promptLen);
  if (trailer.length === 4 + SAMPLER_SIZE) {
    if (trailer.readUInt32LE(0) !== SAMPLER_MAGIC) throw new ProtocolError('bad sampler magic');
    let off = 4;
    const temperature = trailer.readFloatLE(off); off += 4;
    const topP = trailer.readFloatLE(off); off += 4;
    const topK = trailer.readUInt32LE(off); off += 4;
    const maxTokens = trailer.readUInt32LE(off); off += 4;
    const seed = Number(trailer.readBigUInt64LE(off));
    sampler = { temperature, topP, topK, maxTokens, seed };
  } else if (trailer.length !== 0) {
    throw new ProtocolError('malformed generate trailer');
  }
  return { prompt, sampler };
}

/* EMBED: { u16 text_len; u8 text[text_len] } (UTF-8). */
export function encodeEmbedPayload(text: string): Buffer {
  const bytes = Buffer.from(text, 'utf8');
  if (bytes.length > 0xffff) throw new ProtocolError('text too long');
  const buf = Buffer.alloc(2 + bytes.length);
  buf.writeUInt16LE(bytes.length, 0);
  bytes.copy(buf, 2);
  return buf;
}

export function decodeEmbedPayload(payload: Buffer): string {
  if (payload.length < 2) throw new ProtocolError('short embed payload');
  const len = payload.readUInt16LE(0);
  if (2 + len > payload.length) throw new ProtocolError('truncated embed text');
  return payload.toString('utf8', 2, 2 + len);
}

/* EMBEDDING: { u32 dim; f32 values[dim] } little-endian. */
export function encodeEmbeddingPayload(values: number[] | Float32Array): Buffer {
  if (values.length > MAX_EMBED_DIM) throw new ProtocolError('embedding dim too large');
  const buf = Buffer.alloc(4 + values.length * 4);
  buf.writeUInt32LE(values.length, 0);
  for (let i = 0; i < values.length; i++) buf.writeFloatLE(values[i], 4 + i * 4);
  return buf;
}

export function decodeEmbeddingPayload(payload: Buffer): number[] {
  if (payload.length < 4) throw new ProtocolError('short embedding payload');
  const dim = payload.readUInt32LE(0);
  if (dim > MAX_EMBED_DIM) throw new ProtocolError('embedding dim too large');
  if (4 + dim * 4 > payload.length) throw new ProtocolError('truncated embedding vector');
  const values: number[] = new Array(dim);
  for (let i = 0; i < dim; i++) values[i] = payload.readFloatLE(4 + i * 4);
  return values;
}

/*
 * STREAM_DATA: { u8 kind; u8 reserved[3]; u32 seq; u32 data_len;
 *                u8 data[data_len] }.
 */
export function encodeStreamDataPayload(kind: number, seq: number, data: Buffer): Buffer {
  const buf = Buffer.alloc(12 + data.length);
  buf[0] = kind & 0xff;
  buf.writeUInt32LE(seq >>> 0, 4);
  buf.writeUInt32LE(data.length, 8);
  data.copy(buf, 12);
  return buf;
}

export function decodeStreamDataPayload(payload: Buffer): { kind: number; seq: number; data: Buffer } {
  if (payload.length < 12) throw new ProtocolError('short stream payload');
  const kind = payload[0];
  const seq = payload.readUInt32LE(4);
  const dataLen = payload.readUInt32LE(8);
  if (12 + dataLen > payload.length) throw new ProtocolError('truncated stream data');
  return { kind, seq, data: payload.subarray(12, 12 + dataLen) };
}

/* Stable name for a message type (diagnostics). */
const MSG_NAMES: Record<number, string> = {
  [MSG.HELLO]: 'HELLO',
  [MSG.HELLO_ACK]: 'HELLO_ACK',
  [MSG.HELLO_NACK]: 'HELLO_NACK',
  [MSG.PING]: 'PING',
  [MSG.PONG]: 'PONG',
  [MSG.ERROR]: 'ERROR',
  [MSG.CLOSE]: 'CLOSE',
  [MSG.SESSION_CREATE]: 'SESSION_CREATE',
  [MSG.SESSION_CREATED]: 'SESSION_CREATED',
  [MSG.SESSION_CLOSE]: 'SESSION_CLOSE',
  [MSG.SESSION_CLOSED]: 'SESSION_CLOSED',
  [MSG.GENERATE]: 'GENERATE',
  [MSG.ACCEPTED]: 'ACCEPTED',
  [MSG.TOKEN]: 'TOKEN',
  [MSG.COMPLETE]: 'COMPLETE',
  [MSG.EMBED]: 'EMBED',
  [MSG.EMBEDDING]: 'EMBEDDING',
  [MSG.STREAM_DATA]: 'STREAM_DATA',
  [MSG.STREAM_CLOSE]: 'STREAM_CLOSE',
};

export function protoMsgName(msgType: number): string {
  return MSG_NAMES[msgType] ?? `UNKNOWN(0x${msgType.toString(16)})`;
}

/* Human-readable capability names (devices view). */
const CAP_NAMES: Array<[number, string]> = [
  [CAP.STREAMS, 'streams'],
  [CAP.SESSIONS, 'sessions'],
  [CAP.GENERATE, 'generate'],
  [CAP.EMBED, 'embed'],
  [CAP.TRANSFER, 'transfer'],
  [CAP.TELEMETRY, 'telemetry'],
  [CAP.COMPRESSION, 'compression'],
];

export function capNames(caps: number): string[] {
  return CAP_NAMES.filter(([bit]) => (caps & bit) !== 0).map(([, name]) => name);
}
