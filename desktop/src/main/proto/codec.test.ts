/*
 * ProsperoAI — codec unit tests (node:test).
 *
 * Ported from the C golden vectors in tests/test_protocol.c: the
 * CRC-32 vector "123456789" -> 0xCBF43926, the frame layout checks
 * (magic, version, flags, message type, payload length, request and
 * session ids, CRC), tamper rejection, and payload codec round trips
 * with the same boundary conditions (u16 lengths, dim caps, trailer
 * validation).
 *
 * Run with: npm run test:proto  (builds the main process first)
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  CAP,
  FLAG_REPLY,
  FLAG_STREAM_END,
  FLAG_STREAM_START,
  HEADER_SIZE,
  MAX_EMBED_DIM,
  MAX_PAYLOAD,
  MSG,
  NACK,
  PROTO_MAGIC,
  PROTO_VERSION,
  PROTO_VERSION_MAJOR,
  PROTO_VERSION_MINOR,
  ProtocolError,
  SAMPLER_MAGIC,
  SAMPLER_SIZE,
  capNames,
  crc32,
  crc32Finish,
  crc32Init,
  crc32Update,
  decodeCapsPayload,
  decodeEmbedPayload,
  decodeEmbeddingPayload,
  decodeErrorPayload,
  decodeGeneratePayload,
  decodeHeader,
  decodeStreamDataPayload,
  decodeTokenPayload,
  decodeU32Payload,
  encodeCapsPayload,
  encodeEmbedPayload,
  encodeEmbeddingPayload,
  encodeErrorPayload,
  encodeFrame,
  encodeGeneratePayload,
  encodeStreamDataPayload,
  encodeTokenPayload,
  encodeU32Payload,
  protoMsgName,
  validateFrame,
} from './codec';

test('crc32 golden vector "123456789"', () => {
  assert.equal(crc32(Buffer.from('123456789', 'ascii')), 0xcbf43926);
});

test('crc32 chunked continuation matches the one-shot result', () => {
  const s = Buffer.from('123456789', 'ascii');
  let running = crc32Init();
  running = crc32Update(running, s, 0, 4);
  running = crc32Update(running, s, 4, 9);
  assert.equal(crc32Finish(running), 0xcbf43926);
});

test('frame encode: header layout matches the wire format', () => {
  const payload = Buffer.from([0x01, 0x02, 0x03]);
  const wire = encodeFrame(MSG.GENERATE, FLAG_STREAM_START, 42, 7, payload);

  assert.equal(wire.length, HEADER_SIZE + 3);
  assert.equal(wire.readUInt32LE(0), PROTO_MAGIC); /* "PAIP" */
  assert.equal(wire[4], PROTO_VERSION_MAJOR);
  assert.equal(wire[5], PROTO_VERSION_MINOR);
  assert.equal(wire.readUInt16LE(6), FLAG_STREAM_START);
  assert.equal(wire.readUInt32LE(8), MSG.GENERATE);
  assert.equal(wire.readUInt32LE(12), 3);
  assert.equal(Number(wire.readBigUInt64LE(16)), 42);
  assert.equal(Number(wire.readBigUInt64LE(24)), 7);
  /* CRC over header[0..32) + payload; reserved field stays zero. */
  const expected = crc32(Buffer.concat([wire.subarray(0, 32), payload]));
  assert.equal(wire.readUInt32LE(32), expected);
  assert.equal(wire.readUInt32LE(36), 0);
  assert.deepEqual([...wire.subarray(40)], [1, 2, 3]);
});

test('frame decode + validate round trip', () => {
  const wire = encodeFrame(MSG.PING, 0, 5, 0, Buffer.alloc(0));
  const header = decodeHeader(wire.subarray(0, HEADER_SIZE));
  assert.equal(header.msgType, MSG.PING);
  assert.equal(header.requestId, 5);
  assert.equal(header.payloadLen, 0);
  assert.ok(validateFrame(wire.subarray(0, HEADER_SIZE), wire.subarray(HEADER_SIZE)));
});

test('frame decode rejects tampered integrity', () => {
  const wire = encodeFrame(MSG.PONG, FLAG_REPLY, 9, 0, Buffer.from('abc'));
  wire[41] ^= 0xff; /* flip a payload byte */
  assert.ok(!validateFrame(wire.subarray(0, HEADER_SIZE), wire.subarray(HEADER_SIZE)));
});

test('frame decode rejects bad magic and reserved flags', () => {
  const wire = encodeFrame(MSG.PING, 0, 1, 0, Buffer.alloc(0));
  wire[0] ^= 0xff;
  assert.throws(() => decodeHeader(wire.subarray(0, HEADER_SIZE)), ProtocolError);
  wire[0] ^= 0xff;
  wire[36] = 1; /* nonzero reserved field */
  assert.throws(() => decodeHeader(wire.subarray(0, HEADER_SIZE)), ProtocolError);
});

test('frame encode rejects oversized payloads and the reserved compression flag', () => {
  assert.throws(
    () => encodeFrame(MSG.GENERATE, 0, 1, 0, Buffer.alloc(MAX_PAYLOAD + 1)),
    ProtocolError,
  );
  assert.throws(
    () => encodeFrame(MSG.GENERATE, 0x8, 1, 0, Buffer.alloc(0)),
    ProtocolError,
  );
});

test('caps payload round trip', () => {
  const caps = CAP.STREAMS | CAP.GENERATE | CAP.EMBED | CAP.TELEMETRY;
  assert.equal(decodeCapsPayload(encodeCapsPayload(caps)), caps);
  assert.deepEqual(capNames(caps), ['streams', 'generate', 'embed', 'telemetry']);
  assert.throws(() => decodeCapsPayload(Buffer.alloc(4)), ProtocolError);
});

test('u32 payload round trip (HELLO_NACK / CLOSE reasons)', () => {
  assert.equal(decodeU32Payload(encodeU32Payload(NACK.CAPS)), NACK.CAPS);
  assert.throws(() => decodeU32Payload(Buffer.alloc(4)), ProtocolError);
});

test('token payload round trip and u16 boundary', () => {
  const token = Buffer.from('hello', 'utf8');
  assert.deepEqual(decodeTokenPayload(encodeTokenPayload(token)), token);
  assert.equal(decodeTokenPayload(encodeTokenPayload('x')).toString('utf8'), 'x');
  assert.throws(() => encodeTokenPayload(Buffer.alloc(0x10000)), ProtocolError);
  assert.throws(() => decodeTokenPayload(Buffer.from([1])), ProtocolError);
});

test('GENERATE v1: prompt only, no trailer', () => {
  const payload = encodeGeneratePayload('ciao');
  assert.equal(payload.length, 2 + 4);
  const decoded = decodeGeneratePayload(payload);
  assert.equal(decoded.prompt, 'ciao');
  assert.equal(decoded.sampler, null);
});

test('GENERATE v2: sampler trailer with magic and wire layout', () => {
  /* float32-exact temperatures (0.7/0.9 are not representable in f32). */
  const payload = encodeGeneratePayload('a', {
    temperature: 0.5,
    topP: 0.25,
    topK: 40,
    maxTokens: 128,
    seed: 99,
  });
  assert.equal(payload.length, 2 + 1 + 4 + SAMPLER_SIZE);
  const trailer = payload.subarray(3);
  assert.equal(trailer.readUInt32LE(0), SAMPLER_MAGIC);
  const decoded = decodeGeneratePayload(payload);
  assert.equal(decoded.prompt, 'a');
  assert.ok(decoded.sampler);
  assert.equal(decoded.sampler.temperature, 0.5);
  assert.equal(decoded.sampler.topP, 0.25);
  assert.equal(decoded.sampler.topK, 40);
  assert.equal(decoded.sampler.maxTokens, 128);
  assert.equal(decoded.sampler.seed, 99);
});

test('GENERATE rejects prompts beyond the u16 wire length', () => {
  assert.throws(() => encodeGeneratePayload('x'.repeat(0x10000)), ProtocolError);
});

test('ERROR payload round trip', () => {
  const payload = encodeErrorPayload(-5, 'boom');
  const decoded = decodeErrorPayload(payload);
  assert.equal(decoded.status, -5);
  assert.equal(decoded.message, 'boom');
  assert.throws(() => decodeErrorPayload(Buffer.alloc(4)), ProtocolError);
});

test('EMBED / EMBEDDING payload round trips', () => {
  assert.equal(decodeEmbedPayload(encodeEmbedPayload('testo')), 'testo');
  /* float32-exact values: 3e-4 is not representable in f32. */
  const values = [1.5, -2.25, 0.0625];
  const decoded = decodeEmbeddingPayload(encodeEmbeddingPayload(values));
  assert.deepEqual(decoded, values);
});

test('EMBEDDING rejects hostile dims and truncated vectors', () => {
  const bad = Buffer.alloc(8);
  bad.writeUInt32LE(MAX_EMBED_DIM + 1, 0);
  assert.throws(() => decodeEmbeddingPayload(bad), ProtocolError);
  const truncated = Buffer.alloc(4 + 2 * 4);
  truncated.writeUInt32LE(3, 0); /* claims 3 floats, has 2 */
  assert.throws(() => decodeEmbeddingPayload(truncated), ProtocolError);
});

test('STREAM_DATA payload round trip', () => {
  const data = Buffer.from('chunk', 'utf8');
  const payload = encodeStreamDataPayload(0, 7, data);
  const decoded = decodeStreamDataPayload(payload);
  assert.equal(decoded.kind, 0);
  assert.equal(decoded.seq, 7);
  assert.equal(decoded.data.toString('utf8'), 'chunk');
});

test('message names for diagnostics', () => {
  assert.equal(protoMsgName(MSG.HELLO), 'HELLO');
  assert.equal(protoMsgName(MSG.COMPLETE), 'COMPLETE');
  assert.equal(protoMsgName(MSG.TOKEN), 'TOKEN');
  assert.equal(protoMsgName(0x7fff), 'UNKNOWN(0x7fff)');
});

test('frame encode uses the packed protocol version', () => {
  assert.equal(PROTO_VERSION, (0 << 8) | 1);
  const wire = encodeFrame(MSG.PING, 0, 1, 0, Buffer.alloc(0));
  assert.equal((wire[4] << 8) | wire[5], PROTO_VERSION);
});

test('stream flags survive a round trip', () => {
  const wire = encodeFrame(MSG.TOKEN, FLAG_STREAM_START | FLAG_STREAM_END, 3, 1, encodeTokenPayload('ab'));
  const header = decodeHeader(wire.subarray(0, HEADER_SIZE));
  assert.equal(header.flags, FLAG_STREAM_START | FLAG_STREAM_END);
  assert.deepEqual(decodeTokenPayload(wire.subarray(HEADER_SIZE)), Buffer.from('ab'));
});
