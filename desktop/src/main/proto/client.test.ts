/*
 * ProsperoAI — protocol client integration tests (node:test).
 *
 * The codec itself is validated against the C golden vectors in
 * codec.test.ts; these tests exercise the socket client end to end:
 * HELLO -> HELLO_ACK negotiation, PING -> PONG round trips and honest
 * failure reporting against an unreachable endpoint.
 */

import assert from 'node:assert/strict';
import net from 'node:net';
import { test } from 'node:test';
import { probeDevice } from './client';
import {
  CAP_KNOWN,
  FLAG_REPLY,
  HEADER_SIZE,
  MSG,
  decodeCapsPayload,
  decodeHeader,
  encodeCapsPayload,
  encodeFrame,
  encodeU32Payload,
} from './codec';

interface TestServer {
  port: number;
  close: () => void;
}

/* Minimal protocol server: auto-answers HELLO/PING, closes on CLOSE. */
function startServer(options: { refuse?: boolean; versionMajor?: number } = {}): Promise<TestServer> {
  return new Promise((resolve) => {
    const server = net.createServer((socket) => {
      let buf: Buffer = Buffer.alloc(0);
      socket.on('data', (chunk: Buffer) => {
        buf = buf.length === 0 ? chunk : Buffer.concat([buf, chunk]);
        for (;;) {
          if (buf.length < HEADER_SIZE) return;
          let header;
          try {
            header = decodeHeader(buf.subarray(0, HEADER_SIZE));
          } catch {
            socket.destroy();
            return;
          }
          if (buf.length < HEADER_SIZE + header.payloadLen) return;
          const payload = buf.subarray(HEADER_SIZE, HEADER_SIZE + header.payloadLen);
          buf = buf.subarray(HEADER_SIZE + header.payloadLen);
          if (header.msgType === MSG.HELLO) {
            if (options.refuse) {
              socket.write(encodeFrame(MSG.HELLO_NACK, FLAG_REPLY, header.requestId, 0, encodeU32Payload(2)));
              continue;
            }
            const caps = decodeCapsPayload(payload) & CAP_KNOWN;
            const frame = encodeFrame(MSG.HELLO_ACK, FLAG_REPLY, header.requestId, 0, encodeCapsPayload(caps));
            if (options.versionMajor !== undefined) frame[4] = options.versionMajor;
            socket.write(frame);
          } else if (header.msgType === MSG.PING) {
            socket.write(encodeFrame(MSG.PONG, FLAG_REPLY, header.requestId, 0, Buffer.alloc(0)));
          } else if (header.msgType === MSG.CLOSE) {
            socket.end();
          }
        }
      });
    });
    server.listen(0, '127.0.0.1', () => {
      const address = server.address() as net.AddressInfo;
      resolve({ port: address.port, close: () => server.close() });
    });
  });
}

/* Bind an ephemeral port and release it: nothing is listening. */
function deadPort(): Promise<number> {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as net.AddressInfo).port;
      server.close(() => resolve(port));
    });
  });
}

test('probeDevice: HELLO negotiation + 5 PING/PONG over a real socket', async () => {
  const server = await startServer();
  try {
    const result = await probeDevice('127.0.0.1', server.port, { pings: 5 });
    assert.equal(result.ok, true);
    assert.ok(result.caps > 0, 'capability mask negotiated');
    assert.ok(result.capsNames.includes('generate'));
    assert.equal(result.pings.length, 5);
    assert.equal(result.lost, 0);
    assert.ok(result.latencyMs !== undefined && result.latencyMs >= 0, 'median RTT measured');
  } finally {
    server.close();
  }
});

test('probeDevice: refused negotiation surfaces an error', async () => {
  const server = await startServer({ refuse: true });
  try {
    const result = await probeDevice('127.0.0.1', server.port, { connectTimeoutMs: 1500 });
    assert.equal(result.ok, false);
    assert.ok(result.error && result.error.length > 0);
  } finally {
    server.close();
  }
});

test('probeDevice: version-mismatched HELLO_ACK is a protocol failure', async () => {
  const server = await startServer({ versionMajor: 1 });
  try {
    const result = await probeDevice('127.0.0.1', server.port, { connectTimeoutMs: 1500 });
    assert.equal(result.ok, false);
    assert.ok(result.error);
  } finally {
    server.close();
  }
});

test('probeDevice: unreachable endpoint reports an error honestly', async () => {
  const port = await deadPort();
  const result = await probeDevice('127.0.0.1', port, { connectTimeoutMs: 1500 });
  assert.equal(result.ok, false);
  assert.ok(result.error, 'error message present');
  assert.equal(result.pings.length, 0);
});
