/*
 * ProsperoAI Desktop — PS5 device fleet (§24/§25).
 *
 * Devices are endpoints the user configures (name + host:port). Every
 * probe is a real Prospero Protocol exchange from proto/client.ts:
 * TCP connect, HELLO/HELLO_ACK capability negotiation and PING/PONG
 * latency sampling. There are no fabricated consoles: an endpoint that
 * is not configured does not exist, and an endpoint that does not
 * answer reports its error honestly.
 */

import crypto from 'node:crypto';
import type { AddDeviceInput, DeviceConfig, DeviceEntry, DeviceProbe } from '../../shared/types';
import { probeDevice, type ProbeResult } from '../proto/client';
import type { ConfigStore } from './config';

function probeToResult(result: ProbeResult): DeviceProbe {
  return {
    ok: result.ok,
    caps: result.caps,
    capsNames: result.capsNames,
    latencyMs: result.latencyMs,
    lost: result.lost,
    error: result.error,
    probedAt: new Date().toISOString(),
  };
}

export class DeviceFleet {
  constructor(private readonly config: ConfigStore) {}

  list(): DeviceEntry[] {
    return this.config.get().devices;
  }

  add(input: AddDeviceInput): DeviceEntry[] {
    const name = input.name.trim().slice(0, 64) || `${input.host}:${input.port}`;
    const host = input.host.trim().slice(0, 255);
    const port = Math.trunc(input.port);
    /* Number.isInteger also rejects NaN (NaN comparisons are false). */
    if (!host || !Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Endpoint non valido');
    const device: DeviceConfig = { id: crypto.randomUUID(), name, host, port };
    this.config.setDevices([...this.config.get().devices, device]);
    return this.config.get().devices;
  }

  remove(id: string): DeviceEntry[] {
    this.config.setDevices(this.config.get().devices.filter((device) => device.id !== id));
    return this.config.get().devices;
  }

  async probeOne(device: DeviceConfig): Promise<DeviceEntry> {
    const result = await probeDevice(device.host, device.port, { pings: 5 });
    return { ...device, probe: probeToResult(result) };
  }

  async probeAll(): Promise<DeviceEntry[]> {
    const devices = this.config.get().devices;
    const entries: DeviceEntry[] = [];
    for (const device of devices) {
      entries.push(await this.probeOne(device));
    }
    /* Persist probe results as the last known state. */
    this.config.setDevices(entries.map(({ probe, ...config }) => config));
    return entries;
  }
}
