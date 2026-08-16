/*
 * ProsperoAI Desktop — persistent configuration.
 *
 * userData/config.json holds the real desktop state: gateway URL,
 * experience settings and the configured PS5 device endpoints. The
 * renderer never fabricates defaults; what is not configured simply
 * does not exist.
 */

import fs from 'node:fs';
import path from 'node:path';
import type { DesktopSettings, DeviceConfig } from '../../shared/types';

const DEFAULT_GATEWAY = 'http://127.0.0.1:8080';

export interface AppConfig {
  settings: DesktopSettings;
  devices: DeviceConfig[];
  benchmark: {
    iterations: number;
    maxTokens: number;
  };
}

function defaults(): AppConfig {
  return {
    settings: {
      gatewayUrl: DEFAULT_GATEWAY,
      launchAtStartup: false,
      expertMode: false,
      reduceMotion: false,
    },
    devices: [],
    benchmark: { iterations: 3, maxTokens: 128 },
  };
}

export class ConfigStore {
  private readonly filePath: string;
  private config: AppConfig;

  constructor(baseDir: string) {
    this.filePath = path.join(baseDir, 'config.json');
    this.config = defaults();
  }

  init(): void {
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, 'utf8')) as Partial<AppConfig>;
      this.config = {
        settings: { ...defaults().settings, ...(raw.settings ?? {}) },
        devices: Array.isArray(raw.devices)
          ? raw.devices.filter((d): d is DeviceConfig => Boolean(d && typeof d.id === 'string' && typeof d.host === 'string' && typeof d.port === 'number' && d.port > 0 && d.port < 65536))
          : [],
        benchmark: { ...defaults().benchmark, ...(raw.benchmark ?? {}) },
      };
    } catch {
      this.config = defaults();
    }
  }

  get(): AppConfig {
    return this.config;
  }

  patchSettings(patch: Partial<DesktopSettings>): DesktopSettings {
    this.config.settings = { ...this.config.settings, ...patch };
    this.save();
    return this.config.settings;
  }

  setDevices(devices: DeviceConfig[]): DeviceConfig[] {
    this.config.devices = devices;
    this.save();
    return devices;
  }

  private save(): void {
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
      fs.writeFileSync(this.filePath, `${JSON.stringify(this.config, null, 2)}\n`);
    } catch {
      /* persistence is best effort */
    }
  }
}
