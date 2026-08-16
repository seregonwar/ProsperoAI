/*
 * ProsperoAI Desktop — main process (§7).
 *
 * Owns the real backend services behind the renderer: persistent
 * configuration, the desktop event log (§33), the PS5 device fleet
 * probed over the Prospero Protocol (§24/§25), the model library with
 * the `pai` toolchain pipeline (§8), the benchmark runner (§31) and
 * the Model Hub adapters. The renderer is a view over these services;
 * it never fabricates state.
 */

import { app, BrowserWindow, dialog, ipcMain } from 'electron';
import fs from 'node:fs';
import path from 'node:path';
import { ConfigStore } from './services/config';
import { EventLog } from './services/events';
import { DeviceFleet } from './services/devices';
import { BenchmarkStore } from './services/benchmark';
import { ModelLibrary } from './services/library';
import { HubManager } from './services/hub';
import { checkGateway, completeChat, streamChat } from './services/gateway';
import type {
  AppInfo,
  BenchmarkRun,
  ChatRequestParams,
  ChatStreamDelta,
  ChatStreamResult,
  DesktopEvent,
  DesktopSettings,
  DeviceEntry,
  GatewayStatus,
  HubSearchResult,
  ImportResult,
  InspectResult,
  LibraryEntry,
  OptimizeResult,
} from '../shared/types';

let mainWindow: BrowserWindow | null = null;
let config: ConfigStore;
let events: EventLog;
let devices: DeviceFleet;
let benchmarks: BenchmarkStore;
let library: ModelLibrary;
let hub: HubManager;
const activeChatStreams = new Map<string, AbortController>();

function assertTrustedSender(senderId: number): void {
  if (!mainWindow || senderId !== mainWindow.webContents.id) throw new Error('IPC sender non autorizzato');
}

function sendToRenderer(channel: string, payload: unknown): void {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send(channel, payload);
  }
}

/* ------------------------------------------------------------------ */
/* Diagnostics bundle (§33)                                            */
/* ------------------------------------------------------------------ */

function exportDiagnostics(): string | null {
  try {
    const bundle = {
      generatedAt: new Date().toISOString(),
      app: { version: app.getVersion(), platform: process.platform, isPackaged: app.isPackaged },
      settings: config.get().settings,
      devices: devices.list(),
      library: library.list(),
      benchmarks: benchmarks.list(),
      events: events.list(),
    };
    const dir = path.join(app.getPath('userData'), 'diagnostics');
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, `prosperoai-bundle-${Date.now()}.json`);
    fs.writeFileSync(file, `${JSON.stringify(bundle, null, 2)}\n`);
    return file;
  } catch {
    return null;
  }
}

/* ------------------------------------------------------------------ */
/* IPC surface                                                         */
/* ------------------------------------------------------------------ */

function registerIpc(): void {
  ipcMain.handle('app-info', (event): AppInfo => {
    assertTrustedSender(event.sender.id);
    return { version: app.getVersion(), platform: process.platform, isPackaged: app.isPackaged };
  });

  ipcMain.handle('settings-get', (event): DesktopSettings => {
    assertTrustedSender(event.sender.id);
    return config.get().settings;
  });

  ipcMain.handle('settings-set', (event, rawPatch: unknown): DesktopSettings => {
    assertTrustedSender(event.sender.id);
    const source = rawPatch && typeof rawPatch === 'object' ? rawPatch as Record<string, unknown> : {};
    /* Validate field types before they reach the config store or the OS. */
    const patch: Partial<DesktopSettings> = {};
    if (typeof source.gatewayUrl === 'string' && source.gatewayUrl.trim().length > 0 && source.gatewayUrl.length <= 512) {
      patch.gatewayUrl = source.gatewayUrl.trim();
    }
    for (const key of ['launchAtStartup', 'expertMode', 'reduceMotion'] as const) {
      if (typeof source[key] === 'boolean') patch[key] = source[key];
    }
    const settings = config.patchSettings(patch);
    if (typeof patch.launchAtStartup === 'boolean') {
      app.setLoginItemSettings({ openAtLogin: settings.launchAtStartup });
      events.push('INFO', 'desktop', `launch at startup ${settings.launchAtStartup ? 'enabled' : 'disabled'}`);
    }
    if (typeof patch.gatewayUrl === 'string') {
      events.push('INFO', 'desktop', `gateway endpoint set · ${settings.gatewayUrl}`);
    }
    return settings;
  });

  ipcMain.handle('gateway-check', async (event, rawBaseUrl: unknown): Promise<GatewayStatus> => {
    assertTrustedSender(event.sender.id);
    const baseUrl = rawBaseUrl ?? config.get().settings.gatewayUrl;
    const result = await checkGateway(baseUrl);
    events.push(
      result.ok ? 'INFO' : 'WARN',
      'gateway',
      result.ok
        ? `gateway online · ${result.latencyMs ?? '—'} ms RTT · ${result.models.length} modell${result.models.length === 1 ? 'o' : 'i'}`
        : `gateway offline · ${result.error ?? 'nessuna risposta'}`,
    );
    return result;
  });

  ipcMain.handle('gateway-chat', async (event, baseUrl: unknown, model: unknown, messages: unknown, params: unknown): Promise<string> => {
    assertTrustedSender(event.sender.id);
    events.push('TRACE', 'gateway', `completion richiesta · ${String(model ?? '')}`);
    const answer = await completeChat(baseUrl, model, messages, params);
    events.push('INFO', 'gateway', `completion completata · ${String(model ?? '')}`);
    return answer;
  });

  ipcMain.handle('chat-stream-start', async (event, requestId: unknown, baseUrl: unknown, model: unknown, messages: unknown, params: unknown): Promise<ChatStreamResult> => {
    assertTrustedSender(event.sender.id);
    const id = typeof requestId === 'string' ? requestId : 'anon';
    const controller = new AbortController();
    activeChatStreams.set(id, controller);
    events.push('TRACE', 'gateway', `stream avviato · ${String(model ?? '')}`);
    try {
      const result = await streamChat(baseUrl, model, messages, params ?? {}, id, {
        signal: controller.signal,
        onDelta: (delta) => sendToRenderer('chat-delta', delta),
      });
      sendToRenderer('chat-delta', { requestId: id, token: '', done: true } satisfies ChatStreamDelta);
      events.push('INFO', 'gateway', `stream completato · ${result.tokens} token${result.finishReason ? ` · ${result.finishReason}` : ''} · ${String(model ?? '')}`);
      return {
        ok: true,
        tokens: result.tokens,
        finishReason: result.finishReason,
        usage: result.usage,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : 'stream fallito';
      sendToRenderer('chat-delta', { requestId: id, token: '', done: true, error: message } satisfies ChatStreamDelta);
      events.push('ERROR', 'gateway', `stream fallito · ${message}`);
      return { ok: false, error: message, tokens: 0 };
    } finally {
      activeChatStreams.delete(id);
    }
  });

  ipcMain.handle('chat-stream-cancel', (event, requestId: unknown): boolean => {
    assertTrustedSender(event.sender.id);
    if (typeof requestId !== 'string') return false;
    const controller = activeChatStreams.get(requestId);
    if (!controller) return false;
    controller.abort();
    return true;
  });

  ipcMain.handle('devices-list', (event): DeviceEntry[] => {
    assertTrustedSender(event.sender.id);
    return devices.list();
  });

  ipcMain.handle('devices-add', (event, rawInput: unknown): DeviceEntry[] => {
    assertTrustedSender(event.sender.id);
    const input = (rawInput && typeof rawInput === 'object' ? rawInput : {}) as { name?: unknown; host?: unknown; port?: unknown };
    const added = devices.add({
      name: String(input.name ?? ''),
      host: String(input.host ?? ''),
      port: Number(input.port),
    });
    const last = added[added.length - 1];
    events.push('INFO', 'proto', `endpoint configurato · ${last.name} (${last.host}:${last.port})`);
    return added;
  });

  ipcMain.handle('devices-remove', (event, id: unknown): DeviceEntry[] => {
    assertTrustedSender(event.sender.id);
    const list = devices.remove(String(id ?? ''));
    events.push('INFO', 'proto', `endpoint rimosso`);
    return list;
  });

  ipcMain.handle('devices-probe', async (event): Promise<DeviceEntry[]> => {
    assertTrustedSender(event.sender.id);
    const entries = await devices.probeAll();
    for (const entry of entries) {
      if (entry.probe) {
        events.push(
          entry.probe.ok ? 'INFO' : 'WARN',
          'proto',
          entry.probe.ok
            ? `probe ${entry.name} · caps=${entry.probe.capsNames.join(',') || 'nessuna'} · ${entry.probe.latencyMs?.toFixed(1) ?? '—'} ms RTT`
            : `probe ${entry.name} fallita · ${entry.probe.error ?? 'nessuna risposta'}`,
        );
      }
    }
    return entries;
  });

  ipcMain.handle('events-list', (event): DesktopEvent[] => {
    assertTrustedSender(event.sender.id);
    return events.list();
  });

  ipcMain.handle('events-log', (event, level: unknown, subsystem: unknown, message: unknown): void => {
    assertTrustedSender(event.sender.id);
    const validLevel = ['TRACE', 'INFO', 'WARN', 'ERROR'].includes(String(level));
    events.push(
      validLevel ? (level as DesktopEvent['level']) : 'INFO',
      String(subsystem ?? 'desktop').slice(0, 32),
      String(message ?? '').slice(0, 500),
    );
  });

  ipcMain.handle('diagnostics-export', (event): string | null => {
    assertTrustedSender(event.sender.id);
    const file = exportDiagnostics();
    events.push(file ? 'INFO' : 'ERROR', 'diag', file ? `bundle diagnostico scritto · ${file}` : 'export bundle fallito');
    return file;
  });

  ipcMain.handle('library-list', (event): LibraryEntry[] => {
    assertTrustedSender(event.sender.id);
    return library.list();
  });

  ipcMain.handle('library-import', async (event): Promise<ImportResult> => {
    assertTrustedSender(event.sender.id);
    const result = await dialog.showOpenDialog({
      title: 'Importa modelli in ProsperoAI',
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Modelli ProsperoAI', extensions: ['pai', 'gguf', 'safetensors'] },
        { name: 'Tutti i file', extensions: ['*'] },
      ],
    });
    if (result.canceled || result.filePaths.length === 0) return { queued: 0, entries: [] };
    const entries = await library.importFiles(result.filePaths);
    const ready = entries.filter((entry) => entry.status === 'ready').length;
    const failed = entries.filter((entry) => entry.status === 'failed').length;
    events.push(
      failed > 0 ? 'WARN' : 'INFO',
      'library',
      `${entries.length} import · ${ready} pront${ready === 1 ? 'o' : 'i'}, ${failed} fallit${failed === 1 ? 'o' : 'i'}`,
    );
    return { queued: entries.length, entries };
  });

  ipcMain.handle('library-optimize', async (event, entryId: unknown): Promise<OptimizeResult> => {
    assertTrustedSender(event.sender.id);
    const result = await library.optimize(String(entryId ?? ''));
    events.push(
      result.ok ? 'INFO' : 'WARN',
      'library',
      result.ok
        ? `piano di ottimizzazione · ${result.plan?.model} · ${((((result.plan?.currentBytes ?? 0) - (result.plan?.planBytes ?? 0)) / 1048576)).toFixed(2)} MiB risparmiati (${result.plan?.feasible ? '' : 'oltre budget '}§15)`
        : `ottimizzazione fallita · ${result.error ?? 'errore'}`,
    );
    return result;
  });

  ipcMain.handle('library-inspect', async (event, entryId: unknown): Promise<InspectResult> => {
    assertTrustedSender(event.sender.id);
    const result = await library.inspect(String(entryId ?? ''));
    events.push(
      result.ok ? 'INFO' : 'WARN',
      'library',
      result.ok
        ? `manifest letto · ${result.data?.meta?.name ?? result.data?.path ?? '?'} · ${result.data?.tensors.length ?? 0} tensor${(result.data?.tensors.length ?? 0) === 1 ? 'e' : 'i'} (§20)`
        : `inspect fallito · ${result.error ?? 'errore'}`,
    );
    return result;
  });

  ipcMain.handle('benchmark-run', async (event, model: unknown, rawOptions: unknown): Promise<BenchmarkRun> => {
    assertTrustedSender(event.sender.id);
    events.push('TRACE', 'benchmark', `run avviato · ${String(model ?? '')}`);
    try {
      const run = await benchmarks.run(config.get().settings.gatewayUrl, model, rawOptions);
      events.push('INFO', 'benchmark', `run completato · ${run.model} · ${run.medianTokPerSec?.toFixed(1) ?? '—'} tok/s mediana`);
      return run;
    } catch (error) {
      events.push('ERROR', 'benchmark', `run fallito · ${error instanceof Error ? error.message : 'errore'}`);
      throw error;
    }
  });

  ipcMain.handle('benchmark-list', (event): BenchmarkRun[] => {
    assertTrustedSender(event.sender.id);
    return benchmarks.list();
  });

  ipcMain.handle('benchmark-delete', (event, id: unknown): BenchmarkRun[] => {
    assertTrustedSender(event.sender.id);
    return benchmarks.delete(String(id ?? ''));
  });

  /* --- Model Hub --- */
  ipcMain.handle('hub-search', (event, request: unknown): Promise<HubSearchResult> => {
    assertTrustedSender(event.sender.id);
    return hub.search(request);
  });

  ipcMain.handle('hub-pull-start', (event, request: unknown) => {
    assertTrustedSender(event.sender.id);
    return hub.startPull(request);
  });

  ipcMain.handle('hub-pull-cancel', (event, operationId: unknown): boolean => {
    assertTrustedSender(event.sender.id);
    return hub.cancelPull(operationId);
  });

  ipcMain.handle('list-local-model-files', (event): Promise<string[]> => {
    assertTrustedSender(event.sender.id);
    return hub.listLocalModelFiles();
  });

  ipcMain.handle('select-model-files', async (event): Promise<string[]> => {
    assertTrustedSender(event.sender.id);
    const result = await dialog.showOpenDialog({
      title: 'Importa modelli in ProsperoAI',
      properties: ['openFile', 'multiSelections'],
      filters: [
        { name: 'Modelli ProsperoAI', extensions: ['pai', 'gguf', 'safetensors'] },
        { name: 'Tutti i file', extensions: ['*'] },
      ],
    });
    return result.canceled ? [] : result.filePaths;
  });
}

/* ------------------------------------------------------------------ */
/* Window                                                              */
/* ------------------------------------------------------------------ */

function createWindow(): void {
  const window = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    backgroundColor: '#090c12',
    title: 'ProsperoAI Desktop',
    webPreferences: {
      preload: path.join(__dirname, '../preload/preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow = window;
  window.on('closed', () => { mainWindow = null; });
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) void window.loadURL(devUrl);
  else void window.loadFile(path.join(__dirname, '../renderer/index.html'));
}

app.whenReady().then(() => {
  const baseDir = app.getPath('userData');
  config = new ConfigStore(baseDir);
  config.init();
  events = new EventLog(baseDir);
  events.init();
  devices = new DeviceFleet(config);
  benchmarks = new BenchmarkStore(baseDir);
  library = new ModelLibrary(baseDir);
  library.init();
  hub = new HubManager(baseDir, (progress) => sendToRenderer('hub-progress', progress));

  if (config.get().settings.launchAtStartup) {
    app.setLoginItemSettings({ openAtLogin: true });
  }

  events.push('INFO', 'desktop', `desktop avviato · v${app.getVersion()} · ${process.platform}`);

  registerIpc();
  createWindow();

  /* Live event streaming to the renderer (§33). */
  events.subscribe((entry) => sendToRenderer('desktop-event', entry));

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

/* Abort any in-flight stream before quitting (no orphaned fetches). */
app.on('before-quit', () => {
  for (const controller of activeChatStreams.values()) {
    controller.abort();
  }
  activeChatStreams.clear();
});
