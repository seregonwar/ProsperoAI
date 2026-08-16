/* ProsperoAI Desktop — shared renderer/main process contracts. */

export type ConnectionState = 'online' | 'offline' | 'checking';

export interface GatewayModel {
  id: string;
  object?: string;
  owned_by?: string;
  /* Real metadata served by the gateway (§26): local .pai entries
   * report their on-disk size, remote entries their endpoint. */
  kind?: 'local' | 'remote';
  sizeBytes?: number;
  endpoint?: string;
  broken?: boolean;
}

export interface GatewayStatus {
  ok: boolean;
  baseUrl: string;
  latencyMs?: number;
  models: GatewayModel[];
  error?: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatRequestParams {
  temperature?: number;
  maxTokens?: number;
  topP?: number;
}

export interface ChatStreamDelta {
  requestId: string;
  token: string;
  done: boolean;
  error?: string;
}

export interface ChatStreamResult {
  ok: boolean;
  error?: string;
  tokens: number;
}

export interface AppInfo {
  version: string;
  platform: string;
  isPackaged: boolean;
}

/* ------------------------------------------------------------------ */
/* Settings / devices                                                  */
/* ------------------------------------------------------------------ */

export interface DesktopSettings {
  gatewayUrl: string;
  launchAtStartup: boolean;
  expertMode: boolean;
  reduceMotion: boolean;
}

export interface DeviceConfig {
  id: string;
  name: string;
  host: string;
  port: number;
}

export interface DeviceProbe {
  ok: boolean;
  caps: number;
  capsNames: string[];
  latencyMs?: number; /* median RTT of the non-lost pings */
  lost: number;
  error?: string;
  probedAt: string;
}

export interface DeviceEntry extends DeviceConfig {
  probe?: DeviceProbe;
}

export interface AddDeviceInput {
  name: string;
  host: string;
  port: number;
}

/* ------------------------------------------------------------------ */
/* Desktop event log (§33 diagnostics)                                 */
/* ------------------------------------------------------------------ */

export type EventLevel = 'TRACE' | 'INFO' | 'WARN' | 'ERROR';

export interface DesktopEvent {
  time: string; /* HH:MM:SS.mmm */
  level: EventLevel;
  subsystem: string;
  message: string;
}

/* ------------------------------------------------------------------ */
/* Model library + import pipeline (§7/§8)                             */
/* ------------------------------------------------------------------ */

export type LibraryEntryKind = 'pai' | 'gguf' | 'safetensors' | 'bin' | 'unknown';
export type LibraryEntryStatus = 'imported' | 'converting' | 'ready' | 'failed';

export interface LibraryEntry {
  id: string;
  name: string;
  kind: LibraryEntryKind;
  sourcePath: string;
  libraryPath?: string;
  sizeBytes: number;
  status: LibraryEntryStatus;
  quant?: string;
  message?: string;
  updatedAt: string;
}

export interface ImportResult {
  queued: number;
  entries: LibraryEntry[];
}

/* ------------------------------------------------------------------ */
/* Benchmarks (§31: reproducible runs over the real gateway)           */
/* ------------------------------------------------------------------ */

export interface BenchmarkRunSample {
  iteration: number;
  tokens: number;
  ttftMs: number;   /* time to first token (streaming) */
  totalMs: number;
  tokPerSec: number; /* tokens / total wall time */
}

export interface BenchmarkRun {
  id: string;
  model: string;
  gatewayUrl: string;
  startedAt: string;
  maxTokens: number;
  prompt: string;
  samples: BenchmarkRunSample[];
  medianTokPerSec?: number;
  medianTtftMs?: number;
  medianTokens?: number;
  error?: string;
}

export interface BenchmarkOptions {
  iterations?: number;
  maxTokens?: number;
  prompt?: string;
}

/* ------------------------------------------------------------------ */
/* Model Hub (provider adapters)                                       */
/* ------------------------------------------------------------------ */

export type HubProvider = 'huggingface' | 'ollama' | 'openai-compatible';

export interface HubFile {
  path: string;
  size?: number;
  downloadUrl?: string;
}

export interface HubModel {
  id: string;
  name: string;
  provider: HubProvider;
  author?: string;
  description?: string;
  downloads?: number;
  likes?: number;
  size?: number;
  family?: string;
  quantization?: string;
  lastModified?: string;
  tags: string[];
  files: HubFile[];
  pullable: boolean;
  local: boolean;
}

export interface HubSearchRequest {
  provider: HubProvider;
  query: string;
  baseUrl?: string;
  limit?: number;
}

export interface HubSearchResult {
  provider: HubProvider;
  models: HubModel[];
  latencyMs: number;
  error?: string;
}

export interface HubPullRequest {
  provider: HubProvider;
  modelId: string;
  file?: HubFile;
  baseUrl?: string;
}

export type HubProgressPhase = 'starting' | 'downloading' | 'complete' | 'error' | 'cancelled';

export interface HubProgress {
  operationId: string;
  provider: HubProvider;
  modelId: string;
  phase: HubProgressPhase;
  downloaded: number;
  total?: number;
  percent?: number;
  status?: string;
  localPath?: string;
  error?: string;
}

/* ------------------------------------------------------------------ */
/* Renderer API                                                        */
/* ------------------------------------------------------------------ */

export interface ProsperoApi {
  /* App / settings */
  getAppInfo(): Promise<AppInfo>;
  getSettings(): Promise<DesktopSettings>;
  setSettings(patch: Partial<DesktopSettings>): Promise<DesktopSettings>;

  /* Gateway (§26) */
  checkGateway(baseUrl?: string): Promise<GatewayStatus>;
  completeChat(baseUrl: string, model: string, messages: ChatMessage[], params?: ChatRequestParams): Promise<string>;
  streamChat(
    requestId: string,
    baseUrl: string,
    model: string,
    messages: ChatMessage[],
    params: ChatRequestParams,
    onDelta: (delta: ChatStreamDelta) => void,
  ): Promise<ChatStreamResult>;
  cancelChatStream(requestId: string): void;

  /* PS5 devices (§24/§25 protocol probe) */
  listDevices(): Promise<DeviceEntry[]>;
  addDevice(input: AddDeviceInput): Promise<DeviceEntry[]>;
  removeDevice(id: string): Promise<DeviceEntry[]>;
  probeDevices(): Promise<DeviceEntry[]>;

  /* Diagnostics (§33) */
  listEvents(): Promise<DesktopEvent[]>;
  logEvent(level: EventLevel, subsystem: string, message: string): Promise<void>;
  exportDiagnostics(): Promise<string | null>;
  onEvent(callback: (event: DesktopEvent) => void): () => void;

  /* Model library / import pipeline (§7/§8) */
  listLibrary(): Promise<LibraryEntry[]>;
  importModels(): Promise<ImportResult>;

  /* Benchmarks (§31) */
  runBenchmark(model: string, options?: BenchmarkOptions): Promise<BenchmarkRun>;
  listBenchmarks(): Promise<BenchmarkRun[]>;
  deleteBenchmark(id: string): Promise<BenchmarkRun[]>;

  /* Model Hub */
  selectModelFiles(): Promise<string[]>;
  listLocalModelFiles(): Promise<string[]>;
  searchHub(request: HubSearchRequest): Promise<HubSearchResult>;
  startHubPull(request: HubPullRequest): Promise<{ operationId: string }>;
  cancelHubPull(operationId: string): Promise<boolean>;
  onHubProgress(callback: (progress: HubProgress) => void): () => void;
}
