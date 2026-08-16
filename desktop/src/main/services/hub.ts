/*
 * ProsperoAI Desktop — Model Hub adapters (§7 model discovery).
 *
 * Real provider integrations, all executed in the main process:
 * Hugging Face public search + artifact downloads (with progress,
 * partial files and cancellation), Ollama local daemon tags + pulls
 * (NDJSON progress), and OpenAI-compatible registry discovery.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import { mkdir, unlink, rename, readdir } from 'node:fs/promises';
import path from 'node:path';
import { once } from 'node:events';
import type {
  HubFile,
  HubModel,
  HubProgress,
  HubProvider,
  HubPullRequest,
  HubSearchRequest,
  HubSearchResult,
} from '../../shared/types';
import { detectFamilyId } from '../../shared/families';
import { isRecord, normaliseBaseUrl, parseModels, requestJson, asNumber } from './gateway';

const DEFAULT_OLLAMA = 'http://127.0.0.1:11434';
const DEFAULT_GATEWAY = 'http://127.0.0.1:8080';
const MAX_HUB_RESULTS = 40;

function fileUrl(repoId: string, filename: string): string {
  const encoded = filename.split('/').map((part) => encodeURIComponent(part)).join('/');
  return `https://huggingface.co/${repoId}/resolve/main/${encoded}`;
}

function hubFiles(repoId: string, siblings: unknown[]): HubFile[] {
  const files = siblings.filter(isRecord).map((item) => {
    const filename = typeof item.rfilename === 'string' ? item.rfilename : '';
    return { path: filename, size: asNumber(item.size), downloadUrl: filename ? fileUrl(repoId, filename) : undefined };
  }).filter((item) => item.path.length > 0 && /\.(gguf|safetensors|bin)$/i.test(item.path));
  const score = (file: HubFile): number =>
    /\.gguf$/i.test(file.path) ? 0 : /(^|\/)(model|pytorch_model)(\.safetensors|\.bin)$/i.test(file.path) ? 1 : 2;
  return files.sort((left, right) => score(left) - score(right)).slice(0, 24);
}

async function searchHuggingFace(query: string, limit: number): Promise<HubModel[]> {
  const params = new URLSearchParams({ limit: String(limit), sort: 'downloads', direction: '-1' });
  params.append('expand[]', 'siblings');
  if (query) params.set('search', query);
  const headers: Record<string, string> = {};
  if (process.env.HUGGINGFACE_TOKEN) headers.Authorization = `Bearer ${process.env.HUGGINGFACE_TOKEN}`;
  const value = await requestJson('https://huggingface.co', `/api/models?${params}`, { headers }, 15_000);
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((item) => {
    const id = typeof item.id === 'string' ? item.id : '';
    const tags = Array.isArray(item.tags) ? item.tags.filter((tag): tag is string => typeof tag === 'string').slice(0, 12) : [];
    const siblings = Array.isArray(item.siblings) ? item.siblings : [];
    const files = hubFiles(id, siblings);
    return {
      id,
      name: id.split('/').pop() ?? id,
      provider: 'huggingface' as const,
      author: typeof item.author === 'string' ? item.author : id.split('/')[0],
      description: tags.slice(0, 3).join(' · '),
      downloads: asNumber(item.downloads),
      likes: asNumber(item.likes),
      lastModified: typeof item.lastModified === 'string' ? item.lastModified : undefined,
      tags,
      family: detectFamilyId(id, tags),
      files,
      pullable: files.length > 0,
      local: false,
    };
  }).filter((item) => item.id.length > 0);
}

function ollamaModel(item: Record<string, unknown>, local: boolean): HubModel {
  const details = isRecord(item.details) ? item.details : null;
  const id = typeof item.name === 'string' ? item.name : typeof item.model === 'string' ? item.model : 'unknown';
  return {
    id,
    name: id,
    provider: 'ollama',
    local,
    pullable: true,
    tags: [],
    files: [],
    size: asNumber(item.size),
    family: typeof details?.family === 'string' ? detectFamilyId(id, [], details.family) : undefined,
    quantization: typeof details?.quantization_level === 'string' ? details.quantization_level : undefined,
  };
}

async function searchOllama(query: string, baseUrl: string, limit: number): Promise<HubModel[]> {
  const raw = await requestJson(baseUrl, '/api/tags', undefined, 5000);
  const value = isRecord(raw) ? raw : null;
  const models = Array.isArray(value?.models) ? value.models.filter(isRecord).map((item) => ollamaModel(item, true)) : [];
  const normalized = query.toLowerCase();
  const filtered = models.filter((model) => model.id.toLowerCase().includes(normalized));
  if (filtered.length > 0 || !query) return filtered.slice(0, limit);
  // Ollama's documented API exposes local tags; exact library pulls are valid even before a tag is local.
  return [{ id: query, name: query, provider: 'ollama', local: false, pullable: true, tags: ['Ollama Library'], files: [] }];
}

async function searchOpenAI(query: string, baseUrl: string, limit: number): Promise<HubModel[]> {
  const models = parseModels(await requestJson(baseUrl, '/v1/models'));
  return models.filter((model) => !query || model.id.toLowerCase().includes(query.toLowerCase())).slice(0, limit).map((model) => ({
    id: model.id,
    name: model.id,
    provider: 'openai-compatible' as const,
    author: model.owned_by,
    local: false,
    pullable: false,
    tags: ['OpenAI-compatible'],
    files: [],
  }));
}

export class HubManager {
  private readonly activePulls = new Map<string, AbortController>();

  constructor(
    private readonly userDataDir: string,
    private readonly emitProgress: (progress: HubProgress) => void,
  ) {}

  async search(rawRequest: unknown): Promise<HubSearchResult> {
    const request = isRecord(rawRequest) ? rawRequest : null;
    if (!request || !['huggingface', 'ollama', 'openai-compatible'].includes(String(request.provider))) {
      throw new Error('Provider Hub non valido');
    }
    const provider = request.provider as HubProvider;
    const query = typeof request.query === 'string' ? request.query.trim().slice(0, 256) : '';
    const limit = Math.max(1, Math.min(MAX_HUB_RESULTS, Number(request.limit) || 18));
    const started = performance.now();
    try {
      let models: HubModel[];
      if (provider === 'huggingface') models = await searchHuggingFace(query, limit);
      else if (provider === 'ollama') models = await searchOllama(query, normaliseBaseUrl(request.baseUrl || DEFAULT_OLLAMA), limit);
      else models = await searchOpenAI(query, normaliseBaseUrl(request.baseUrl || DEFAULT_GATEWAY), limit);
      return { provider, models, latencyMs: Math.round(performance.now() - started) };
    } catch (error) {
      return { provider, models: [], latencyMs: Math.round(performance.now() - started), error: error instanceof Error ? error.message : 'Ricerca provider fallita' };
    }
  }

  private validatePullRequest(rawRequest: unknown): HubPullRequest {
    const request = isRecord(rawRequest) ? rawRequest : null;
    if (!request || !['huggingface', 'ollama', 'openai-compatible'].includes(String(request.provider)) || typeof request.modelId !== 'string' || request.modelId.length > 256) {
      throw new Error('Richiesta pull non valida');
    }
    const provider = request.provider as HubProvider;
    const fileRecord = isRecord(request.file) ? request.file : null;
    const file: HubFile | undefined = fileRecord && typeof fileRecord.path === 'string'
      ? { path: fileRecord.path, size: asNumber(fileRecord.size), downloadUrl: typeof fileRecord.downloadUrl === 'string' ? fileRecord.downloadUrl : undefined }
      : undefined;
    return { provider, modelId: request.modelId, baseUrl: typeof request.baseUrl === 'string' ? request.baseUrl : undefined, file };
  }

  startPull(rawRequest: unknown): { operationId: string } {
    const request = this.validatePullRequest(rawRequest);
    const operationId = crypto.randomUUID();
    const controller = new AbortController();
    this.activePulls.set(operationId, controller);
    void this.runPull(operationId, request, controller.signal);
    return { operationId };
  }

  cancelPull(operationId: unknown): boolean {
    if (typeof operationId !== 'string') return false;
    const controller = this.activePulls.get(operationId);
    if (!controller) return false;
    controller.abort();
    return true;
  }

  private async downloadHuggingFace(operationId: string, request: HubPullRequest, signal: AbortSignal): Promise<string> {
    const url = request.file?.downloadUrl;
    if (!url || !url.startsWith('https://huggingface.co/')) throw new Error('File Hugging Face non selezionabile');
    const response = await fetch(url, { signal, headers: process.env.HUGGINGFACE_TOKEN ? { Authorization: `Bearer ${process.env.HUGGINGFACE_TOKEN}` } : undefined });
    if (!response.ok || !response.body) throw new Error(`Download Hugging Face HTTP ${response.status}`);
    const targetDir = path.join(this.userDataDir, 'models', 'inbox');
    await mkdir(targetDir, { recursive: true });
    const name = `${request.modelId.replace(/[^a-zA-Z0-9._-]+/g, '_')}-${path.basename(new URL(url).pathname)}`;
    const target = path.join(targetDir, name.slice(0, 220));
    const partial = `${target}.${operationId}.part`;
    const output = fs.createWriteStream(partial);
    const reader = response.body.getReader();
    const total = Number(response.headers.get('content-length') ?? request.file?.size ?? 0) || undefined;
    let downloaded = 0;
    try {
      this.emitProgress({ operationId, provider: 'huggingface', modelId: request.modelId, phase: 'downloading', downloaded, total, percent: 0, status: 'Download avviato' });
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        const buffer = Buffer.from(chunk.value);
        downloaded += buffer.length;
        if (!output.write(buffer)) await once(output, 'drain');
        this.emitProgress({
          operationId, provider: 'huggingface', modelId: request.modelId, phase: 'downloading',
          downloaded, total, percent: total ? Math.min(99, (downloaded / total) * 100) : undefined, status: 'Download in corso',
        });
      }
      await new Promise<void>((resolve, reject) => { output.once('finish', resolve); output.once('error', reject); output.end(); });
      await rename(partial, target);
      return target;
    } catch (error) {
      output.destroy();
      await unlink(partial).catch(() => undefined);
      throw error;
    }
  }

  private async pullOllama(operationId: string, request: HubPullRequest, signal: AbortSignal): Promise<void> {
    const baseUrl = normaliseBaseUrl(request.baseUrl || DEFAULT_OLLAMA);
    const response = await fetch(`${baseUrl}/api/pull`, {
      method: 'POST',
      signal,
      headers: { 'Content-Type': 'application/json', Accept: 'application/x-ndjson' },
      body: JSON.stringify({ model: request.modelId, stream: true }),
    });
    if (!response.ok || !response.body) throw new Error(`Ollama HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    const consume = (line: string): void => {
      if (!line.trim()) return;
      try {
        const parsed: unknown = JSON.parse(line);
        const data = isRecord(parsed) ? parsed : null;
        if (!data) return;
        const total = asNumber(data.total);
        const completed = asNumber(data.completed) ?? 0;
        const status = typeof data.status === 'string' ? data.status : 'Pull in corso';
        this.emitProgress({
          operationId, provider: 'ollama', modelId: request.modelId,
          phase: status === 'success' ? 'complete' : 'downloading',
          downloaded: completed, total, percent: total ? (completed / total) * 100 : undefined, status,
        });
      } catch {
        this.emitProgress({ operationId, provider: 'ollama', modelId: request.modelId, phase: 'downloading', downloaded: 0, status: 'Risposta NDJSON non riconosciuta' });
      }
    };
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) consume(line);
    }
    buffer += decoder.decode();
    for (const line of buffer.split('\n')) consume(line);
  }

  private async runPull(operationId: string, request: HubPullRequest, signal: AbortSignal): Promise<void> {
    this.emitProgress({ operationId, provider: request.provider, modelId: request.modelId, phase: 'starting', downloaded: 0, status: 'Preparazione' });
    try {
      let localPath: string | undefined;
      if (request.provider === 'huggingface') localPath = await this.downloadHuggingFace(operationId, request, signal);
      else if (request.provider === 'ollama') await this.pullOllama(operationId, request, signal);
      else throw new Error('Il registry OpenAI-compatible espone modelli ma non gestisce download');
      this.emitProgress({ operationId, provider: request.provider, modelId: request.modelId, phase: 'complete', downloaded: 1, percent: 100, status: 'Completato', localPath });
    } catch (error) {
      const cancelled = signal.aborted;
      this.emitProgress({
        operationId, provider: request.provider, modelId: request.modelId,
        phase: cancelled ? 'cancelled' : 'error',
        downloaded: 0,
        error: cancelled ? 'Operazione annullata' : error instanceof Error ? error.message : 'Pull fallito',
        status: cancelled ? 'Annullato' : 'Errore',
      });
    } finally {
      this.activePulls.delete(operationId);
    }
  }

  async listLocalModelFiles(): Promise<string[]> {
    const inbox = path.join(this.userDataDir, 'models', 'inbox');
    try {
      const names = await readdir(inbox, { withFileTypes: true });
      return names
        .filter((entry) => entry.isFile() && /\.(pai|gguf|safetensors|bin)$/i.test(entry.name))
        .map((entry) => path.join(inbox, entry.name));
    } catch {
      return [];
    }
  }
}
