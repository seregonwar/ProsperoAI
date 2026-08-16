/*
 * ProsperoAI Desktop — gateway HTTP client (§26).
 *
 * Talks to the local OpenAI-compatible gateway (`pai serve`): health,
 * model listing with real metadata (kind/size/endpoint/broken), chat
 * completions and SSE streaming. All requests run in the main process
 * so the renderer is never exposed to CORS.
 */

import type { ChatMessage, ChatRequestParams, ChatStreamDelta, GatewayModel, GatewayStatus } from '../../shared/types';

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;

export type JsonRecord = Record<string, unknown>;

export function isRecord(value: unknown): value is JsonRecord {
  return value !== null && typeof value === 'object';
}

export function asNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

export function isLocalOrPrivateHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host === '::1' || host === '127.0.0.1') return true;
  const octets = host.split('.').map(Number);
  if (octets.length !== 4 || octets.some((octet) => !Number.isInteger(octet) || octet < 0 || octet > 255)) return false;
  return octets[0] === 10 || (octets[0] === 192 && octets[1] === 168) || (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31);
}

export function normaliseBaseUrl(value: unknown): string {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > 512) {
    throw new Error('Gateway URL non valida');
  }
  const url = new URL(value.trim());
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new Error('Il provider deve usare HTTP o HTTPS');
  }
  if (url.protocol === 'http:' && !isLocalOrPrivateHost(url.hostname)) {
    throw new Error('I registry HTTP devono essere locali o su una rete privata; usa HTTPS per endpoint pubblici');
  }
  if (url.username || url.password || url.search || url.hash) {
    throw new Error('L’URL non può contenere credenziali, query o fragment');
  }
  return url.toString().replace(/\/$/, '');
}

export async function requestJson(baseUrl: string, endpoint: string, init?: RequestInit, timeoutMs = 5000): Promise<unknown> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: 'application/json', ...(init?.headers ?? {}) },
    });
    const declaredLength = Number(response.headers.get('content-length') ?? 0);
    if (declaredLength > MAX_RESPONSE_BYTES) throw new Error('Risposta del provider troppo grande');
    const text = await response.text();
    if (text.length > MAX_RESPONSE_BYTES) throw new Error('Risposta del provider troppo grande');
    let body: unknown = null;
    if (text.length > 0) {
      try { body = JSON.parse(text); } catch { body = text; }
    }
    if (!response.ok) throw new Error(`Provider HTTP ${response.status}`);
    return body;
  } finally {
    clearTimeout(timer);
  }
}

/* Parse the real gateway model objects (id + kind/size/endpoint). */
export function parseModels(value: unknown): GatewayModel[] {
  const record = isRecord(value) ? value : null;
  if (!record || !Array.isArray(record.data)) return [];
  return record.data.filter(isRecord).map((item) => ({
    id: typeof item.id === 'string' ? item.id : 'unknown-model',
    object: typeof item.object === 'string' ? item.object : undefined,
    owned_by: typeof item.owned_by === 'string' ? item.owned_by : undefined,
    kind: item.kind === 'remote' || item.kind === 'local' ? item.kind : undefined,
    sizeBytes: asNumber(item.size_bytes),
    endpoint: typeof item.endpoint === 'string' ? item.endpoint : undefined,
    broken: typeof item.broken === 'boolean' ? item.broken : undefined,
  }));
}

export function validateMessages(value: unknown): ChatMessage[] {
  if (!Array.isArray(value) || value.length === 0 || value.length > 100) throw new Error('Messaggi non validi');
  return value.map((message) => {
    const candidate = isRecord(message) ? message : null;
    if (!candidate || !['system', 'user', 'assistant'].includes(String(candidate.role)) || typeof candidate.content !== 'string') {
      throw new Error('Messaggio non valido');
    }
    return { role: candidate.role as ChatMessage['role'], content: candidate.content.slice(0, 32_000) };
  });
}

export interface GatewayCheckResult extends GatewayStatus { }

export async function checkGateway(rawBaseUrl: unknown): Promise<GatewayCheckResult> {
  const baseUrl = normaliseBaseUrl(rawBaseUrl);
  const started = performance.now();
  try {
    await requestJson(baseUrl, '/healthz');
    let models: GatewayModel[] = [];
    try { models = parseModels(await requestJson(baseUrl, '/v1/models')); } catch { /* healthy without registry is valid */ }
    return { ok: true, baseUrl, latencyMs: Math.round(performance.now() - started), models };
  } catch (error) {
    return {
      ok: false,
      baseUrl,
      latencyMs: Math.round(performance.now() - started),
      models: [],
      error: error instanceof Error ? error.message : 'Gateway non raggiungibile',
    };
  }
}

/* Serialize the sampling parameters a request can carry (§26):
 * temperature, max_tokens, top_p, top_k, seed and stop sequences
 * (validated to the gateway's contract: up to 4 strings, 1..64 chars
 * each, empty sequences dropped). */
export function samplingBody(params: ChatRequestParams): JsonRecord {
  const body: JsonRecord = {};
  /* Defensive clamps to the gateway's §26 contract so a caller can
   * never produce a 400: temperature 0..2, top_p (0,1], top_k 1..200,
   * max_tokens 1..2048, and stop sequences trimmed to 1..64 chars. */
  if (typeof params.temperature === 'number' && Number.isFinite(params.temperature)) body.temperature = Math.max(0, Math.min(2, params.temperature));
  if (typeof params.maxTokens === 'number' && Number.isFinite(params.maxTokens)) body.max_tokens = Math.max(1, Math.min(2048, Math.floor(params.maxTokens)));
  if (typeof params.topP === 'number' && Number.isFinite(params.topP)) body.top_p = Math.max(0.05, Math.min(1, params.topP));
  if (typeof params.topK === 'number' && Number.isFinite(params.topK) && params.topK > 0) body.top_k = Math.floor(Math.min(200, params.topK));
  if (typeof params.seed === 'number' && Number.isFinite(params.seed) && params.seed >= 0) body.seed = Math.floor(params.seed);
  if (Array.isArray(params.stop)) {
    const stops = params.stop
      .map((item) => String(item).slice(0, 64))
      .filter((item) => item.length > 0)
      .slice(0, 4);
    if (stops.length > 0) body.stop = stops;
  }
  return body;
}

export async function completeChat(rawBaseUrl: unknown, rawModel: unknown, rawMessages: unknown, rawParams: unknown): Promise<string> {
  const baseUrl = normaliseBaseUrl(rawBaseUrl);
  if (typeof rawModel !== 'string' || rawModel.length === 0 || rawModel.length > 256) throw new Error('Modello non valido');
  const params = (isRecord(rawParams) ? rawParams : {}) as ChatRequestParams;
  const body: JsonRecord = {
    model: rawModel,
    messages: validateMessages(rawMessages),
    stream: false,
    ...samplingBody(params),
  };
  const value = await requestJson(baseUrl, '/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }, 120_000);
  const result = isRecord(value) ? value : null;
  const choices = result?.choices;
  const firstChoice = Array.isArray(choices) ? choices[0] : undefined;
  const choice = isRecord(firstChoice) ? firstChoice : null;
  if (!choice) throw new Error('Il gateway non ha restituito una completion');
  const message = isRecord(choice.message) ? choice.message : null;
  if (!message || typeof message.content !== 'string') throw new Error('Completion senza contenuto');
  return message.content;
}

export interface StreamChatOptions {
  onDelta?: (delta: ChatStreamDelta) => void;
  signal?: AbortSignal;
}

/*
 * Streaming chat over SSE (`data: {...}\n\n`, terminal `data: [DONE]`).
 * Each `data:` JSON payload is a chat.completion.chunk whose
 * choices[0].delta.content is one token. Resolves with the token
 * count; rejects on HTTP errors, aborts and mid-stream error events.
 */
export interface StreamChatResult {
  tokens: number;
  finishReason?: string;
  usage?: { promptTokens: number; completionTokens: number; totalTokens: number };
}

export async function streamChat(
  rawBaseUrl: unknown,
  rawModel: unknown,
  rawMessages: unknown,
  rawParams: unknown,
  requestId: string,
  options: StreamChatOptions = {},
): Promise<StreamChatResult> {
  const baseUrl = normaliseBaseUrl(rawBaseUrl);
  if (typeof rawModel !== 'string' || rawModel.length === 0 || rawModel.length > 256) throw new Error('Modello non valido');
  const params = (isRecord(rawParams) ? rawParams : {}) as ChatRequestParams;
  const body: JsonRecord = {
    model: rawModel,
    messages: validateMessages(rawMessages),
    stream: true,
    ...samplingBody(params),
  };

  const controller = new AbortController();
  const onAbort = () => controller.abort();
  options.signal?.addEventListener('abort', onAbort);
  try {
    const response = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!response.ok || !response.body) throw new Error(`Gateway HTTP ${response.status}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let tokens = 0;
    let finished = false;
    let midStreamError: Error | null = null;
    /* §26 extras surfaced to the caller: the finish_reason of the
     * terminal chunk and the usage totals of the optional usage
     * chunk (stream_options.include_usage). */
    let finishReason: string | undefined;
    let usage: { promptTokens: number; completionTokens: number; totalTokens: number } | undefined;
    try {
      while (!finished) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        let newline: number;
        while ((newline = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          if (line.startsWith('data:')) {
            const data = line.slice(5).trim();
            if (data === '[DONE]') {
              finished = true;
              break;
            }
            if (data.length === 0) continue;
            try {
              const payload = JSON.parse(data) as JsonRecord;
              /* Mid-stream failures are surfaced as SSE error events
               * before [DONE] (the gateway's §26 contract). */
              if (isRecord(payload.error)) {
                const message = typeof payload.error.message === 'string'
                  ? payload.error.message
                  : 'errore mid-stream dal gateway';
                midStreamError = new Error(message);
                finished = true;
                break;
              }
              const choices = payload.choices;
              const firstChoice = Array.isArray(choices) ? choices[0] : undefined;
              const choice = isRecord(firstChoice) ? firstChoice : null;
              const delta = isRecord(choice?.delta) ? choice.delta : null;
              const token = typeof delta?.content === 'string' ? delta.content : '';
              if (token.length > 0) {
                tokens++;
                options.onDelta?.({ requestId, token, done: false });
              }
              /* The terminal chunk carries the finish_reason; the
               * optional usage chunk carries empty choices + the
               * usage totals (both are single, final chunks). */
              if (typeof choice?.finish_reason === 'string') finishReason = choice.finish_reason;
              const rawUsage = isRecord(payload.usage) ? payload.usage : null;
              if (rawUsage && typeof rawUsage.prompt_tokens === 'number' && typeof rawUsage.completion_tokens === 'number') {
                const promptTokens = rawUsage.prompt_tokens;
                const completionTokens = rawUsage.completion_tokens;
                usage = { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens };
              }
            } catch {
              /* ignore malformed chunks */
            }
          }
        }
      }
      if (midStreamError) throw midStreamError;
      return { tokens, finishReason, usage };
    } finally {
      /* Release the wire reader on every path (mid-stream errors
       * included) so no stream handle leaks into process exit. */
      await reader.cancel().catch(() => undefined);
    }
  } finally {
    options.signal?.removeEventListener('abort', onAbort);
  }
}
