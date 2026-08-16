/*
 * ProsperoAI Desktop — renderer bridge bootstrap.
 *
 * Inside Electron the `prospero` bridge comes from the preload. When
 * the renderer runs in a plain browser (vite preview / development),
 * the bridge is stubbed with an honest read-only implementation: every
 * view shows its real empty state instead of crashing or fabricating
 * data.
 */

import type { HubModel, ProsperoApi } from '../../shared/types';

/* ------------------------------------------------------------------ */
/* Visual preview fixture (browser only, opt-in via #hub-preview).     */
/*                                                                     */
/* The real Model Hub search runs in the Electron main process, so a   */
/* plain-browser preview cannot show result cards. Opening the dev     */
/* server with `#hub-preview` in the URL swaps in this deterministic   */
/* fixture — real model ids/families, clearly shaped like live API     */
/* responses — purely to inspect card rendering. It is unreachable in  */
/* Electron (the preload bridge wins there) and ships only with the    */
/* dev server, never in the packaged app data path.                    */
/* ------------------------------------------------------------------ */

const HUB_PREVIEW_MODELS: HubModel[] = [
  {
    id: 'meta-llama/Llama-3.1-8B-Instruct', name: 'Llama-3.1-8B-Instruct', provider: 'huggingface',
    author: 'meta-llama', description: 'llama · text-generation · instruct · english · 8b · 128k',
    downloads: 48200000, likes: 12800, tags: ['llama', 'text-generation', 'instruct', 'english', '8b', '128k'],
    family: 'llama', quantization: 'Q4_K_M',
    files: [{ path: 'Llama-3.1-8B-Instruct-Q4_K_M.gguf', size: 4919000000, downloadUrl: 'https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'Qwen/Qwen2.5-7B-Instruct', name: 'Qwen2.5-7B-Instruct', provider: 'huggingface',
    author: 'Qwen', description: 'qwen2 · text-generation · instruct · multilingual · 7b · 32k',
    downloads: 9160000, likes: 4100, tags: ['qwen2', 'text-generation', 'instruct', 'multilingual', '7b'],
    family: 'qwen',
    files: [
      { path: 'ggml-model-Q4_K_M.gguf', size: 4681000000, downloadUrl: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/ggml-model-Q4_K_M.gguf' },
      { path: 'config.json', size: 640 },
    ],
    pullable: true, local: false,
  },
  {
    id: 'mistralai/Mistral-7B-Instruct-v0.3', name: 'Mistral-7B-Instruct-v0.3', provider: 'huggingface',
    author: 'mistralai', description: 'mistral · text-generation · instruct · english · 7b · 32k',
    downloads: 7140000, likes: 2900, tags: ['mistral', 'text-generation', 'instruct', 'english', '7b'],
    family: 'mistral', quantization: 'Q5_K_M',
    files: [{ path: 'mistral-7b-instruct-v0.3.Q5_K_M.gguf', size: 5270000000, downloadUrl: 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/mistral-7b-instruct-v0.3.Q5_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'google/gemma-2-9b-it', name: 'gemma-2-9b-it', provider: 'huggingface',
    author: 'google', description: 'gemma2 · text-generation · instruct · multilingual · 9b · 8k',
    downloads: 4890000, likes: 1900, tags: ['gemma2', 'text-generation', 'instruct', 'multilingual', '9b'],
    family: 'gemma',
    files: [{ path: 'gemma-2-9b-it-Q4_K_M.gguf', size: 5580000000, downloadUrl: 'https://huggingface.co/google/gemma-2-9b-it/resolve/main/gemma-2-9b-it-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'microsoft/Phi-3.5-mini-instruct', name: 'Phi-3.5-mini-instruct', provider: 'huggingface',
    author: 'microsoft', description: 'phi3 · text-generation · instruct · english · 3.8b · 128k',
    downloads: 2260000, likes: 900, tags: ['phi3', 'text-generation', 'instruct', 'english'],
    family: 'phi', quantization: 'Q4_K_M',
    files: [{ path: 'Phi-3.5-mini-instruct-Q4_K_M.gguf', size: 2330000000, downloadUrl: 'https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'deepseek-ai/DeepSeek-V2.5', name: 'DeepSeek-V2.5', provider: 'huggingface',
    author: 'deepseek-ai', description: 'deepseek · text-generation · expert · multilingual · 236b · 128k',
    downloads: 1180000, likes: 700, tags: ['deepseek', 'text-generation', 'expert', 'multilingual'],
    family: 'deepseek',
    files: [{ path: 'DeepSeek-V2.5-Q4_K_M.gguf', size: 124000000000, downloadUrl: 'https://huggingface.co/deepseek-ai/DeepSeek-V2.5/resolve/main/DeepSeek-V2.5-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'ibm-granite/granite-3.0-8b-instruct', name: 'granite-3.0-8b-instruct', provider: 'huggingface',
    author: 'ibm-granite', description: 'granite · text-generation · instruct · english · 8b · 128k',
    downloads: 940000, likes: 310, tags: ['granite', 'text-generation', 'instruct'],
    family: 'granite',
    files: [{ path: 'granite-3.0-8b-instruct-Q4_K_M.gguf', size: 5100000000, downloadUrl: 'https://huggingface.co/ibm-granite/granite-3.0-8b-instruct/resolve/main/granite-3.0-8b-instruct-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'HuggingFaceTB/SmolLM2-1.7B-Instruct', name: 'SmolLM2-1.7B-Instruct', provider: 'huggingface',
    author: 'HuggingFaceTB', description: 'smollm2 · text-generation · instruct · english · 1.7b · 8k',
    downloads: 760000, likes: 280, tags: ['smollm2', 'text-generation', 'instruct', 'english'],
    files: [{ path: 'SmolLM2-1.7B-Instruct-Q4_K_M.gguf', size: 1090000000, downloadUrl: 'https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf' }],
    pullable: true, local: false,
  },
  {
    id: 'llama3.2:3b', name: 'llama3.2:3b', provider: 'ollama', local: true, pullable: true,
    size: 2000000000, family: 'llama', quantization: 'Q4_K_M', tags: [], files: [],
  },
  {
    id: 'gw_tiny', name: 'gw_tiny', provider: 'openai-compatible', author: 'prosperoai',
    local: false, pullable: false, tags: ['OpenAI-compatible'], files: [],
  },
];

const PREVIEW_SETTINGS = {
  gatewayUrl: 'http://127.0.0.1:8080',
  launchAtStartup: false,
  expertMode: false,
  reduceMotion: false,
};

function noopApi(): ProsperoApi {
  return {
    getAppInfo: async () => ({ version: '0.1.0', platform: 'browser-preview', isPackaged: false }),
    getSettings: async () => ({ ...PREVIEW_SETTINGS }),
    setSettings: async (patch) => ({ ...PREVIEW_SETTINGS, ...patch }),
    checkGateway: async (baseUrl) => ({
      ok: false,
      baseUrl: baseUrl ?? PREVIEW_SETTINGS.gatewayUrl,
      models: [],
      error: 'anteprima browser: avvia l’app Electron per parlare col gateway',
    }),
    completeChat: async () => { throw new Error('anteprima browser: inference non disponibile'); },
    streamChat: async () => ({ ok: false, error: 'anteprima browser: inference non disponibile', tokens: 0 }),
    cancelChatStream: () => undefined,
    listDevices: async () => [],
    addDevice: async () => [],
    removeDevice: async () => [],
    probeDevices: async () => [],
    listEvents: async () => [],
    logEvent: async () => undefined,
    exportDiagnostics: async () => null,
    onEvent: () => () => undefined,
    listLibrary: async () => [],
    importModels: async () => ({ queued: 0, entries: [] }),
    runBenchmark: async () => { throw new Error('anteprima browser: benchmark non disponibile'); },
    listBenchmarks: async () => [],
    deleteBenchmark: async () => [],
    selectModelFiles: async () => [],
    listLocalModelFiles: async () => [],
    searchHub: async (request) => {
      if (import.meta.env.DEV && window.location.hash === '#hub-preview') {
        /* Preview fixture: deterministic cards to inspect the layout. */
        const query = request.query.toLowerCase();
        const models = HUB_PREVIEW_MODELS
          .filter((model) => model.provider === request.provider)
          .filter((model) => !query || model.id.toLowerCase().includes(query) || (model.family ?? '').toLowerCase().includes(query))
          .slice(0, request.limit ?? 18);
        return { provider: request.provider, models, latencyMs: 41, error: models.length ? undefined : 'Nessun risultato (fixture di anteprima)' };
      }
      return { provider: request.provider, models: [], latencyMs: 0, error: 'anteprima browser: avvia l’app Electron per il Model Hub' };
    },
    startHubPull: async () => { throw new Error('anteprima browser: pull non disponibile'); },
    cancelHubPull: async () => false,
    onHubProgress: () => () => undefined,
  };
}

export function ensureBridge(): void {
  const holder = window as unknown as { prospero?: ProsperoApi };
  if (!holder.prospero) {
    holder.prospero = noopApi();
  }
}
