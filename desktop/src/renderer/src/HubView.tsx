import { useEffect, useMemo, useState } from 'react';
import type { HubModel, HubProgress, HubProvider } from '../../shared/types';
import { Icon, formatBytes, formatCount } from './components';
import { FamilyLogo, detectFamily } from './familyLogos';

type HubViewProps = {
  gatewayUrl: string;
  onImport: () => void;
  onNavigate: (view: 'playground' | 'models') => void;
  onLog: (message: string, level?: 'INFO' | 'TRACE' | 'WARN' | 'ERROR') => void;
};

type ProviderInfo = { id: HubProvider; name: string; short: string; description: string; tone: string; endpoint: string };

const providers: ProviderInfo[] = [
  { id: 'huggingface', name: 'Hugging Face Hub', short: 'HF', description: 'Search the public Hub and download GGUF / safetensors artifacts.', tone: 'violet', endpoint: 'huggingface.co' },
  { id: 'ollama', name: 'Ollama', short: 'O', description: 'Inspect the local daemon and pull any library tag by name.', tone: 'cyan', endpoint: '127.0.0.1:11434' },
  { id: 'openai-compatible', name: 'OpenAI-compatible', short: 'API', description: 'Discover models exposed by a local or custom gateway.', tone: 'orange', endpoint: 'custom endpoint' },
];

function modelKey(model: Pick<HubModel, 'provider' | 'id'>): string {
  return `${model.provider}:${model.id}`;
}

function progressLabel(progress?: HubProgress): string {
  if (!progress) return '';
  if (progress.phase === 'complete') return progress.localPath ? 'Saved to local inbox' : 'Ready';
  if (progress.phase === 'error') return progress.error ?? 'Operation failed';
  if (progress.phase === 'cancelled') return 'Cancelled';
  return progress.status ?? 'Working';
}

export function HubView({ gatewayUrl, onImport, onNavigate, onLog }: HubViewProps) {
  const [provider, setProvider] = useState<HubProvider>('huggingface');
  const [query, setQuery] = useState('llama');
  const [customUrl, setCustomUrl] = useState(gatewayUrl);
  const [ollamaUrl, setOllamaUrl] = useState('http://127.0.0.1:11434');
  const [results, setResults] = useState<HubModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [latency, setLatency] = useState<number>();
  const [pulls, setPulls] = useState<Record<string, { operationId: string; progress?: HubProgress }>>({});

  const selectedProvider = useMemo(() => providers.find((item) => item.id === provider) ?? providers[0], [provider]);

  useEffect(() => {
    if (!window.prospero) return undefined;
    return window.prospero.onHubProgress((progress) => {
      setPulls((current) => {
        const key = `${progress.provider}:${progress.modelId}`;
        const currentPull = current[key];
        if (!currentPull || currentPull.operationId !== progress.operationId) return current;
        return { ...current, [key]: { ...currentPull, progress } };
      });
      if (progress.phase === 'complete') onLog(`${progress.provider} pull completed · ${progress.modelId}`);
      if (progress.phase === 'error') onLog(`${progress.provider} pull failed · ${progress.error ?? progress.modelId}`, 'ERROR');
    });
  }, [onLog]);

  async function search() {
    setLoading(true);
    setError('');
    onLog(`Hub search · ${provider} · ${query || 'trending'}`, 'TRACE');
    try {
      if (!window.prospero) throw new Error('Model Hub disponibile nell’app Electron, non nell’anteprima browser');
      const result = await window.prospero.searchHub({ provider, query, limit: 18, baseUrl: provider === 'openai-compatible' ? customUrl : provider === 'ollama' ? ollamaUrl : undefined });
      setResults(result.models);
      setLatency(result.latencyMs);
      if (result.error) setError(result.error);
      onLog(`Hub returned ${result.models.length} result${result.models.length === 1 ? '' : 's'} · ${result.latencyMs} ms`);
    } catch (reason) {
      setResults([]);
      const message = reason instanceof Error ? reason.message : 'Provider search failed';
      setError(message);
      onLog(message, 'ERROR');
    } finally {
      setLoading(false);
    }
  }

  async function pull(model: HubModel, file = model.files[0]) {
    if (!window.prospero) return;
    setError('');
    try {
      const started = await window.prospero.startHubPull({ provider: model.provider, modelId: model.id, file, baseUrl: model.provider === 'openai-compatible' ? customUrl : model.provider === 'ollama' ? ollamaUrl : undefined });
      setPulls((current) => ({ ...current, [modelKey(model)]: { operationId: started.operationId } }));
      onLog(`${model.provider} pull started · ${model.id}`, 'INFO');
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : 'Unable to start pull';
      setError(message);
      onLog(message, 'ERROR');
    }
  }

  async function cancel(model: HubModel) {
    const active = pulls[modelKey(model)];
    if (!active || !window.prospero) return;
    await window.prospero.cancelHubPull(active.operationId);
  }

  return <>
    <div className="hub-heading section-header">
      <div><div className="eyebrow">Model Hub / provider adapters</div><h1>Find your next model.</h1><p>Search, pull and prepare models without leaving the Prospero workspace.</p></div>
      <div className="hub-actions"><button className="button secondary" onClick={onImport}><Icon name="folder" size={15} /> Import local</button><button className="button primary" onClick={() => void search()} disabled={loading}><Icon name="search" size={15} /> {loading ? 'Searching...' : 'Search Hub'}</button></div>
    </div>
    <div className="provider-rail">{providers.map((item) => <button key={item.id} className={`provider-tile ${item.tone} ${provider === item.id ? 'selected' : ''}`} onClick={() => { setProvider(item.id); setResults([]); setError(''); }}><span className="provider-logo">{item.short}</span><span><strong>{item.name}</strong><small>{item.description}</small></span><Icon name="chevron" size={15} /></button>)}</div>
    <section className="hub-search-panel panel"><div className="hub-search-line"><div className="hub-search-input"><Icon name="search" size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void search(); }} placeholder={provider === 'ollama' ? 'Search local tags or type a library tag, e.g. llama3.2' : 'Search models, families or authors...'} /></div><button className="button primary" onClick={() => void search()} disabled={loading}>{loading ? 'Working...' : 'Search'}</button></div><div className="hub-search-meta"><span><span className="status-dot online" /> Adapter ready · {selectedProvider.endpoint}</span>{latency !== undefined && <span>Last search {latency} ms</span>}</div>{provider === 'openai-compatible' && <label className="hub-endpoint">Registry base URL<input value={customUrl} onChange={(event) => setCustomUrl(event.target.value)} placeholder="http://127.0.0.1:8080" /></label>}{provider === 'ollama' && <label className="hub-endpoint">Ollama daemon URL<input value={ollamaUrl} onChange={(event) => setOllamaUrl(event.target.value)} placeholder="http://127.0.0.1:11434" /></label>}</section>
    {error && <div className="hub-error"><Icon name="pulse" size={16} /><div><strong>Provider response</strong><span>{error}</span></div><button className="icon-button" onClick={() => setError('')}>×</button></div>}
    <div className="hub-results-heading"><div><span className="eyebrow">{results.length ? `${results.length} results` : 'Ready when you are'}</span><h2>{results.length ? 'Explore models' : 'One workspace, every source'}</h2></div>{results.length > 0 && <button className="ghost-button" onClick={() => setResults([])}>Clear results</button>}</div>
    {results.length === 0 ? <div className="hub-empty"><div className="hub-empty-orbit"><Icon name="spark" size={29} /></div><h3>Search across your model sources</h3><p>Hugging Face searches the public Hub, Ollama checks your local daemon and accepts exact library tags, while OpenAI-compatible discovers models already served on your network.</p><div className="hub-flow"><span>1. Choose provider</span><Icon name="arrow" size={14} /><span>2. Search</span><Icon name="arrow" size={14} /><span>3. Pull or run</span></div></div> : <div className="hub-results">{results.map((model) => <HubResultCard key={`${model.provider}:${model.id}`} model={model} pull={pull} cancel={cancel} progress={pulls[modelKey(model)]?.progress} onNavigate={onNavigate} />)}</div>}
    <div className="hub-footnote"><Icon name="shield" size={15} /><span>Downloads run in the Electron main process. Hugging Face public search works without a token; gated/private repositories use <code>HUGGINGFACE_TOKEN</code>. Ollama stays local.</span></div>
  </>;
}

function HubResultCard({ model, pull, cancel, progress, onNavigate }: { model: HubModel; pull: (model: HubModel, file?: HubModel['files'][number]) => void; cancel: (model: HubModel) => void; progress?: HubProgress; onNavigate: (view: 'playground' | 'models') => void }) {
  const [selectedFile, setSelectedFile] = useState(model.files[0]);
  const busy = Boolean(progress && ['starting', 'downloading'].includes(progress.phase));
  const canPull = model.pullable && !busy;
  const isOpenAI = model.provider === 'openai-compatible';
  const family = detectFamily(model);
  const providerLabel = model.provider === 'huggingface' ? 'HUGGING FACE' : model.provider === 'ollama' ? (model.local ? 'LOCAL OLLAMA' : 'OLLAMA LIBRARY') : 'OPENAI-COMPATIBLE';
  return <article className={`hub-result-card ${model.provider}`}><div className="hub-card-header"><FamilyLogo model={model} size={46} /><div className="hub-card-heading"><span className="hub-card-meta">{family.known ? `${family.label} · ${providerLabel}` : providerLabel}</span><h3 title={model.id}>{model.name}</h3><span className="hub-card-sub">{model.author ?? model.id}</span></div><button className="icon-button"><Icon name="more" size={17} /></button></div>{model.description && <p className="hub-description">{model.description}</p>}<div className="hub-result-tags">{model.quantization && <span>{model.quantization}</span>}{family.known && <span>{family.label}</span>}{model.tags.slice(0, 2).map((tag) => <span key={tag}>{tag}</span>)}</div><div className="hub-result-stats"><span><b>{model.provider === 'huggingface' ? formatCount(model.downloads) : model.local ? formatBytes(model.size) : 'library'}</b><small>{model.provider === 'huggingface' ? 'downloads' : model.local ? 'on disk' : 'source'}</small></span><span><b>{selectedFile ? formatBytes(selectedFile.size ?? model.size) : model.quantization ?? '—'}</b><small>{model.provider === 'ollama' ? 'format' : 'artifact'}</small></span></div>{model.provider === 'huggingface' && model.files.length > 1 && <label className="hub-file-picker">Artifact<select value={selectedFile?.path ?? ''} onChange={(event) => setSelectedFile(model.files.find((file) => file.path === event.target.value) ?? model.files[0])}>{model.files.map((file) => <option value={file.path} key={file.path}>{file.path.split('/').pop()} · {formatBytes(file.size)}</option>)}</select></label>}{progress && <div className={`hub-progress ${progress.phase}`}><div className="hub-progress-line"><span>{progressLabel(progress)}</span><b>{progress.percent !== undefined ? `${Math.round(progress.percent)}%` : ''}</b></div><div className="progress-track"><i style={{ width: `${progress.percent ?? (progress.phase === 'complete' ? 100 : 4)}%` }} /></div>{progress.error && <small>{progress.error}</small>}</div>}<div className="hub-result-actions">{busy ? <button className="small-button muted" onClick={() => cancel(model)}>Cancel</button> : isOpenAI ? <button className="small-button" onClick={() => onNavigate('playground')}>Use in playground <Icon name="arrow" size={13} /></button> : <button className="small-button" disabled={!canPull} onClick={() => pull(model, selectedFile)}>{progress?.phase === 'complete' ? 'Pull again' : model.provider === 'ollama' ? 'Pull model' : selectedFile ? 'Download artifact' : 'No artifact'} <Icon name="download" size={13} /></button>}<button className="icon-button"><Icon name="chevron" size={15} /></button></div></article>;
}
