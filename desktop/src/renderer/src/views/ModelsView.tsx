import { useMemo, useState } from 'react';
import type { GatewayModel, InspectData, LibraryEntry, OptimizePlan } from '../../../shared/types';
import { EmptyState, Icon, SectionHeader, StatusDot, formatBytes, formatCount } from '../components';
import type { View } from './Overview';

interface ModelsViewProps {
  gatewayModels: GatewayModel[];
  library: LibraryEntry[];
  importing: boolean;
  onImport: () => void;
  onNavigate: (view: View) => void;
  expertMode: boolean;
}

type Filter = 'all' | 'ready' | 'remote' | 'library';

export function ModelsView({ gatewayModels, library, importing, onImport, onNavigate, expertMode }: ModelsViewProps) {
  const [query, setQuery] = useState('');
  const [filter, setFilter] = useState<Filter>('all');
  const [optimizingId, setOptimizingId] = useState<string | null>(null);
  const [plan, setPlan] = useState<OptimizePlan | null>(null);
  const [optimizeError, setOptimizeError] = useState<string | null>(null);
  const [inspectingId, setInspectingId] = useState<string | null>(null);
  const [inspect, setInspect] = useState<InspectData | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);

  async function runOptimize(entry: LibraryEntry): Promise<void> {
    setOptimizingId(entry.id);
    setOptimizeError(null);
    try {
      const result = await window.prospero.optimizeModel(entry.id);
      if (result.ok && result.plan) {
        setPlan(result.plan);
      } else {
        setPlan(null);
        setOptimizeError(result.error ?? 'ottimizzazione fallita');
      }
    } catch (error) {
      setPlan(null);
      setOptimizeError(error instanceof Error ? error.message : 'ottimizzazione fallita');
    } finally {
      setOptimizingId(null);
    }
  }

  const planSavings = plan && plan.currentBytes > 0
    ? 100 * (1 - plan.planBytes / plan.currentBytes)
    : 0;

  async function runInspect(entry: LibraryEntry): Promise<void> {
    setInspectingId(entry.id);
    setInspectError(null);
    try {
      const result = await window.prospero.inspectModel(entry.id);
      if (result.ok && result.data) {
        setInspect(result.data);
      } else {
        setInspect(null);
        setInspectError(result.error ?? 'lettura manifest fallita');
      }
    } catch (error) {
      setInspect(null);
      setInspectError(error instanceof Error ? error.message : 'lettura manifest fallita');
    } finally {
      setInspectingId(null);
    }
  }

  const all = useMemo(() => {
    const gateway = gatewayModels.map((model): GatewayModel & { source: 'gateway' } => ({ ...model, source: 'gateway' }));
    return gateway;
  }, [gatewayModels]);

  const readyCount = all.filter((model) => !model.broken).length;
  const remoteCount = all.filter((model) => model.kind === 'remote').length;
  const libraryCount = library.length;

  const filtered = all.filter((model) => {
    if (filter === 'remote' && model.kind !== 'remote') return false;
    if (filter === 'ready' && model.broken) return false;
    const haystack = `${model.id} ${model.kind ?? ''} ${model.endpoint ?? ''}`.toLowerCase();
    return haystack.includes(query.toLowerCase());
  });

  return <>
    <SectionHeader
      eyebrow="Model manager / repository"
      title="Model library"
      description="Modelli serviti dal gateway e file gestiti dalla libreria desktop (§7)."
      action={<button className="button primary" onClick={onImport} disabled={importing}><Icon name="plus" size={16} /> {importing ? 'Importing...' : 'Import model'}</button>}
    />

    <div className="toolbar">
      <div className="search-field"><Icon name="search" size={17} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Cerca tra gateway e libreria..." /></div>
      <div className="filter-pills">
        <button className={`filter-pill ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>Tutti <b>{all.length + libraryCount}</b></button>
        <button className={`filter-pill ${filter === 'ready' ? 'active' : ''}`} onClick={() => setFilter('ready')}>Pronti <b>{readyCount}</b></button>
        <button className={`filter-pill ${filter === 'remote' ? 'active' : ''}`} onClick={() => setFilter('remote')}>Remote <b>{remoteCount}</b></button>
      </div>
    </div>

    {all.length === 0 ? (
      <EmptyState
        icon="layers"
        title="Nessun modello servito dal gateway"
        description="Avvia il gateway con `pai serve <model.pai>` (o un endpoint --remote) e ricarica: la libreria riflette il registro reale."
        action={<button className="button secondary" onClick={() => onNavigate('playground')}><Icon name="refresh" size={15} /> Vai al playground</button>}
      />
    ) : (
      <div className="model-grid">
        {filtered.map((model) => (
          <article className={`model-card ${model.kind === 'remote' ? 'accent-green' : 'accent-violet'}`} key={model.id}>
            <div className="model-card-top">
              <span className="model-mark"><span>{model.kind === 'remote' ? 'R' : model.id.slice(0, 1).toUpperCase()}</span></span>
              <button className="icon-button"><Icon name="more" size={18} /></button>
            </div>
            <div className="model-card-title">
              <h3>{model.id}</h3>
              <span>{model.kind === 'remote' ? `payload ${model.endpoint ?? ''}` : 'container .pai locale'}</span>
            </div>
            <div className="model-meta">
              <span><b>Formato</b>{model.kind === 'remote' ? 'Protocol' : 'PAI'}</span>
              <span><b>Dimensione</b>{model.kind === 'remote' ? '—' : formatBytes(model.sizeBytes)}</span>
              <span><b>Stato</b>{model.broken ? 'broken' : 'ok'}</span>
            </div>
            <div className="model-card-bottom">
              <span className={`state-badge ${model.broken ? 'preparing' : model.kind === 'remote' ? 'remote' : 'ready'}`}>
                <StatusDot status={model.broken ? 'offline' : 'online'} />
                {model.broken ? 'Load failed' : model.kind === 'remote' ? 'Remote' : 'Ready'}
              </span>
              {model.broken
                ? <button className="small-button muted">Dettagli <Icon name="chevron" size={13} /></button>
                : <button className="small-button" onClick={() => onNavigate('playground')}>Run <Icon name="arrow" size={13} /></button>}
            </div>
          </article>
        ))}
        <button className="import-card" onClick={onImport} disabled={importing}>
          <span><Icon name="plus" size={22} /></span>
          <strong>Import a model</strong>
          <small>GGUF · safetensors · .pai</small>
        </button>
      </div>
    )}

    <section className="panel library-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Desktop library / userData</span><h3>File gestiti dalla pipeline (§8)</h3></div>
        <span className="secure-pill"><Icon name="terminal" size={14} /> {libraryCount} file in gestione</span>
      </div>
      {library.length === 0 ? (
        <div className="library-empty">Nessun file importato. Usa “Import model”: i file vengono copiati nella libreria e, se il CLI `pai` è disponibile, convertiti e validati davvero.</div>
      ) : (
        <div className="library-table">
          <div className="library-row library-head"><span>Modello</span><span>Formato</span><span>Dimensione</span><span>Stato</span><span>Pipeline</span></div>
          {library.map((entry) => (
            <div className="library-row" key={entry.id}>
              <span><strong>{entry.name}</strong><small>{entry.updatedAt.slice(0, 19).replace('T', ' ')}</small></span>
              <span className="library-kind">{entry.kind}</span>
              <span>{formatBytes(entry.sizeBytes)}</span>
              <span className={`state-badge ${entry.status === 'ready' ? 'ready' : entry.status === 'failed' ? 'preparing' : ''}`}>
                <StatusDot status={entry.status === 'ready' ? 'online' : entry.status === 'failed' ? 'offline' : 'checking'} />
                {entry.status}
              </span>
              <span className="library-message">
                {entry.message ?? (entry.status === 'ready' ? 'integrità verificata' : '—')}
                {entry.kind === 'pai' && entry.status === 'ready' && (
                  <>
                    <button
                      className="small-button"
                      style={{ marginLeft: 8 }}
                      onClick={() => void runOptimize(entry)}
                      disabled={optimizingId !== null}
                      title="Piano mixed-precision §15 (pai optimize)"
                    >
                      {optimizingId === entry.id ? 'Pianifico…' : 'Ottimizza'}
                    </button>
                    <button
                      className="small-button muted"
                      style={{ marginLeft: 6 }}
                      onClick={() => void runInspect(entry)}
                      disabled={inspectingId !== null}
                      title="Manifest del container §20 (pai inspect)"
                    >
                      {inspectingId === entry.id ? 'Leggo…' : 'Dettagli'}
                    </button>
                  </>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>

    {optimizeError && <div className="optimize-error"><Icon name="pulse" size={15} /> {optimizeError}</div>}

    {inspectError && <div className="optimize-error"><Icon name="pulse" size={15} /> {inspectError}</div>}

    {inspect && (
      <section className="panel inspect-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Container manifest · §20</span><h3>{inspect.meta?.name ?? inspect.path.split(/[\\/]/).pop() ?? 'Manifest'}</h3></div>
          <div className="optimize-actions">
            <span className="secure-pill"><Icon name="terminal" size={14} /> v{inspect.version} · {formatBytes(inspect.bytes)}</span>
            <button className="icon-button" onClick={() => setInspect(null)} title="Chiudi report"><Icon name="more" size={15} /></button>
          </div>
        </div>
        {inspect.meta && (
          <div className="inspect-meta">
            <div><strong>{inspect.meta.numLayers}</strong><span>layer</span></div>
            <div><strong>{formatCount(inspect.meta.contextLen)}</strong><span>context tokens</span></div>
            <div><strong>{formatCount(inspect.meta.vocabSize)}</strong><span>vocab</span></div>
            <div><strong>{formatBytes(inspect.meta.kvBytesPerToken)}</strong><span>KV / token</span></div>
            <div><strong>{inspect.meta.family}</strong><span>family</span></div>
          </div>
        )}
        <div className="inspect-cols">
          <div>
            <div className="panel-heading"><div><span className="eyebrow">Sezioni</span><h3>Container</h3></div></div>
            <div className="library-table">
              <div className="library-row library-head"><span>Sezione</span><span>Offset</span><span>Dimensione</span></div>
              {inspect.sections.map((section) => (
                <div className="library-row" key={section.type}>
                  <span><strong>{section.type}</strong></span>
                  <span className="library-kind">{section.offset} B</span>
                  <span>{formatBytes(section.size)}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="panel-heading"><div><span className="eyebrow">{inspect.tensors.length} tensor{inspect.tensors.length === 1 ? 'e' : 'i'}</span><h3>Manifest</h3></div></div>
            <div className="library-table">
              <div className="library-row library-head"><span>Tensor</span><span>Dtype</span><span>Shape</span><span>Peso</span></div>
              {[...inspect.tensors].sort((a, b) => b.sizeBytes - a.sizeBytes).slice(0, 8).map((tensor) => (
                <div className="library-row" key={tensor.name}>
                  <span><strong>{tensor.name}</strong></span>
                  <span className="library-kind">{tensor.dtype}</span>
                  <span className="inspect-shape">[{tensor.shape.join(',')}]</span>
                  <span>{formatBytes(tensor.sizeBytes)}</span>
                </div>
              ))}
              {inspect.tensors.length > 8 && (
                <div className="library-row"><span><small>…e altri {inspect.tensors.length - 8} tensor{inspect.tensors.length - 8 === 1 ? 'e' : 'i'}</small></span></div>
              )}
            </div>
          </div>
          {inspect.ir && (
            <div>
              <div className="panel-heading"><div><span className="eyebrow">Grafo compilato</span><h3>IR</h3></div></div>
              <div className="inspect-ir">
                <span>{inspect.ir.ops.length} op{inspect.ir.ops.length === 1 ? 'erazione' : 'erazioni'}</span>
                <span>{inspect.ir.values} valori</span>
                <span>{inspect.ir.inputs} in · {inspect.ir.outputs} out</span>
                {inspect.tokenizer && <span>tokenizer: {formatCount(inspect.tokenizer.tokens)} token · {formatCount(inspect.tokenizer.merges)} merge</span>}
              </div>
            </div>
          )}
        </div>
      </section>
    )}

    {plan && (
      <section className="panel optimize-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Mixed-precision planner · §15</span><h3>Piano di ottimizzazione · {plan.model}</h3></div>
          <div className="optimize-actions">
            <span className="secure-pill"><Icon name="chart" size={14} /> −{planSavings.toFixed(1)}% di peso</span>
            <button className="icon-button" onClick={() => setPlan(null)} title="Chiudi piano"><Icon name="more" size={15} /></button>
          </div>
        </div>
        <div className="optimize-stats">
          <div><strong>{formatBytes(plan.currentBytes)}</strong><span>peso attuale</span></div>
          <div><strong>{formatBytes(plan.planBytes)}</strong><span>piano §15</span></div>
          <div><strong>{formatBytes(Math.max(0, plan.currentBytes - plan.planBytes))}</strong><span>risparmio</span></div>
          {plan.budget ? (
            <div><strong>{formatBytes(plan.budget)}</strong><span>budget · {plan.feasible ? 'raggiunto' : 'non raggiungibile (min ' + formatBytes(plan.minBytes) + ')'}</span></div>
          ) : (
            <div><strong>{formatBytes(plan.minBytes)}</strong><span>minimo teorico</span></div>
          )}
        </div>
        <div className="library-table">
          <div className="library-row library-head"><span>Tensor</span><span>Ruolo</span><span>Attuale</span><span>Piano</span><span>Risparmio</span></div>
          {plan.tensors.map((tensor) => (
            <div className="library-row" key={tensor.name}>
              <span><strong>{tensor.name}</strong><small>{tensor.valueId > 0 ? `value #${tensor.valueId} · ${tensor.numel.toLocaleString('it-IT')} elementi` : `${tensor.numel.toLocaleString('it-IT')} elementi`}</small></span>
              <span className="library-kind">{tensor.role}</span>
              <span>{formatBytes(tensor.currentBytes)} <small>{tensor.currentScheme}</small></span>
              <span>{formatBytes(tensor.planBytes)} <small>{tensor.planScheme}</small></span>
              <span>{formatBytes(Math.max(0, tensor.currentBytes - tensor.planBytes))}</span>
            </div>
          ))}
        </div>
        <p className="optimize-note">Il piano è un report (§15): per applicarlo, riquantizza i tensor scelti con `pai convert --quant` e ricarica il modello nel gateway.</p>
      </section>
    )}

    {expertMode && (
      <div className="pipeline-callout">
        <div className="pipeline-icon"><Icon name="spark" size={20} /></div>
        <div><strong>Pipeline di conversione nativa</strong>
          <p>GGUF importati vengono convertiti in `.pai` con `pai convert` e validati con `pai validate` quando il toolchain è raggiungibile (variabile PAI_BIN o PATH). I safetensors richiedono il descrittore modello del toolchain Rust.</p>
        </div>
        <button className="ghost-button">Read pipeline <Icon name="arrow" size={14} /></button>
      </div>
    )}
  </>;
}
