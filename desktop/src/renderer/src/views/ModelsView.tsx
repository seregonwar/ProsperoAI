import { useMemo, useState } from 'react';
import type { GatewayModel, LibraryEntry } from '../../../shared/types';
import { EmptyState, Icon, SectionHeader, StatusDot, formatBytes } from '../components';
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
              <span className="library-message">{entry.message ?? (entry.status === 'ready' ? 'integrità verificata' : '—')}</span>
            </div>
          ))}
        </div>
      )}
    </section>

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
