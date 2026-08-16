import { useState } from 'react';
import type { DesktopEvent } from '../../../shared/types';
import { Icon, SectionHeader } from '../components';

interface DiagnosticsViewProps {
  events: DesktopEvent[];
  onExport: () => Promise<string | null>;
}

export function DiagnosticsView({ events, onExport }: DiagnosticsViewProps) {
  const [exporting, setExporting] = useState(false);
  const [exportedPath, setExportedPath] = useState<string | null>(null);

  const warns = events.filter((entry) => entry.level === 'WARN').length;
  const errors = events.filter((entry) => entry.level === 'ERROR').length;
  const traces = events.filter((entry) => entry.level === 'TRACE').length;

  async function exportBundle() {
    setExporting(true);
    try {
      setExportedPath(await onExport());
    } finally {
      setExporting(false);
    }
  }

  return <>
    <SectionHeader
      eyebrow="Observability / ring buffer (§33)"
      title="Diagnostics"
      description="Eventi strutturati dal main process: gateway, protocollo, libreria, benchmark e hub. Nessun log di esempio."
      action={<button className="button secondary" onClick={() => void exportBundle()} disabled={exporting}><Icon name="download" size={16} /> {exporting ? 'Scrivo...' : 'Diagnostic bundle'}</button>}
    />
    {exportedPath && <div className="hub-error" style={{ borderColor: 'rgba(102,211,142,.25)', color: '#9ee8b7', background: 'rgba(102,211,142,.06)' }}>
      <Icon name="check" size={16} /><div><strong>Bundle scritto</strong><span>{exportedPath}</span></div>
      <button className="icon-button" onClick={() => setExportedPath(null)}>×</button>
    </div>}

    <div className="diagnostic-stats">
      <div><span className="severity info" /><strong>{events.length}</strong><small>eventi in sessione</small></div>
      <div><span className="severity trace" /><strong>{traces}</strong><small>trace</small></div>
      <div><span className="severity warn" /><strong>{warns}</strong><small>warning</small></div>
      <div><span className="severity error" /><strong>{errors}</strong><small>errori</small></div>
    </div>

    <section className="panel log-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Live stream</span><h3>Runtime events</h3></div>
        <div className="log-actions"><span className="live-label"><i className="live-pulse" /> live</span></div>
      </div>
      <div className="log-table">
        {events.length === 0 && <div className="log-empty">In attesa del primo evento...</div>}
        {events.map((entry, index) => (
          <div className="log-row" key={`${entry.time}-${index}`}>
            <time>{entry.time}</time>
            <span className={`log-level ${entry.level.toLowerCase()}`}>{entry.level}</span>
            <span className="log-subsystem">{entry.subsystem}</span>
            <span>{entry.message}</span>
          </div>
        ))}
      </div>
      <div className="log-footer">
        <span><i className="live-pulse" /> {events.length} eventi · ring buffer 500</span>
        <span className="log-ring-note">persistiti in logs/events.jsonl</span>
      </div>
    </section>

    <div className="diagnostic-note"><Icon name="shield" size={18} />
      <div><strong>I bundle diagnostici sono sicuri da condividere</strong>
        <p>Il bundle JSON include eventi, configurazione, dispositivi, libreria e benchmark. Credenziali e token (es. HUGGINGFACE_TOKEN) non vengono mai inclusi.</p>
      </div>
    </div>
  </>;
}
