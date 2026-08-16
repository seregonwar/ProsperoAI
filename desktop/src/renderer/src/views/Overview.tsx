import type { BenchmarkRun, ConnectionState, DesktopEvent, DeviceEntry, GatewayModel, LibraryEntry } from '../../../shared/types';
import { Icon, MetricCard, SectionHeader, Sparkline, StatusDot } from '../components';

export type View = 'overview' | 'models' | 'hub' | 'devices' | 'playground' | 'benchmarks' | 'diagnostics' | 'settings';

interface OverviewProps {
  connection: ConnectionState;
  latencyMs?: number;
  gatewayModels: GatewayModel[];
  library: LibraryEntry[];
  devices: DeviceEntry[];
  benchmarks: BenchmarkRun[];
  events: DesktopEvent[];
  onNavigate: (view: View) => void;
}


export function Overview({ connection, latencyMs, gatewayModels, library, devices, benchmarks, events, onNavigate }: OverviewProps) {
  const online = connection === 'online';
  const onlineDevices = devices.filter((device) => device.probe?.ok).length;
  const readyLibrary = library.filter((entry) => entry.status === 'ready').length;
  const served = gatewayModels.length;
  const lastBenchmark = benchmarks[0];
  const recentEvents = events.slice(0, 5);

  return <>
    <SectionHeader
      eyebrow="Control center / real-time state"
      title={online ? 'Good morning, operator.' : 'Gateway offline.'}
      description={online
        ? 'Your local inference stack is connected and reporting real state.'
        : 'Start the gateway (`pai serve`) to bring the workspace online.'}
      action={<button className="button primary" onClick={() => onNavigate('playground')} disabled={!online}><Icon name="spark" size={16} /> Open playground <Icon name="arrow" size={15} /></button>}
    />

    <div className="hero-strip">
      <div className="hero-glow" />
      <div className="hero-copy">
        <span className="live-label" style={online ? undefined : { color: '#e8a76f' }}>
          <span className="live-pulse" style={online ? undefined : { background: '#e06f7c', boxShadow: '0 0 0 4px rgba(224,111,124,.12)' }} />
          {online ? 'GATEWAY ONLINE' : 'GATEWAY OFFLINE'}
        </span>
        <h2>PS5 inference, <em>measured.</em></h2>
        <p>{online
          ? `${served} model${served === 1 ? '' : 'i'} serviti dal gateway · ${latencyMs ?? '—'} ms RTT misurati`
          : 'Nessun dato fabbricato: tutto ciò che vedi viene dal gateway o dal protocollo.'}</p>
      </div>
      <div className="hero-stats">
        <div><strong>{online ? String(latencyMs ?? '—') : '—'}</strong><span>gateway RTT (ms)</span></div>
        <div><strong>{String(served)}</strong><span>modelli serviti</span></div>
        <div><strong>{onlineDevices > 0 ? String(onlineDevices) : '0'}</strong><span>payload online</span></div>
      </div>
    </div>

    <div className="metric-grid">
      <MetricCard label="Modelli disponibili" value={String(served + readyLibrary)} detail={`${served} gateway · ${readyLibrary} libreria`} color="#9a7cff" icon="layers" />
      <MetricCard label="Dispositivi online" value={`${onlineDevices} / ${devices.length}`} detail={devices.length === 0 ? 'nessun endpoint configurato' : 'probe protocollo reale'} color="#43d6c7" icon="console" />
      <MetricCard label="Ultimo benchmark" value={lastBenchmark?.medianTokPerSec !== undefined ? lastBenchmark.medianTokPerSec.toFixed(1) : '—'} detail={lastBenchmark ? `${lastBenchmark.model} · ${lastBenchmark.samples.length} run` : 'nessun run registrato'} color="#f5a45b" icon="bolt" />
      <MetricCard label="Eventi in sessione" value={String(events.length)} detail={`${events.filter((e) => e.level === 'WARN').length} warning`} color="#6e9dff" icon="pulse" />
    </div>

    <div className="content-grid overview-grid">
      <section className="panel chart-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Benchmark / ultimo run</span><h3>Throughput reale</h3></div>
          <button className="ghost-button" onClick={() => onNavigate('benchmarks')}>Vedi benchmark <Icon name="arrow" size={14} /></button>
        </div>
        {lastBenchmark && lastBenchmark.samples.length > 1 ? (
          <div className="big-chart">
            <div className="chart-axis"><span>max</span><span>med</span><span>min</span></div>
            <div className="chart-body">
              <div className="chart-lines"><i /><i /><i /></div>
              <Sparkline values={lastBenchmark.samples.map((sample) => sample.tokPerSec)} color="#9a7cff" />
              <div className="chart-labels">
                {lastBenchmark.samples.map((sample) => <span key={sample.iteration}>run {sample.iteration}</span>)}
              </div>
            </div>
          </div>
        ) : (
          <div className="panel-placeholder">
            <Icon name="chart" size={20} />
            <p>Nessun benchmark eseguito. Avvia un run nella vista Benchmark per avere dati reali.</p>
            <button className="small-button" onClick={() => onNavigate('benchmarks')}>Nuovo benchmark <Icon name="arrow" size={13} /></button>
          </div>
        )}
        <div className="chart-legend"><span><i className="legend-dot purple" /> tokens / sec misurati</span></div>
      </section>

      <section className="panel activity-panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Live feed</span><h3>Attività recente</h3></div>
          <button className="icon-button"><Icon name="more" size={18} /></button>
        </div>
        <div className="activity-list">
          {recentEvents.length === 0 && <div className="activity-empty">Nessun evento registrato finora.</div>}
          {recentEvents.map((entry, index) => (
            <div className="activity-item" key={`${entry.time}-${index}`}>
              <span className="activity-icon" style={entry.level === 'ERROR' ? { color: '#ee8791', background: 'rgba(224,111,124,.1)' } : entry.level === 'WARN' ? { color: 'var(--orange)', background: 'rgba(245,164,91,.11)' } : undefined}>
                <Icon name={entry.level === 'ERROR' ? 'stop' : entry.level === 'WARN' ? 'pulse' : 'check'} size={15} />
              </span>
              <div><strong>{entry.message}</strong><span>{entry.subsystem}</span></div>
              <time>{entry.time}</time>
            </div>
          ))}
        </div>
        <button className="inline-link" onClick={() => onNavigate('diagnostics')}>Tutti gli eventi <Icon name="arrow" size={14} /></button>
      </section>
    </div>

    <section className="panel fleet-panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Runtime fleet</span><h3>Endpoint configurati</h3></div>
        <button className="ghost-button" onClick={() => onNavigate('devices')}>Gestisci fleet <Icon name="arrow" size={14} /></button>
      </div>
      <div className="device-row-list">
        {devices.length === 0 && <div className="activity-empty">Nessun endpoint configurato: aggiungi un payload PS5 nella vista Devices.</div>}
        {devices.map((device) => (
          <div className="device-row" key={device.id}>
            <div className="device-main">
              <span className={`device-icon ${device.probe?.ok ? '' : 'offline'}`}><Icon name="console" size={18} /></span>
              <div><strong>{device.name}</strong><span>{device.host}:{device.port}</span></div>
            </div>
            <div className="device-capabilities">
              {(device.probe?.capsNames ?? []).slice(0, 3).map((capability) => <span key={capability}>{capability}</span>)}
              {!device.probe && <span>mai sondato</span>}
            </div>
            <div className="device-load"><span>{device.probe?.ok ? `${device.probe.latencyMs?.toFixed(1) ?? '—'} ms RTT` : 'unavailable'}</span></div>
            <div className="device-state"><StatusDot status={device.probe?.ok ? 'online' : 'offline'} />{device.probe?.ok ? 'Online' : 'Offline'}</div>
            <Icon name="chevron" size={15} />
          </div>
        ))}
      </div>
    </section>
  </>;
}
