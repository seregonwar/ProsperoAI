import { useState } from 'react';
import type { DeviceEntry } from '../../../shared/types';
import { EmptyState, Icon, SectionHeader, StatusDot, formatMs } from '../components';

interface DevicesViewProps {
  devices: DeviceEntry[];
  probing: boolean;
  onProbe: () => Promise<void>;
  onAdd: (name: string, host: string, port: number) => Promise<void>;
  onRemove: (id: string) => Promise<void>;
}

export function DevicesView({ devices, probing, onProbe, onAdd, onRemove }: DevicesViewProps) {
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [port, setPort] = useState('9021');
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState('');

  const online = devices.filter((device) => device.probe?.ok).length;
  const rtts = devices
    .map((device) => device.probe?.latencyMs)
    .filter((value): value is number => value !== undefined);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setError('');
    setAdding(true);
    try {
      await onAdd(name, host, Number(port));
      setName('');
      setHost('');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Endpoint non valido');
    } finally {
      setAdding(false);
    }
  }

  return <>
    <SectionHeader
      eyebrow="Prospero Protocol / TCP (§24/§25)"
      title="PS5 devices"
      description="Endpoint configurati e sondati con il protocollo binario reale: HELLO, negoziazione capability e PING/PONG."
      action={<button className="button secondary" onClick={() => void onProbe()} disabled={probing || devices.length === 0}><Icon name="refresh" size={16} className={probing ? 'spin' : ''} /> {probing ? 'Probing...' : 'Probe fleet'}</button>}
    />

    <div className="device-overview">
      <div className="device-overview-stat"><span className="icon-box cyan"><Icon name="console" size={18} /></span><div><strong>{online} / {devices.length}</strong><span>payload raggiunti</span></div></div>
      <div className="device-overview-stat"><span className="icon-box violet"><Icon name="wifi" size={18} /></span><div><strong>{rtts.length > 0 ? `${Math.min(...rtts).toFixed(1)} ms` : '—'}</strong><span>RTT migliore misurato</span></div></div>
      <div className="device-overview-stat"><span className="icon-box orange"><Icon name="pulse" size={18} /></span><div><strong>{devices.reduce((sum, device) => sum + (device.probe?.lost ?? 0), 0)}</strong><span>ping persi (ultimo probe)</span></div></div>
      <div className="device-overview-stat"><span className="icon-box green"><Icon name="check" size={18} /></span><div><strong>{new Set(devices.flatMap((device) => device.probe?.capsNames ?? [])).size}</strong><span>capability distinte</span></div></div>
    </div>

    <form className="device-add-form" onSubmit={(event) => void submit(event)}>
      <span className="eyebrow">Configura endpoint</span>
      <div className="device-add-fields">
        <label>Nome<input value={name} onChange={(event) => setName(event.target.value)} placeholder="es. PS5 · living room" /></label>
        <label>Host<input value={host} onChange={(event) => setHost(event.target.value)} placeholder="192.168.1.42" required /></label>
        <label>Porta<input value={port} onChange={(event) => setPort(event.target.value)} placeholder="9021" inputMode="numeric" required /></label>
        <button className="button primary" type="submit" disabled={adding || !host.trim()}><Icon name="plus" size={15} /> {adding ? 'Aggiungo...' : 'Aggiungi endpoint'}</button>
      </div>
      {error && <span className="device-add-error">{error}</span>}
    </form>

    {devices.length === 0 ? (
      <EmptyState
        icon="console"
        title="Nessun endpoint configurato"
        description="Configura host e porta del payload ProsperoAI (prosperoai.elf). Il probe esegue una connessione TCP reale, la negoziazione HELLO/HELLO_ACK e 5 PING."
      />
    ) : (
      <div className="device-cards">
        {devices.map((device) => <DeviceDetailCard device={device} key={device.id} onRemove={() => void onRemove(device.id)} onProbe={() => void onProbe()} probing={probing} />)}
      </div>
    )}

    <section className="panel protocol-panel">
      <div className="panel-heading"><div><span className="eyebrow">Connection security</span><h3>Il pairing è locale e verificabile</h3></div>
        <span className="secure-pill"><Icon name="shield" size={14} /> handshake verificato via CRC-32</span></div>
      <p>Ogni probe valida magic, versione, capability negoziate e integrità CRC-32 dei frame. Le console non configurate non vengono inventate: lo stato di ogni endpoint è il risultato dell'ultimo probe (§24).</p>
    </section>
  </>;
}

function DeviceDetailCard({ device, onRemove, onProbe, probing }: { device: DeviceEntry; onRemove: () => void; onProbe: () => void; probing: boolean }) {
  const probe = device.probe;
  return <article className={`device-detail-card ${probe?.ok ? '' : 'offline'}`}>
    <div className="detail-card-head">
      <div className="device-icon large"><Icon name="console" size={24} /></div>
      <div><h3>{device.name}</h3><span><StatusDot status={probe?.ok ? 'online' : 'offline'} /> {probe?.ok ? 'Online' : probe ? 'Offline' : 'Mai sondato'}</span></div>
      <button className="icon-button" onClick={onRemove} title="Rimuovi endpoint"><Icon name="trash" size={16} /></button>
    </div>
    <div className="device-address"><span>Endpoint</span><code>{device.host}:{device.port}</code>
      <button className="copy-button" title="Copia endpoint" onClick={() => void navigator.clipboard.writeText(`${device.host}:${device.port}`)}><Icon name="copy" size={14} /></button>
    </div>
    <div className="device-detail-stats">
      <div><span>Ultimo probe</span><strong>{probe ? new Date(probe.probedAt).toLocaleTimeString() : '—'}</strong></div>
      <div><span>RTT mediano</span><strong>{probe ? formatMs(probe.latencyMs) : '—'}</strong></div>
      <div><span>Ping persi</span><strong>{probe ? `${probe.lost}/5` : '—'}</strong></div>
    </div>
    <div className="capability-list">
      {probe?.ok && probe.capsNames.length > 0
        ? probe.capsNames.map((capability) => <span key={capability}><Icon name="check" size={12} /> {capability}</span>)
        : <span className="no-caps">{probe?.ok ? 'nessuna capability negoziata' : probe?.error ?? 'in attesa di probe'}</span>}
    </div>
    <button className="device-action" onClick={onProbe} disabled={probing}>{probing ? 'Probing...' : 'Probe now'} <Icon name="arrow" size={14} /></button>
  </article>;
}
