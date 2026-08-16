import { useEffect, useMemo, useState } from 'react';
import type {
  BenchmarkRun,
  BenchmarkOptions,
  ConnectionState,
  DesktopEvent,
  DesktopSettings,
  DeviceEntry,
  EventLevel,
  GatewayModel,
  LibraryEntry,
} from '../../shared/types';
import { Icon, StatusDot, type IconName } from './components';
import { HubView } from './HubView';
import { Overview, type View } from './views/Overview';
import { ModelsView } from './views/ModelsView';
import { DevicesView } from './views/DevicesView';
import { Playground as PlaygroundView } from './views/PlaygroundView';
import { BenchmarksView } from './views/BenchmarksView';
import { DiagnosticsView } from './views/DiagnosticsView';
import { SettingsView } from './views/SettingsView';
import './styles.css';

const DEFAULT_SETTINGS: DesktopSettings = {
  gatewayUrl: 'http://127.0.0.1:8080',
  launchAtStartup: false,
  expertMode: false,
  reduceMotion: false,
};

const navItems: Array<{ id: View; label: string; icon: IconName }> = [
  { id: 'overview', label: 'Overview', icon: 'grid' },
  { id: 'models', label: 'Model library', icon: 'layers' },
  { id: 'hub', label: 'Model Hub', icon: 'download' },
  { id: 'devices', label: 'PS5 devices', icon: 'console' },
  { id: 'playground', label: 'Playground', icon: 'spark' },
  { id: 'benchmarks', label: 'Benchmarks', icon: 'chart' },
  { id: 'diagnostics', label: 'Diagnostics', icon: 'pulse' },
];

export default function App() {
  const [activeView, setActiveView] = useState<View>('overview');
  const [settings, setSettings] = useState<DesktopSettings>(DEFAULT_SETTINGS);
  const [connection, setConnection] = useState<ConnectionState>('checking');
  const [latency, setLatency] = useState<number>();
  const [gatewayModels, setGatewayModels] = useState<GatewayModel[]>([]);
  const [devices, setDevices] = useState<DeviceEntry[]>([]);
  const [library, setLibrary] = useState<LibraryEntry[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkRun[]>([]);
  const [events, setEvents] = useState<DesktopEvent[]>([]);
  const [appVersion, setAppVersion] = useState('0.1.0');
  const [probing, setProbing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);

  /* ---- event helpers (renderer-side additions to the real log) ---- */
  function addLog(message: string, level: EventLevel = 'INFO', subsystem = 'desktop') {
    void window.prospero.logEvent(level, subsystem, message);
  }

  /* ---- gateway ---- */
  async function refreshConnection() {
    setConnection('checking');
    const result = await window.prospero.checkGateway(settings.gatewayUrl);
    setLatency(result.latencyMs);
    setConnection(result.ok ? 'online' : 'offline');
    setGatewayModels(result.models);
    return result;
  }

  /* ---- devices ---- */
  async function probeDevices() {
    setProbing(true);
    try {
      setDevices(await window.prospero.probeDevices());
    } finally {
      setProbing(false);
    }
  }

  async function addDevice(name: string, host: string, port: number) {
    setDevices(await window.prospero.addDevice({ name, host, port }));
  }

  async function removeDevice(id: string) {
    setDevices(await window.prospero.removeDevice(id));
  }

  /* ---- library ---- */
  async function importModels() {
    setImporting(true);
    try {
      const result = await window.prospero.importModels();
      if (result.queued > 0) {
        setLibrary(await window.prospero.listLibrary());
        setActiveView('models');
      }
    } finally {
      setImporting(false);
    }
  }

  /* ---- benchmarks ---- */
  async function runBenchmark(model: string, options: BenchmarkOptions) {
    setBenchmarkRunning(true);
    try {
      const run = await window.prospero.runBenchmark(model, options);
      setBenchmarks(await window.prospero.listBenchmarks());
      addLog(`benchmark completato · ${run.model} · ${run.medianTokPerSec?.toFixed(1) ?? '—'} tok/s`, 'INFO', 'benchmark');
    } finally {
      setBenchmarkRunning(false);
    }
  }

  async function deleteBenchmark(id: string) {
    setBenchmarks(await window.prospero.deleteBenchmark(id));
  }

  /* ---- settings ---- */
  async function patchSettings(patch: Partial<DesktopSettings>) {
    const next = await window.prospero.setSettings(patch);
    setSettings(next);
  }

  /* ---- boot ---- */
  useEffect(() => {
    const unsubscribeEvents = window.prospero.onEvent((entry) => {
      setEvents((current) => [entry, ...current].slice(0, 500));
    });
    void window.prospero.getAppInfo().then((info) => setAppVersion(info.version));
    void window.prospero.getSettings().then(setSettings);
    void window.prospero.listDevices().then(setDevices);
    void window.prospero.listLibrary().then(setLibrary);
    void window.prospero.listBenchmarks().then(setBenchmarks);
    void window.prospero.listEvents().then(setEvents);
    void refreshConnection();
    void probeDevices();
    return unsubscribeEvents;
    // The boot sequence intentionally runs once against the persisted endpoint.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const modelCount = gatewayModels.length + library.length;
  const viewTitle = useMemo(() => ({
    overview: 'Overview',
    models: 'Model library',
    hub: 'Model Hub',
    devices: 'PS5 devices',
    playground: 'Playground',
    benchmarks: 'Benchmarks',
    diagnostics: 'Diagnostics',
    settings: 'Settings',
  }[activeView]), [activeView]);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-symbol"><Icon name="spark" size={19} /></div>
          <div><strong>Prospero<span>AI</span></strong><small>DESKTOP</small></div>
        </div>
        <div className="workspace-switcher">
          <span className="workspace-avatar">P</span>
          <div><strong>Local workspace</strong><span>{settings.gatewayUrl}</span></div>
          <Icon name="chevron" size={14} />
        </div>
        <nav className="main-nav">
          <span className="nav-label">Workspace</span>
          {navItems.map((item) => (
            <button className={`nav-item ${activeView === item.id ? 'active' : ''}`} key={item.id} onClick={() => setActiveView(item.id)}>
              <Icon name={item.icon} size={17} />
              <span>{item.label}</span>
              {item.id === 'models' && modelCount > 0 && <b>{modelCount}</b>}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <button className={`nav-item ${activeView === 'settings' ? 'active' : ''}`} onClick={() => setActiveView('settings')}>
            <Icon name="settings" size={17} /><span>Settings</span>
          </button>
          <div className="sidebar-status">
            <span className="status-ring"><StatusDot status={connection} /></span>
            <div>
              <strong>{connection === 'online' ? 'Gateway online' : connection === 'checking' ? 'Checking gateway' : 'Gateway offline'}</strong>
              <span>{connection === 'online' ? `${latency ?? '—'} ms · ${settings.gatewayUrl}` : 'Start pai serve to connect'}</span>
            </div>
            <button className="icon-button" onClick={() => void refreshConnection()}><Icon name="refresh" size={15} /></button>
          </div>
          <div className="user-row">
            <span className="user-avatar">O</span>
            <div><strong>operator</strong><span>Local environment</span></div>
            <button className="icon-button"><Icon name="more" size={16} /></button>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>ProsperoAI</span><Icon name="chevron" size={13} /><strong>{viewTitle}</strong></div>
          <div className="topbar-actions">
            <div className="protocol-status"><StatusDot status={connection} /><span>{connection === 'online' ? 'Protocol ready' : 'Protocol disconnected'}</span></div>
            <button className="icon-button top-icon"><Icon name="search" size={18} /></button>
            <button className="avatar-button">O</button>
          </div>
        </header>

        <div className="page-content">
          {activeView === 'overview' && (
            <Overview
              connection={connection}
              latencyMs={latency}
              gatewayModels={gatewayModels}
              library={library}
              devices={devices}
              benchmarks={benchmarks}
              events={events}
              onNavigate={setActiveView}
            />
          )}
          {activeView === 'models' && (
            <ModelsView
              gatewayModels={gatewayModels}
              library={library}
              importing={importing}
              onImport={() => void importModels()}
              onNavigate={setActiveView}
              expertMode={settings.expertMode}
            />
          )}
          {activeView === 'hub' && (
            <HubView
              gatewayUrl={settings.gatewayUrl}
              onImport={() => void importModels()}
              onNavigate={setActiveView}
              onLog={addLog}
            />
          )}
          {activeView === 'devices' && (
            <DevicesView
              devices={devices}
              probing={probing}
              onProbe={probeDevices}
              onAdd={addDevice}
              onRemove={removeDevice}
            />
          )}
          {activeView === 'playground' && (
            <PlaygroundView
              models={gatewayModels}
              gatewayOnline={connection === 'online'}
              gatewayUrl={settings.gatewayUrl}
              onRefresh={() => void refreshConnection()}
              onLog={addLog}
            />
          )}
          {activeView === 'benchmarks' && (
            <BenchmarksView
              models={gatewayModels}
              runs={benchmarks}
              running={benchmarkRunning}
              onRun={runBenchmark}
              onDelete={deleteBenchmark}
            />
          )}
          {activeView === 'diagnostics' && (
            <DiagnosticsView
              events={events}
              onExport={() => window.prospero.exportDiagnostics()}
            />
          )}
          {activeView === 'settings' && (
            <SettingsView
              settings={settings}
              appVersion={appVersion}
              connection={connection}
              latencyMs={latency}
              onPatch={patchSettings}
              onTest={async () => { await refreshConnection(); }}
            />
          )}
        </div>

        <footer className="status-footer">
          <span><i className="live-pulse" /> Desktop shell v{appVersion}</span>
          <span>Prospero Protocol v0.1 <b>·</b> {modelCount} model entries</span>
        </footer>
      </main>
    </div>
  );
}
