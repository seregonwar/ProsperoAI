import { useState } from 'react';
import type { ConnectionState, DesktopSettings } from '../../../shared/types';
import { Icon, SectionHeader, StatusDot, Toggle } from '../components';

interface SettingsViewProps {
  settings: DesktopSettings;
  appVersion: string;
  connection: ConnectionState;
  latencyMs?: number;
  onPatch: (patch: Partial<DesktopSettings>) => Promise<void>;
  onTest: () => Promise<void>;
}

export function SettingsView({ settings, appVersion, connection, latencyMs, onPatch, onTest }: SettingsViewProps) {
  const [gatewayDraft, setGatewayDraft] = useState(settings.gatewayUrl);
  const [saved, setSaved] = useState(false);

  async function saveGateway() {
    await onPatch({ gatewayUrl: gatewayDraft.trim() || settings.gatewayUrl });
    setSaved(true);
    setTimeout(() => setSaved(false), 1800);
  }

  return <>
    <SectionHeader eyebrow="Desktop preferences" title="Settings" description="Configurazione persistente nel main process (userData/config.json)." />

    <div className="settings-layout">
      <section className="panel settings-panel">
        <div className="settings-section">
          <span className="eyebrow">Gateway connection</span>
          <h3>OpenAI-compatible endpoint</h3>
          <p>Prospero Desktop parla con il gateway locale (`pai serve`); il Prospero Protocol binario resta il percorso di controllo verso i payload PS5.</p>
          <label className="wide-label">Base URL
            <div className="input-with-icon"><Icon name="terminal" size={16} />
              <input value={gatewayDraft} onChange={(event) => setGatewayDraft(event.target.value)} placeholder="http://127.0.0.1:8080" />
            </div>
          </label>
          <div className="settings-actions">
            <button className="button primary" onClick={() => void saveGateway()} disabled={gatewayDraft.trim() === settings.gatewayUrl}><Icon name="check" size={15} /> {saved ? 'Salvato' : 'Salva endpoint'}</button>
            <button className="button secondary" onClick={() => void onTest()} disabled={connection === 'checking'}><Icon name="refresh" size={15} /> Test connection</button>
            <span className={`connection-chip ${connection === 'online' ? 'online' : 'offline'}`}><StatusDot status={connection} /> {connection === 'online' ? `${latencyMs ?? '—'} ms` : connection === 'checking' ? 'checking' : 'offline'}</span>
          </div>
        </div>

        <div className="settings-divider" />

        <div className="settings-section">
          <span className="eyebrow">Experience</span>
          <h3>Preferenze desktop</h3>
          <Toggle label="Launch on system startup" description="Registra l'avvio automatico nel sistema operativo (setLoginItemSettings)." enabled={settings.launchAtStartup} onChange={(next) => void onPatch({ launchAtStartup: next })} />
          <Toggle label="Expert mode" description="Espone dettagli della pipeline di conversione e dei controlli avanzati." enabled={settings.expertMode} onChange={(next) => void onPatch({ expertMode: next })} />
          <Toggle label="Reduce motion" description="Preferisce transizioni statiche nell'interfaccia." enabled={settings.reduceMotion} onChange={(next) => void onPatch({ reduceMotion: next })} />
        </div>
      </section>

      <aside className="settings-aside">
        <div className="brand-card">
          <div className="brand-mark"><Icon name="spark" size={22} /></div>
          <strong>ProsperoAI Desktop</strong>
          <span>v{appVersion}</span>
          <div className="brand-divider" />
          <span className="brand-status"><StatusDot status={connection} /> {connection === 'online' ? 'Gateway connesso' : 'Gateway offline'}</span>
        </div>
        <div className="settings-help">
          <Icon name="terminal" size={17} />
          <div><strong>Quick start</strong>
            <code>pai serve model.pai</code>
            <p>Avvia il gateway host, poi testa l'endpoint. `pai convert model.gguf out.pai` prepara i container dalla libreria.</p>
          </div>
        </div>
      </aside>
    </div>
  </>;
}
