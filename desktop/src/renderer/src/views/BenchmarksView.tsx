import { useState } from 'react';
import type { BenchmarkOptions, BenchmarkRun, GatewayModel } from '../../../shared/types';
import { EmptyState, Icon, SectionHeader, Sparkline } from '../components';

interface BenchmarksViewProps {
  models: GatewayModel[];
  runs: BenchmarkRun[];
  running: boolean;
  onRun: (model: string, options: BenchmarkOptions) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

function winner(runs: BenchmarkRun[]): BenchmarkRun | undefined {
  return [...runs].sort((a, b) => (b.medianTokPerSec ?? 0) - (a.medianTokPerSec ?? 0))[0];
}

export function BenchmarksView({ models, runs, running, onRun, onDelete }: BenchmarksViewProps) {
  const usable = models.filter((model) => !model.broken);
  const [model, setModel] = useState('');
  const [iterations, setIterations] = useState(3);
  const [maxTokens, setMaxTokens] = useState(128);
  const [prompt, setPrompt] = useState('');
  const best = winner(runs);

  async function startRun() {
    if (!model || running) return;
    await onRun(model, { iterations, maxTokens, prompt: prompt.trim() || undefined });
  }

  return <>
    <SectionHeader
      eyebrow="Performance lab / reproducible runs (§31)"
      title="Benchmarks"
      description="Ogni run misura tempi reali contro il gateway: TTFT, token totali e tokens/sec per iterazione."
      action={usable.length > 0
        ? <button className="button primary" onClick={() => void startRun()} disabled={running || !model}><Icon name="plus" size={16} /> {running ? 'Running...' : 'New benchmark'}</button>
        : undefined}
    />

    <form className="bench-form panel" onSubmit={(event) => { event.preventDefault(); void startRun(); }}>
      <span className="eyebrow">Configura il run</span>
      <div className="bench-fields">
        <label>Modello
          <select value={model} onChange={(event) => setModel(event.target.value)}>
            <option value="">Seleziona un modello servito</option>
            {usable.map((item) => <option value={item.id} key={item.id}>{item.id}{item.kind === 'remote' ? ' · remote' : ''}</option>)}
          </select>
        </label>
        <label>Iterazioni
          <input type="number" min="1" max="10" value={iterations} onChange={(event) => setIterations(Math.max(1, Math.min(10, Number(event.target.value) || 1)))} />
        </label>
        <label>Max tokens
          <input type="number" min="16" max="2048" step="16" value={maxTokens} onChange={(event) => setMaxTokens(Math.max(16, Math.min(2048, Number(event.target.value) || 16)))} />
        </label>
        <label className="bench-prompt">Prompt (opzionale)
          <input value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="default: storia del computing" />
        </label>
        <button className="button primary bench-submit" type="submit" disabled={running || !model}><Icon name="bolt" size={15} /> {running ? 'Misurando...' : 'Avvia run'}</button>
      </div>
    </form>

    {runs.length === 0 ? (
      <EmptyState
        icon="chart"
        title="Nessun run registrato"
        description="I benchmark vengono misurati contro il gateway reale e salvati in userData/benchmarks.json: nessun numero viene inventato."
      />
    ) : (
      <>
        <div className="benchmark-hero">
          <div>
            <span className="eyebrow">{best ? `Miglior run · ${best.model}` : 'Nessun run'}</span>
            <strong>{best?.medianTokPerSec !== undefined ? best.medianTokPerSec.toFixed(1) : '—'} <small>tok / sec</small></strong>
            <span className="positive"><Icon name="check" size={13} /> {best ? `${best.samples.length} iterazioni · mediana` : '—'}</span>
          </div>
          {best && <div className="benchmark-ring"><span>{best.medianTtftMs !== undefined ? Math.round(best.medianTtftMs) : '—'}<small>ms</small></span><label>TTFT mediano</label></div>}
          {best && best.samples.length > 1 && <Sparkline values={best.samples.map((sample) => sample.tokPerSec)} color="#43d6c7" />}
        </div>

        <section className="panel">
          <div className="panel-heading">
            <div><span className="eyebrow">Run history</span><h3>Cronologia misurazioni</h3></div>
            <span className="secure-pill"><Icon name="info" size={14} /> {runs.length} run persistenti</span>
          </div>
          <div className="benchmark-table">
            <div className="table-row table-head"><span>Modello</span><span>Tok / sec</span><span>TTFT</span><span>Token</span><span>Iterazioni</span><span>Data</span></div>
            {runs.map((run) => (
              <div className="table-row" key={run.id}>
                <span><strong>{run.model}</strong><small>max_tokens={run.maxTokens} · {run.gatewayUrl}</small></span>
                <b>{run.medianTokPerSec !== undefined ? run.medianTokPerSec.toFixed(1) : '—'}</b>
                <span>{run.medianTtftMs !== undefined ? `${Math.round(run.medianTtftMs)} ms` : '—'}</span>
                <span>{run.medianTokens ?? '—'}</span>
                <span>{run.samples.length}</span>
                <span>{new Date(run.startedAt).toLocaleString()}</span>
                <button className="icon-button" title="Elimina run" onClick={() => void onDelete(run.id)}><Icon name="trash" size={14} /></button>
              </div>
            ))}
          </div>
        </section>
      </>
    )}
  </>;
}
