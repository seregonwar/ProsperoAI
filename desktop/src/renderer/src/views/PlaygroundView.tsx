import { useEffect, useRef, useState } from 'react';
import type { ChatMessage, ConnectionState, GatewayModel } from '../../../shared/types';
import { EmptyState, Icon, SectionHeader, StatusDot } from '../components';

interface PlaygroundProps {
  models: GatewayModel[];
  gatewayOnline: boolean;
  gatewayUrl: string;
  onRefresh: () => void;
  onLog: (message: string, level?: 'TRACE' | 'INFO' | 'WARN' | 'ERROR') => void;
}

const WELCOME: ChatMessage = { role: 'assistant', content: 'Playground collegato al gateway. Seleziona un modello e invia un prompt: la risposta arriva in streaming, token per token.' };

export function Playground({ models, gatewayOnline, gatewayUrl, onRefresh, onLog }: PlaygroundProps) {
  const usable = models.filter((model) => !model.broken);
  const [model, setModel] = useState('');
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME]);
  const [busy, setBusy] = useState(false);
  const [streamingText, setStreamingText] = useState<string | null>(null);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);

  const streamingRef = useRef('');
  const requestIdRef = useRef<string | null>(null);
  const cancelledRef = useRef(false);

  useEffect(() => {
    if (!model && usable.length > 0) setModel(usable[0].id);
  }, [usable, model]);

  async function sendPrompt() {
    const trimmed = prompt.trim();
    if (!trimmed || busy || !model) return;
    const nextMessages: ChatMessage[] = [...messages, { role: 'user', content: trimmed }];
    setMessages(nextMessages);
    setPrompt('');
    setBusy(true);
    streamingRef.current = '';
    setStreamingText('');
    cancelledRef.current = false;
    const requestId = crypto.randomUUID();
    requestIdRef.current = requestId;
    onLog(`stream avviato · ${model} · max_tokens=${maxTokens}`, 'TRACE');
    try {
      const result = await window.prospero.streamChat(
        requestId,
        gatewayUrl,
        model,
        nextMessages,
        { temperature, maxTokens },
        (delta) => {
          if (delta.error) {
            streamingRef.current += `\n[${cancelledRef.current ? 'stream interrotto' : `errore: ${delta.error}`}]`;
          } else if (!delta.done) {
            streamingRef.current += delta.token;
          }
          setStreamingText(streamingRef.current);
        },
      );
      if (!result.ok && !cancelledRef.current && !result.error) {
        streamingRef.current += '\n[stream fallito]';
      }
      setMessages((current) => [...current, { role: 'assistant', content: streamingRef.current }]);
      onLog(result.ok ? `stream completato · ${result.tokens} token` : `stream interrotto · ${result.error ?? 'errore'}`, result.ok ? 'INFO' : 'WARN');
    } catch (error) {
      streamingRef.current += `\n[errore: ${error instanceof Error ? error.message : 'stream fallito'}]`;
      setMessages((current) => [...current, { role: 'assistant', content: streamingRef.current }]);
      onLog('stream fallito', 'ERROR');
    } finally {
      setBusy(false);
      setStreamingText(null);
      requestIdRef.current = null;
    }
  }

  function cancel() {
    if (!requestIdRef.current) return;
    cancelledRef.current = true;
    window.prospero.cancelChatStream(requestIdRef.current);
  }

  return <>
    <SectionHeader
      eyebrow="Inference lab / OpenAI-compatible (§26)"
      title="Playground"
      description="Completamenti in streaming dal gateway reale; i parametri di sampling vengono trasmessi nella richiesta."
      action={<div className={`connection-chip ${gatewayOnline ? 'online' : 'offline'}`}><StatusDot status={gatewayOnline ? 'online' : 'offline'} /> {gatewayOnline ? 'Gateway ready' : 'Gateway offline'}</div>}
    />

    {usable.length === 0 ? (
      <EmptyState
        icon="spark"
        title="Nessun modello servito"
        description="Il gateway non espone modelli. Avvia `pai serve <model.pai>` e ricontrolla la connessione dalla sidebar."
        action={<button className="button secondary" onClick={onRefresh}><Icon name="refresh" size={15} /> Ricontrolla</button>}
      />
    ) : (
      <div className="playground-layout">
        <section className="panel chat-panel">
          <div className="chat-head">
            <div><span className="eyebrow">Live session · streaming SSE</span><h3>{model || 'Nessun modello selezionato'}</h3></div>
            {busy && <button className="small-button muted" onClick={cancel}><Icon name="stop" size={13} /> Stop</button>}
          </div>
          <div className="chat-messages">
            {messages.map((message, index) => (
              <div className={`chat-message ${message.role}`} key={`${message.role}-${index}`}>
                <span className="message-avatar">{message.role === 'assistant' ? <Icon name="spark" size={14} /> : 'Y'}</span>
                <div><span className="message-role">{message.role === 'assistant' ? 'ProsperoAI' : 'You'}</span><p>{message.content}</p></div>
              </div>
            ))}
            {busy && streamingText !== null && (
              <div className="chat-message assistant">
                <span className="message-avatar"><Icon name="spark" size={14} /></span>
                <div><span className="message-role">ProsperoAI</span><p>{streamingText}<span className="stream-caret" /></p></div>
              </div>
            )}
          </div>
          <div className="prompt-box">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void sendPrompt(); } }}
              placeholder={model ? 'Chiedi qualcosa al modello locale...' : 'Seleziona prima un modello'}
              rows={2}
            />
            <div className="prompt-footer">
              <span>Enter per inviare · Shift+Enter per nuova riga</span>
              <button className="send-button" onClick={() => void sendPrompt()} disabled={!prompt.trim() || busy || !model}><Icon name="send" size={16} /></button>
            </div>
          </div>
        </section>

        <aside className="playground-sidebar">
          <section className="panel config-panel">
            <div className="panel-heading"><div><span className="eyebrow">Session config</span><h3>Generation</h3></div><Icon name="settings" size={17} /></div>
            <label>Model
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                <option value="">Seleziona un modello</option>
                {usable.map((item) => <option value={item.id} key={item.id}>{item.id}{item.kind === 'remote' ? ' · remote' : ''}</option>)}
              </select>
            </label>
            <div className="range-row"><label>Temperature <b>{temperature.toFixed(2)}</b></label>
              <input type="range" min="0" max="1" step=".05" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></div>
            <div className="range-row"><label>Max tokens <b>{maxTokens}</b></label>
              <input type="range" min="16" max="2048" step="16" value={maxTokens} onChange={(event) => setMaxTokens(Number(event.target.value))} /></div>
            <div className="session-tags"><span><Icon name="bolt" size={12} /> streaming</span><span><Icon name="shield" size={12} /> locale</span></div>
          </section>
          <section className="panel route-panel">
            <span className="eyebrow">Request route</span>
            <code>POST {gatewayUrl}/v1/chat/completions</code>
            <div><StatusDot status={gatewayOnline ? 'online' : 'offline'} /> {gatewayOnline ? 'Gateway connesso · stream:true' : 'Avvia pai serve per l’inference'}</div>
          </section>
        </aside>
      </div>
    )}
  </>;
}
