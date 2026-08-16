# ProsperoAI Desktop

Control center Electron prevista dal whitepaper §7, collegata ai backend reali del monorepo. **Non ci sono dati di esempio**: ogni vista riflette lo stato reale del gateway, del protocollo e dei servizi del main process; ciò che manca viene mostrato come empty state.

## Backend reale integrato

- **Gateway OpenAI-compatible (§26)**: `/healthz`, `/v1/models` (con metadati reali `kind`/`size_bytes`/`endpoint`/`broken`), `/v1/chat/completions` in **streaming SSE** con parametri sampling completi (`temperature`, `top_p`, `top_k`, `seed`, `max_tokens`, `stop` sequences) e `stream_options.include_usage` per il chunk di usage finale. Il **Playground** espone tutti i controlli e mostra token + finish_reason dell'ultima risposta. Avvio: `pai serve <model.pai> [--remote name@host:port]... --host 127.0.0.1 --port 8080`.
- **Client Prospero Protocol in TypeScript (§24/§25)**: porting fedele del codec C (`desktop/src/main/proto/`): frame 40 byte LE, CRC-32 ISO-HDLC, negoziazione HELLO/HELLO_ACK con verifica versione e capability, PING/PONG. La vista **PS5 devices** sonda endpoint configurati dall'utente: nessuna console viene inventata. Unit test portati dai vettori C (`npm run test:proto`, 24 test).
- **Pipeline import (§7/§8)**: i file importati vengono copiati in `userData/models/library`; se il toolchain `pai` è raggiungibile (`PAI_BIN` o PATH), i GGUF vengono convertiti davvero (`pai convert`) e validati (`pai validate`). Lo stato di ogni file (imported/converting/ready/failed) è persistito in `library.json`.
- **Piano mixed-precision (§15/§20)**: ogni `.pai` pronto nella libreria espone un'azione **Ottimizza** che esegue `pai optimize --json` sul container: per ogni tensor il piano mostra ruolo (embedding/attention/mlp/output/norm/other), schema attuale e schema consigliato (f32/q8/q4) con dimensioni e risparmio totali.
- **Manifest container (§20)**: l'azione **Dettagli** esegue `pai inspect --json` e mostra il report reale: sezioni del container, metadati (layer/context/vocab/KV), tensor più pesanti con dtype e shape, riepilogo IR e tokenizer.
- **Benchmark (§31)**: run riproducibili misurati contro il gateway (TTFT, token, tokens/sec, mediana su più iterazioni), persistiti in `userData/benchmarks.json`.
- **Diagnostica (§33)**: ring buffer eventi nel main process (`userData/logs/events.jsonl`) trasmesso live al renderer; export del diagnostic bundle in `userData/diagnostics/`.
- **Config persistente**: `userData/config.json` (gateway URL, device endpoint, preferenze incl. launch-at-startup reale).
- **Model Hub**: adapter reali Hugging Face (ricerca pubblica + download con progress e cancellazione), Ollama locale (`/api/pull` NDJSON) e registry OpenAI-compatible (discovery).

## Sviluppo

Da `desktop/`:

```sh
npm install
npm run dev          # vite + tsc watch + electron
npm run typecheck    # renderer + main
npm run test:proto   # unit test codec/protocollo (vettori C + client su socket)
npm run build
npm start
```

La preview browser (`vite`) usa uno stub read-only: mostra gli empty state reali, senza inventare dati. Hugging Face legge `HUGGINGFACE_TOKEN` per repository gated/private.

## Nota ambiente Windows

Se `npm install` fallisce con `EBUSY` su `node_modules/electron` (lock di sistema su un install parziale), sposta la cartella e reinstalla:

```sh
mv node_modules node_modules.old   # se rename fallisce: chiudi i processi che usano Electron e riprova
npm install
```
