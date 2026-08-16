/*
 * ProsperoAI Desktop — desktop event log (whitepaper §33).
 *
 * Ring buffer in the main process, persisted as JSONL under
 * app.getPath('userData')/logs/events.jsonl and streamed live to the
 * renderer. Every backend action (gateway checks, protocol probes,
 * imports, benchmark runs, hub pulls) writes here — the Diagnostics
 * view is a window on real activity, never seeded data.
 */

import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import type { DesktopEvent, EventLevel } from '../../shared/types';

const MAX_RING = 500;
const MAX_PERSISTED = 2000;

function nowStamp(): string {
  const now = new Date();
  return `${now.toTimeString().slice(0, 8)}.${String(now.getMilliseconds()).padStart(3, '0')}`;
}

export class EventLog {
  private ring: DesktopEvent[] = [];
  private readonly emitter = new EventEmitter();
  private filePath = '';

  constructor(baseDir: string) {
    this.filePath = path.join(baseDir, 'logs', 'events.jsonl');
  }

  /* Load persisted history (best effort). */
  init(): void {
    try {
      const text = fs.readFileSync(this.filePath, 'utf8');
      const lines = text.split('\n').filter((line) => line.length > 0);
      const tail = lines.slice(-MAX_PERSISTED);
      for (const line of tail) {
        try {
          const entry = JSON.parse(line) as DesktopEvent;
          if (entry && typeof entry.time === 'string' && typeof entry.message === 'string') {
            this.ring.push(entry);
          }
        } catch {
          /* skip malformed lines */
        }
      }
      this.ring = this.ring.slice(-MAX_RING);
    } catch {
      /* no history yet */
    }
  }

  push(level: EventLevel, subsystem: string, message: string): DesktopEvent {
    const entry: DesktopEvent = { time: nowStamp(), level, subsystem, message };
    this.ring.push(entry);
    if (this.ring.length > MAX_RING) this.ring.shift();
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
      fs.appendFileSync(this.filePath, `${JSON.stringify(entry)}\n`);
    } catch {
      /* persistence is best effort */
    }
    this.emitter.emit('event', entry);
    return entry;
  }

  list(): DesktopEvent[] {
    return [...this.ring];
  }

  subscribe(listener: (event: DesktopEvent) => void): () => void {
    this.emitter.on('event', listener);
    return () => this.emitter.off('event', listener);
  }
}
