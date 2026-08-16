/*
 * ProsperoAI Desktop — benchmark runner (whitepaper §31).
 *
 * Reproducible performance runs measured against the real gateway:
 * each iteration streams a fixed-prompt completion and records
 * time-to-first-token, total wall time and tokens/sec (one SSE chunk
 * per token). Results are aggregated over the median and persisted to
 * userData/benchmarks.json so the Benchmarks view compares real runs
 * — never fabricated numbers.
 */

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import type { BenchmarkOptions, BenchmarkRun, BenchmarkRunSample } from '../../shared/types';
import { normaliseBaseUrl, streamChat, validateMessages } from './gateway';

const HISTORY_LIMIT = 24;

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

export class BenchmarkStore {
  private readonly filePath: string;

  constructor(baseDir: string) {
    this.filePath = path.join(baseDir, 'benchmarks.json');
  }

  list(): BenchmarkRun[] {
    try {
      const raw = JSON.parse(fs.readFileSync(this.filePath, 'utf8')) as { runs?: BenchmarkRun[] };
      return Array.isArray(raw.runs) ? raw.runs : [];
    } catch {
      return [];
    }
  }

  private persist(runs: BenchmarkRun[]): void {
    try {
      fs.mkdirSync(path.dirname(this.filePath), { recursive: true });
      fs.writeFileSync(this.filePath, `${JSON.stringify({ runs }, null, 2)}\n`);
    } catch {
      /* best effort */
    }
  }

  async run(rawBaseUrl: unknown, rawModel: unknown, rawOptions: unknown): Promise<BenchmarkRun> {
    const baseUrl = normaliseBaseUrl(rawBaseUrl);
    if (typeof rawModel !== 'string' || rawModel.length === 0 || rawModel.length > 256) {
      throw new Error('Modello non valido');
    }
    const options = (rawOptions ?? {}) as BenchmarkOptions;
    const iterations = Math.max(1, Math.min(10, Number(options.iterations) || 3));
    const maxTokens = Math.max(16, Math.min(2048, Number(options.maxTokens) || 128));
    const prompt = typeof options.prompt === 'string' && options.prompt.trim().length > 0
      ? options.prompt.trim().slice(0, 2000)
      : 'Write a short, factual paragraph about the history of computing.';

    const startedAt = new Date().toISOString();
    const samples: BenchmarkRunSample[] = [];
    const messages = [{ role: 'user' as const, content: prompt }];
    validateMessages(messages);

    for (let iteration = 1; iteration <= iterations; iteration++) {
      const started = performance.now();
      let firstTokenAt: number | undefined;
      const result = await streamChat(baseUrl, rawModel, messages, { maxTokens }, `bench-${iteration}`, {
        onDelta: (delta) => {
          if (firstTokenAt === undefined && delta.token.length > 0) {
            firstTokenAt = performance.now() - started;
          }
        },
      });
      const totalMs = performance.now() - started;
      samples.push({
        iteration,
        tokens: result.tokens,
        ttftMs: firstTokenAt ?? totalMs,
        totalMs,
        tokPerSec: totalMs > 0 ? (result.tokens / totalMs) * 1000 : 0,
      });
    }

    const run: BenchmarkRun = {
      id: crypto.randomUUID(),
      model: rawModel,
      gatewayUrl: baseUrl,
      startedAt,
      maxTokens,
      prompt,
      samples,
      medianTokPerSec: median(samples.map((sample) => sample.tokPerSec)),
      medianTtftMs: median(samples.map((sample) => sample.ttftMs)),
      medianTokens: median(samples.map((sample) => sample.tokens)),
    };
    const runs = [run, ...this.list()].slice(0, HISTORY_LIMIT);
    this.persist(runs);
    return run;
  }

  delete(id: string): BenchmarkRun[] {
    const runs = this.list().filter((run) => run.id !== id);
    this.persist(runs);
    return runs;
  }
}
