/*
 * ProsperoAI Desktop — model library and import pipeline (§7/§8).
 *
 * Imports are staged into userData/models/library and, when the `pai`
 * toolchain binary is available (PAI_BIN env var or PATH), converted
 * for real: `pai convert` produces the .pai container from GGUF and
 * `pai validate` verifies its integrity. The library state is
 * persisted so the Model library view shows actual files and pipeline
 * progress — no fabricated models.
 */

import crypto from 'node:crypto';
import { execFile, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import type { InspectData, InspectResult, LibraryEntry, LibraryEntryKind, LibraryEntryStatus, OptimizePlan, OptimizeResult, OptimizeScheme, OptimizeTensor } from '../../shared/types';

const CONVERT_TIMEOUT_MS = 300_000;
const VALIDATE_TIMEOUT_MS = 120_000;
const OPTIMIZE_TIMEOUT_MS = 120_000;
const INSPECT_TIMEOUT_MS = 60_000;

/* Wire shape of `pai optimize --json` (snake_case fields). */
interface RawOptimizeTensor {
  name?: unknown;
  role?: unknown;
  value_id?: unknown;
  numel?: unknown;
  current_scheme?: unknown;
  current_bytes?: unknown;
  plan_scheme?: unknown;
  plan_bytes?: unknown;
}

interface RawOptimizePlan {
  model?: unknown;
  current_bytes?: unknown;
  plan_bytes?: unknown;
  budget?: unknown;
  min_bytes?: unknown;
  feasible?: unknown;
  tensors?: RawOptimizeTensor[];
}

/* Wire shape of `pai inspect --json` (snake_case fields). */
interface RawInspect {
  container?: {
    path?: unknown;
    bytes?: unknown;
    version?: unknown;
    sections?: Array<{ type?: unknown; offset?: unknown; size?: unknown }>;
  };
  meta?: {
    name?: unknown;
    family?: unknown;
    context_len?: unknown;
    num_layers?: unknown;
    kv_bytes_per_token?: unknown;
    vocab_size?: unknown;
  };
  tensors?: Array<{
    name?: unknown;
    value_id?: unknown;
    dtype?: unknown;
    rank?: unknown;
    shape?: unknown[];
    offset?: unknown;
    size_bytes?: unknown;
  }>;
  ir?: {
    values?: unknown;
    inputs?: unknown;
    outputs?: unknown;
    ops?: Array<{ id?: unknown; kind?: unknown; inputs?: unknown; outputs?: unknown }>;
  };
  tokenizer?: { tokens?: unknown; merges?: unknown };
}

export interface PaiCliInfo {
  path: string;
  available: boolean;
}

export class ModelLibrary {
  private readonly libraryDir: string;
  private readonly statePath: string;
  private state: LibraryEntry[] = [];

  constructor(baseDir: string) {
    this.libraryDir = path.join(baseDir, 'models', 'library');
    this.statePath = path.join(this.libraryDir, 'library.json');
  }

  init(): void {
    try {
      this.state = JSON.parse(fs.readFileSync(this.statePath, 'utf8')) as LibraryEntry[];
      if (!Array.isArray(this.state)) this.state = [];
    } catch {
      this.state = [];
    }
  }

  list(): LibraryEntry[] {
    return [...this.state];
  }

  get libraryPath(): string {
    return this.libraryDir;
  }

  private persist(): void {
    try {
      fs.mkdirSync(this.libraryDir, { recursive: true });
      fs.writeFileSync(this.statePath, `${JSON.stringify(this.state, null, 2)}\n`);
    } catch {
      /* best effort */
    }
  }

  private upsert(entry: LibraryEntry): void {
    const index = this.state.findIndex((item) => item.id === entry.id);
    if (index >= 0) this.state[index] = entry;
    else this.state.unshift(entry);
    this.persist();
  }

  private kindFor(filename: string): LibraryEntryKind {
    if (/\.pai$/i.test(filename)) return 'pai';
    if (/\.gguf$/i.test(filename)) return 'gguf';
    if (/\.safetensors$/i.test(filename)) return 'safetensors';
    if (/\.(bin|pt|pth)$/i.test(filename)) return 'bin';
    return 'unknown';
  }

  private safeName(filename: string, index: number): string {
    const base = filename.replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 120) || `model-${index}`;
    return `${Date.now()}-${index}-${base}`;
  }

  /*
   * Locate the `pai` toolchain binary: PAI_BIN env var first, then
   * PATH lookup. Returns null when the toolchain is not available.
   */
  findPaiCli(): PaiCliInfo | null {
    const candidate = process.env.PAI_BIN;
    if (candidate && candidate.trim().length > 0) {
      if (fs.existsSync(candidate)) return { path: candidate, available: true };
    }
    try {
      const command = process.platform === 'win32' ? 'where' : 'which';
      const found = execFileSync(command, ['pai'], { stdio: ['ignore', 'pipe', 'ignore'], timeout: 5000 })
        .toString().split(/\r?\n/).map((line) => line.trim()).find((line) => line.length > 0);
      if (found) return { path: found, available: true };
    } catch {
      /* not on PATH */
    }
    return null;
  }

  private runPai(cli: PaiCliInfo, args: string[], timeoutMs: number): Promise<{ code: number; stdout: string; stderr: string }> {
    return new Promise((resolve) => {
      execFile(cli.path, args, { timeout: timeoutMs, windowsHide: true, maxBuffer: 8 * 1024 * 1024 }, (error, stdout, stderr) => {
        resolve({ code: error && typeof error === 'object' && 'code' in error && typeof error.code === 'number' ? error.code : error ? 1 : 0, stdout, stderr });
      });
    });
  }

  /*
   * Mixed-precision plan (§15/§20): `pai optimize --json` over a
   * managed .pai entry. Returns the parsed plan (per-tensor f32/q8/q4
   * sizes + totals) or an error. The CLI prints a version banner
   * before the JSON, so parsing starts at the first '{'.
   */
  async optimize(entryId: string): Promise<OptimizeResult> {
    const entry = this.state.find((item) => item.id === entryId);
    if (!entry || entry.kind !== 'pai' || !entry.libraryPath) {
      return { ok: false, error: 'nessun container .pai pronto per questa voce' };
    }
    const cli = this.findPaiCli();
    if (!cli) {
      return { ok: false, error: 'CLI `pai` non trovata (imposta PAI_BIN o aggiungi il binario al PATH)' };
    }
    const result = await this.runPai(cli, ['optimize', entry.libraryPath, '--json'], OPTIMIZE_TIMEOUT_MS);
    if (result.code !== 0) {
      return { ok: false, error: (result.stderr || result.stdout || 'pai optimize fallito').trim().slice(0, 500) };
    }
    try {
      const start = result.stdout.indexOf('{');
      if (start < 0) throw new Error('risposta priva di JSON');
      const parsed = JSON.parse(result.stdout.slice(start)) as RawOptimizePlan;
      if (!Array.isArray(parsed.tensors)) throw new Error('piano malformato');
      const tensors: OptimizeTensor[] = parsed.tensors.map((tensor) => ({
        name: String(tensor.name ?? '?'),
        role: (['embedding', 'attention', 'mlp', 'output', 'norm', 'other'].includes(String(tensor.role)) ? String(tensor.role) : 'other') as OptimizeTensor['role'],
        valueId: Number(tensor.value_id ?? 0),
        numel: Number(tensor.numel ?? 0),
        currentScheme: (['f32', 'q8', 'q4'].includes(String(tensor.current_scheme)) ? String(tensor.current_scheme) : 'f32') as OptimizeScheme,
        currentBytes: Number(tensor.current_bytes ?? 0),
        planScheme: (['f32', 'q8', 'q4'].includes(String(tensor.plan_scheme)) ? String(tensor.plan_scheme) : 'f32') as OptimizeScheme,
        planBytes: Number(tensor.plan_bytes ?? 0),
      }));
      return {
        ok: true,
        plan: {
          model: String(parsed.model ?? entry.name),
          currentBytes: Number(parsed.current_bytes ?? 0),
          planBytes: Number(parsed.plan_bytes ?? 0),
          budget: parsed.budget == null ? undefined : Number(parsed.budget),
          minBytes: Number(parsed.min_bytes ?? 0),
          feasible: parsed.feasible !== false,
          tensors,
        },
      };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : 'parse del piano fallito' };
    }
  }

  /*
   * Container manifest report (§20/§7): `pai inspect --json` over a
   * managed .pai entry. Returns the parsed container/meta/tensor/IR
   * report or an error.
   */
  async inspect(entryId: string): Promise<InspectResult> {
    const entry = this.state.find((item) => item.id === entryId);
    if (!entry || entry.kind !== 'pai' || !entry.libraryPath) {
      return { ok: false, error: 'nessun container .pai pronto per questa voce' };
    }
    const cli = this.findPaiCli();
    if (!cli) {
      return { ok: false, error: 'CLI `pai` non trovata (imposta PAI_BIN o aggiungi il binario al PATH)' };
    }
    const result = await this.runPai(cli, ['inspect', entry.libraryPath, '--json'], INSPECT_TIMEOUT_MS);
    if (result.code !== 0) {
      return { ok: false, error: (result.stderr || result.stdout || 'pai inspect fallito').trim().slice(0, 500) };
    }
    try {
      const start = result.stdout.indexOf('{');
      if (start < 0) throw new Error('risposta priva di JSON');
      const parsed = JSON.parse(result.stdout.slice(start)) as RawInspect;
      const container = parsed.container ?? {};
      const sections = Array.isArray(container.sections)
        ? container.sections.map((section) => ({
            type: String(section.type ?? '?'),
            offset: Number(section.offset ?? 0),
            size: Number(section.size ?? 0),
          }))
        : [];
      const tensors = Array.isArray(parsed.tensors)
        ? parsed.tensors.map((tensor) => ({
            name: String(tensor.name ?? '?'),
            valueId: Number(tensor.value_id ?? 0),
            dtype: String(tensor.dtype ?? '?'),
            rank: Number(tensor.rank ?? 0),
            shape: Array.isArray(tensor.shape) ? tensor.shape.map((dim) => Number(dim) || 0) : [],
            offset: Number(tensor.offset ?? 0),
            sizeBytes: Number(tensor.size_bytes ?? 0),
          }))
        : [];
      const data: InspectData = {
        path: String(container.path ?? entry.libraryPath),
        bytes: Number(container.bytes ?? 0),
        version: Number(container.version ?? 0),
        sections,
        tensors,
      };
      if (parsed.meta) {
        data.meta = {
          name: String(parsed.meta.name ?? entry.name),
          family: Number(parsed.meta.family ?? 0),
          contextLen: Number(parsed.meta.context_len ?? 0),
          numLayers: Number(parsed.meta.num_layers ?? 0),
          kvBytesPerToken: Number(parsed.meta.kv_bytes_per_token ?? 0),
          vocabSize: Number(parsed.meta.vocab_size ?? 0),
        };
      }
      if (parsed.ir) {
        data.ir = {
          values: Number(parsed.ir.values ?? 0),
          inputs: Number(parsed.ir.inputs ?? 0),
          outputs: Number(parsed.ir.outputs ?? 0),
          ops: Array.isArray(parsed.ir.ops)
            ? parsed.ir.ops.map((op, index) => ({
                id: Number(op.id ?? index + 1),
                kind: String(op.kind ?? '?'),
                inputs: Number(op.inputs ?? 0),
                outputs: Number(op.outputs ?? 0),
              }))
            : [],
        };
      }
      if (parsed.tokenizer) {
        data.tokenizer = {
          tokens: Number(parsed.tokenizer.tokens ?? 0),
          merges: Number(parsed.tokenizer.merges ?? 0),
        };
      }
      return { ok: true, data };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : 'parse del report fallito' };
    }
  }

  /*
   * Stage model files into the managed library and run the real
   * conversion pipeline where the toolchain allows it:
   *   .pai          -> validate (when the CLI exists)
   *   .gguf         -> pai convert <file> <out.pai> (+ validate)
   *   safetensors   -> staged; conversion needs a model description
   *                    (the Rust toolchain integration, next step)
   */
  async importFiles(sourcePaths: string[]): Promise<LibraryEntry[]> {
    const cli = this.findPaiCli();
    const created: LibraryEntry[] = [];
    let index = 0;

    for (const sourcePath of sourcePaths) {
      let stat: fs.Stats;
      try {
        stat = fs.statSync(sourcePath);
        if (!stat.isFile()) continue;
      } catch {
        continue;
      }
      const filename = path.basename(sourcePath);
      const kind = this.kindFor(filename);
      const id = crypto.randomUUID();
      const libraryPath = path.join(this.libraryDir, this.safeName(filename, index));
      index++;
      try {
        fs.mkdirSync(this.libraryDir, { recursive: true });
        fs.copyFileSync(sourcePath, libraryPath);
      } catch (error) {
        this.upsert({
          id, name: filename, kind, sourcePath, sizeBytes: stat.size,
          status: 'failed', message: error instanceof Error ? error.message : 'copia del file fallita',
          updatedAt: new Date().toISOString(),
        });
        continue;
      }

      const entry: LibraryEntry = {
        id, name: filename, kind, sourcePath, libraryPath,
        sizeBytes: stat.size, status: 'imported', updatedAt: new Date().toISOString(),
      };

      if (kind === 'pai') {
        if (cli) {
          entry.status = 'converting';
          this.upsert(entry);
          const result = await this.runPai(cli, ['validate', libraryPath], VALIDATE_TIMEOUT_MS);
          entry.status = result.code === 0 ? 'ready' : 'failed';
          entry.message = result.code === 0 ? undefined : (result.stderr || result.stdout || 'pai validate fallito').trim().slice(0, 500);
        } else {
          entry.status = 'ready';
          entry.message = 'validazione manuale (CLI pai non trovata)';
        }
      } else if (kind === 'gguf' && cli) {
        entry.status = 'converting';
        this.upsert(entry);
        const outPai = libraryPath.replace(/\.gguf$/i, '') + '.pai';
        const convert = await this.runPai(cli, ['convert', libraryPath, outPai], CONVERT_TIMEOUT_MS);
        if (convert.code === 0) {
          const validate = await this.runPai(cli, ['validate', outPai], VALIDATE_TIMEOUT_MS);
          entry.status = validate.code === 0 ? 'ready' : 'failed';
          entry.libraryPath = outPai;
          entry.message = validate.code === 0 ? undefined : (validate.stderr || 'pai validate fallito').trim().slice(0, 500);
          /* The .pai is the artifact that matters: drop the staged
           * GGUF copy once the conversion is verified. */
          if (validate.code === 0) {
            try { fs.rmSync(libraryPath); } catch { /* best effort */ }
          }
        } else {
          entry.status = 'failed';
          entry.message = (convert.stderr || convert.stdout || 'pai convert fallito').trim().slice(0, 500);
        }
      } else {
        entry.message = kind === 'gguf'
          ? 'conversione in attesa (CLI pai non trovata)'
          : 'conversione richiede il toolchain Rust (descrittore modello)';
      }
      this.upsert(entry);
      created.push(entry);
    }
    return created;
  }
}
