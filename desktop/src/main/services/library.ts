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
import type { LibraryEntry, LibraryEntryKind, LibraryEntryStatus } from '../../shared/types';

const CONVERT_TIMEOUT_MS = 300_000;
const VALIDATE_TIMEOUT_MS = 120_000;

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
