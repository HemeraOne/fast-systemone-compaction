import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { formatReport, summarize } from './report.js';
import { replaySession } from './replay.js';
import type { SessionResult } from './replay.js';
import { parseTranscript } from './transcript.js';

function optionValue(args: readonly string[], flag: string, needs: string): string | undefined {
  const at = args.indexOf(flag);
  const value = at >= 0 ? args[at + 1] : undefined;
  if (at >= 0 && value === undefined) throw new Error(`${flag} needs ${needs}`);
  return value;
}

function rootFromArgs(args: readonly string[]): string {
  return optionValue(args, '--root', 'a directory') ?? join(homedir(), '.claude', 'projects');
}

/** `--before <date>` as epoch ms. A bare date such as 2026-10-04 means 00:00 UTC. */
function beforeFromArgs(args: readonly string[]): number | undefined {
  const value = optionValue(args, '--before', 'a date such as 2026-10-04');
  if (value === undefined) return undefined;
  const ms = Date.parse(value);
  if (Number.isNaN(ms)) throw new Error(`--before: cannot read "${value}" as a date`);
  return ms;
}

/**
 * `<root>/<project>/*.jsonl`, sorted so every run visits the files in the same order. With
 * `before`, only files last modified before that instant, which approximates the corpus as
 * it was then (files deleted since cannot come back).
 */
function sessionFiles(root: string, before: number | undefined): string[] {
  const files: string[] = [];
  for (const project of readdirSync(root).sort()) {
    const dir = join(root, project);
    if (!statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir).sort()) {
      if (!name.endsWith('.jsonl')) continue;
      const path = join(dir, name);
      if (before !== undefined && statSync(path).mtimeMs >= before) continue;
      files.push(path);
    }
  }
  return files;
}

function replayFile(path: string): SessionResult {
  const session = path.slice(Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')) + 1);
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch {
    return { session, messages: 0, malformedLines: 0, skipped: true, reduction: undefined, points: [] };
  }
  const { messages, malformedLines } = parseTranscript(text);
  if (messages.length === 0) {
    return { session, messages: 0, malformedLines, skipped: true, reduction: undefined, points: [] };
  }
  return { ...replaySession(session, messages), malformedLines };
}

function main(): number {
  const args = process.argv.slice(2);
  const root = rootFromArgs(args);
  const before = beforeFromArgs(args);
  if (!existsSync(root)) {
    process.stderr.write(`replay: ${root} does not exist\n`);
    return 1;
  }
  const results = sessionFiles(root, before).map(replayFile);
  const summary = summarize(results);
  if (summary.sessionsRead === 0) {
    process.stderr.write('replay: no readable session found\n');
    return 1;
  }
  process.stdout.write(formatReport(summary));
  return 0;
}

try {
  process.exitCode = main();
} catch (error) {
  process.stderr.write(`replay: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
