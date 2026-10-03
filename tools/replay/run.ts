import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { formatReport, summarize } from './report.js';
import { replaySession } from './replay.js';
import type { SessionResult } from './replay.js';
import { parseTranscript } from './transcript.js';

function rootFromArgs(args: readonly string[]): string {
  const at = args.indexOf('--root');
  const value = at >= 0 ? args[at + 1] : undefined;
  if (at >= 0 && value === undefined) throw new Error('--root needs a directory');
  return value ?? join(homedir(), '.claude', 'projects');
}

/** `<root>/<project>/*.jsonl`, sorted so every run visits the files in the same order. */
function sessionFiles(root: string): string[] {
  const files: string[] = [];
  for (const project of readdirSync(root).sort()) {
    const dir = join(root, project);
    if (!statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir).sort()) {
      if (name.endsWith('.jsonl')) files.push(join(dir, name));
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
    return { session, messages: 0, malformedLines: 0, skipped: true, points: [] };
  }
  const { messages, malformedLines } = parseTranscript(text);
  if (messages.length === 0) return { session, messages: 0, malformedLines, skipped: true, points: [] };
  return { ...replaySession(session, messages), malformedLines };
}

function main(): number {
  const root = rootFromArgs(process.argv.slice(2));
  if (!existsSync(root)) {
    process.stderr.write(`replay: ${root} does not exist\n`);
    return 1;
  }
  const results = sessionFiles(root).map(replayFile);
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
