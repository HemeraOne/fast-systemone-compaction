import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { Message } from '../../src/index.js';
import { familyOf } from './history.js';
import { replaySession } from '../replay/replay.js';
import type { Loss } from '../replay/replay.js';
import { parseTranscript } from '../replay/transcript.js';

/** A checked replay point with the session it came from; `losses` is empty for a kept point. */
export interface LostPoint {
  session: string;
  /** Full path of the session file; only the stub reads it. */
  path: string;
  messageIndex: number;
  toolUseId: string;
  tool: string;
  losses: Loss[];
  /** The whole session; the point's history is `messages.slice(0, messageIndex)`. */
  messages: readonly Message[];
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

/**
 * Every lost point of the corpus, ordered by session name then message index. A point whose
 * recorded message holds other calls of the same kind (parallel edits, say) is skipped and
 * counted: the model's first such call may be one of the siblings, which would score a valid
 * answer as wrong even with the full history.
 */
export function lostPoints(root: string): { points: LostPoint[]; sessions: number; skipped: number } {
  return collect(root, true);
}

/** The checked points that needed an earlier value and lost none: the model has all it needs, so a lookup there is doubt. */
export function keptPoints(root: string): { points: LostPoint[]; sessions: number; skipped: number } {
  return collect(root, false);
}

function collect(root: string, lost: boolean): { points: LostPoint[]; sessions: number; skipped: number } {
  const points: LostPoint[] = [];
  let sessions = 0;
  let skipped = 0;
  for (const path of sessionFiles(root)) {
    let text: string;
    try {
      text = readFileSync(path, 'utf8');
    } catch {
      continue;
    }
    const { messages } = parseTranscript(text);
    if (messages.length === 0) continue;
    sessions++;
    const session = path.slice(Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')) + 1);
    for (const point of replaySession(session, messages).points) {
      if (point.status !== 'checked' || (point.losses.length > 0) !== lost || (!lost && point.needed === 0)) continue;
      const siblings = messages[point.messageIndex]?.toolUses.some(
        (use) => use.tool_use_id !== point.toolUseId && familyOf(use.tool) === familyOf(point.tool),
      );
      if (siblings) {
        skipped++;
        continue;
      }
      points.push({
        session,
        path,
        messageIndex: point.messageIndex,
        toolUseId: point.toolUseId,
        tool: point.tool,
        losses: point.losses,
        messages,
      });
    }
  }
  points.sort((a, b) => (a.session < b.session ? -1 : a.session > b.session ? 1 : a.messageIndex - b.messageIndex));
  return { points, sessions, skipped };
}

/**
 * The points to run. By default `count` evenly spaced ones. With `skip`, the next `count` in
 * order after the first `skip`, so a later run can continue where an earlier one stopped.
 */
export function choose<T>(points: readonly T[], count: number, skip: number | undefined): T[] {
  return skip === undefined ? sample(points, count) : points.slice(skip, skip + count);
}

/** `count` evenly spaced entries (all of them when there are fewer). */
export function sample<T>(points: readonly T[], count: number): T[] {
  if (points.length <= count) return [...points];
  return Array.from({ length: count }, (_, i) => points[Math.floor((i * points.length) / count)]!);
}
