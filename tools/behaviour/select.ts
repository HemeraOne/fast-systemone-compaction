import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type { Message } from '../../src/index.js';
import { replaySession } from '../replay/replay.js';
import type { Loss } from '../replay/replay.js';
import { parseTranscript } from '../replay/transcript.js';

/** A checked replay point with at least one loss, with the session it came from. */
export interface LostPoint {
  session: string;
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

/** Every lost point of the corpus, ordered by session name then message index. */
export function lostPoints(root: string): { points: LostPoint[]; sessions: number } {
  const points: LostPoint[] = [];
  let sessions = 0;
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
      if (point.status !== 'checked' || point.losses.length === 0) continue;
      points.push({
        session,
        messageIndex: point.messageIndex,
        toolUseId: point.toolUseId,
        tool: point.tool,
        losses: point.losses,
        messages,
      });
    }
  }
  points.sort((a, b) => (a.session < b.session ? -1 : a.session > b.session ? 1 : a.messageIndex - b.messageIndex));
  return { points, sessions };
}

/** `count` evenly spaced entries (all of them when there are fewer). */
export function sample<T>(points: readonly T[], count: number): T[] {
  if (points.length <= count) return [...points];
  return Array.from({ length: count }, (_, i) => points[Math.floor((i * points.length) / count)]!);
}
