import { collectToolCalls, compactByRules, reductionRatio } from '../../src/index.js';
import type { CallDecision, Message, ToolUse } from '../../src/index.js';

/** A value shorter than this is too easy to match by chance to count. */
export const MIN_VALUE_CHARS = 12;
/** The same cut-off as spec 002 SC-002: shorter histories are not compacted. */
export const MIN_HISTORY_MESSAGES = 20;
export const MAX_POINTS_PER_SESSION = 10;
/** The hook's default `minReductionRatio`; below it the real hook falls back to the built-in summary. */
export const MIN_REDUCTION_RATIO = 0.25;

export type ValueKind = 'path' | 'command' | 'editTarget';
export type Rule = 'rule1' | 'rule2' | 'both';
export type ExcludedReason = 'history-too-short' | 'below-minimum-reduction';

export interface Candidate {
  kind: ValueKind;
  value: string;
  length: number;
}

export interface Loss {
  kind: ValueKind;
  rule: Rule;
  length: number;
  locator: { session: string; messageIndex: number; tool: string };
}

export interface PointResult {
  messageIndex: number;
  toolUseId: string;
  tool: string;
  status: 'checked' | 'excluded';
  excludedReason?: ExcludedReason;
  needed: number;
  neededByKind: Record<ValueKind, number>;
  losses: Loss[];
}

export interface SessionResult {
  session: string;
  messages: number;
  malformedLines: number;
  skipped: boolean;
  points: PointResult[];
}

const PATH_TOOLS = new Set(['Read', 'Edit', 'MultiEdit', 'Write']);

function candidate(kind: ValueKind, value: unknown): Candidate[] {
  return typeof value === 'string' && value.length >= MIN_VALUE_CHARS
    ? [{ kind, value, length: value.length }]
    : [];
}

/** The values a tool call used that the assistant must have taken from earlier context. */
export function candidatesOf(tool: string, input: Record<string, unknown>): Candidate[] {
  const found: Candidate[] = [];
  if (PATH_TOOLS.has(tool)) found.push(...candidate('path', input['file_path']));
  if (tool === 'Bash') found.push(...candidate('command', input['command']));
  if (tool === 'Edit') found.push(...candidate('editTarget', input['old_string']));
  if (tool === 'MultiEdit' && Array.isArray(input['edits'])) {
    for (const edit of input['edits']) {
      if (typeof edit === 'object' && edit !== null) {
        found.push(...candidate('editTarget', (edit as Record<string, unknown>)['old_string']));
      }
    }
  }
  return found;
}

function stringLeaves(value: unknown, out: string[]): void {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) for (const item of value) stringLeaves(item, out);
  else if (typeof value === 'object' && value !== null) {
    for (const item of Object.values(value)) stringLeaves(item, out);
  }
}

/** Every text block of a message, with the tool call it belongs to (none for plain text). */
function blocks(message: Message): { owner: string | undefined; text: string }[] {
  const found: { owner: string | undefined; text: string }[] = [{ owner: undefined, text: message.text }];
  for (const use of message.toolUses) {
    const leaves: string[] = [];
    stringLeaves(use.input, leaves);
    if (use.text !== undefined) leaves.push(use.text);
    for (const text of leaves) found.push({ owner: use.tool_use_id, text });
  }
  for (const result of message.toolResults ?? []) {
    found.push({ owner: result.tool_use_id, text: result.text });
  }
  return found;
}

function occursIn(messages: readonly Message[], value: string): boolean {
  return messages.some((message) => blocks(message).some((block) => block.text.includes(value)));
}

function attribute(
  prefix: readonly Message[],
  value: string,
  reasonByToolUseId: ReadonlyMap<string, CallDecision['reason']>,
): Rule {
  const rules = new Set<'rule1' | 'rule2'>();
  for (const message of prefix) {
    for (const block of blocks(message)) {
      if (block.owner === undefined || !block.text.includes(value)) continue;
      const reason = reasonByToolUseId.get(block.owner);
      if (reason === 'result_dropped') rules.add('rule1');
      else if (reason === 'call_dropped') rules.add('rule2');
    }
  }
  if (rules.size === 0) {
    throw new Error('a lost value had no occurrence in a call the rules touched');
  }
  return rules.size === 2 ? 'both' : [...rules][0]!;
}

function countByKind(candidates: readonly Candidate[]): Record<ValueKind, number> {
  const counts: Record<ValueKind, number> = { path: 0, command: 0, editTarget: 0 };
  for (const c of candidates) counts[c.kind]++;
  return counts;
}

function excluded(raw: RawPoint, excludedReason: ExcludedReason): PointResult {
  return {
    messageIndex: raw.messageIndex,
    toolUseId: raw.use.tool_use_id,
    tool: raw.use.tool,
    status: 'excluded',
    excludedReason,
    needed: 0,
    neededByKind: countByKind([]),
    losses: [],
  };
}

interface RawPoint {
  messageIndex: number;
  use: ToolUse;
  candidates: Candidate[];
}

function evenlySpaced<T>(items: readonly T[], count: number): T[] {
  if (items.length <= count) return [...items];
  return Array.from({ length: count }, (_, i) => items[Math.floor((i * items.length) / count)]!);
}

function checkPoint(session: string, messages: readonly Message[], raw: RawPoint): PointResult {
  const prefix = messages.slice(0, raw.messageIndex);
  const compacted = compactByRules(prefix);
  if (reductionRatio(compacted) < MIN_REDUCTION_RATIO) return excluded(raw, 'below-minimum-reduction');

  const calls = collectToolCalls(prefix, 6);
  const decisionById = new Map(compacted.decisions.map((decision) => [decision.id, decision]));
  const reasonByToolUseId = new Map<string, CallDecision['reason']>();
  for (const call of calls) {
    const decision = decisionById.get(call.id);
    if (decision) reasonByToolUseId.set(call.tool_use_id, decision.reason);
  }

  const needed = raw.candidates.filter((c) => occursIn(prefix, c.value));
  const losses: Loss[] = needed
    .filter((c) => !occursIn(compacted.messages, c.value))
    .map((c) => ({
      kind: c.kind,
      rule: attribute(prefix, c.value, reasonByToolUseId),
      length: c.length,
      locator: { session, messageIndex: raw.messageIndex, tool: raw.use.tool },
    }));
  return {
    messageIndex: raw.messageIndex,
    toolUseId: raw.use.tool_use_id,
    tool: raw.use.tool,
    status: 'checked',
    needed: needed.length,
    neededByKind: countByKind(needed),
    losses,
  };
}

/**
 * Replays rule-based compaction at points of one session: the history before a tool call
 * is compacted, and every value that call used and that occurred in that history is
 * checked for survival.
 */
export function replaySession(session: string, messages: readonly Message[]): SessionResult {
  const raw: RawPoint[] = [];
  messages.forEach((message, messageIndex) => {
    if (message.role !== 'assistant') return;
    for (const use of message.toolUses) {
      const candidates = candidatesOf(use.tool, use.input);
      if (candidates.length > 0) raw.push({ messageIndex, use, candidates });
    }
  });

  const tooShort = raw.filter((point) => point.messageIndex < MIN_HISTORY_MESSAGES);
  const eligible = evenlySpaced(
    raw.filter((point) => point.messageIndex >= MIN_HISTORY_MESSAGES),
    MAX_POINTS_PER_SESSION,
  );

  const points: PointResult[] = [
    ...tooShort.map((point) => excluded(point, 'history-too-short')),
    ...eligible.map((point) => checkPoint(session, messages, point)),
  ];
  return { session, messages: messages.length, malformedLines: 0, skipped: false, points };
}
