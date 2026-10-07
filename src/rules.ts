import { applyDecisions, messageChars, resolveOptions } from './compact.js';
import { collectToolCalls } from './state.js';
import type { CallDecision, CompactOptions, CompactResult, Message, ToolCall } from './types.js';

/** Results at or below this many characters are never shortened. */
const SHORTEN_ABOVE_CHARS = 2000;
/**
 * Results of a read up to this many characters are never shortened: a read is the usual
 * source of the text a later edit replaces, and a head and tail of 300 characters rarely
 * holds it. A replay of 328 local sessions put 37 of 44 lost edit targets in read results,
 * and this cap cuts those losses by about 45% for 5 points of median size reduction.
 */
const KEEP_READ_UP_TO_CHARS = 8000;
/** `truncatedResultText` leaves a result alone unless it saves more than this plus the head. */
const NOTE_ALLOWANCE_CHARS = 120;

/** Tools whose later success makes an earlier read of the same file stale. */
const SUPERSEDING_TOOLS = new Set(['Read', 'Write']);
/** Tools that read or change a file; a later success by a change keeps an earlier read whole. */
const FILE_TOOLS = new Set(['Read', 'Edit', 'MultiEdit', 'Write']);
const CHANGING_TOOLS = new Set(['Edit', 'MultiEdit', 'Write']);

function filePathKey(call: ToolCall, tools: ReadonlySet<string>): string | undefined {
  if (!tools.has(call.tool)) return undefined;
  const path = call.input['file_path'];
  return typeof path === 'string' ? path.replaceAll('\\', '/') : undefined;
}

function isFullRead(call: ToolCall): boolean {
  return call.input['offset'] == null && call.input['limit'] == null;
}

/**
 * Ids of the unprotected reads that a later successful call makes stale: a full
 * read by a later full read or write of the path, a partial read only by a later
 * full read. An edit or a partial read changes or shows only part of a file, so
 * the earlier read stays the source of the rest. A failed call supersedes nothing.
 */
function supersededReads(calls: readonly ToolCall[]): Set<string> {
  const later = new Map<string, { wrote: boolean; fullRead: boolean }>();
  const superseded = new Set<string>();
  for (let index = calls.length - 1; index >= 0; index--) {
    const call = calls[index]!;
    const key = filePathKey(call, SUPERSEDING_TOOLS);
    if (key === undefined) continue;
    const seen = later.get(key) ?? { wrote: false, fullRead: false };
    if (call.tool === 'Read' && !call.pinned && (isFullRead(call) ? seen.fullRead || seen.wrote : seen.fullRead)) {
      superseded.add(call.id);
    }
    if (!call.isError) {
      if (call.tool === 'Write') seen.wrote = true;
      if (call.tool === 'Read' && isFullRead(call)) seen.fullRead = true;
      later.set(key, seen);
    }
  }
  return superseded;
}

/**
 * Ids of reads of a file that a later successful edit or write of the same file follows.
 * Their text is the likely source of the next edit, so the large result rule keeps it.
 */
function editedReads(calls: readonly ToolCall[]): Set<string> {
  const changedLater = new Set<string>();
  const edited = new Set<string>();
  for (let index = calls.length - 1; index >= 0; index--) {
    const call = calls[index]!;
    const key = filePathKey(call, FILE_TOOLS);
    if (key === undefined) continue;
    if (call.tool === 'Read' && changedLater.has(key)) edited.add(call.id);
    if (CHANGING_TOOLS.has(call.tool) && !call.isError) changedLater.add(key);
  }
  return edited;
}

function decide(
  call: ToolCall,
  superseded: ReadonlySet<string>,
  edited: ReadonlySet<string>,
  resultThreshold: number,
): CallDecision {
  const base = { id: call.id, tool: call.tool };
  if (call.pinned) return { ...base, keepCall: 1, keepResult: 1, action: 'keep', reason: 'pinned' };
  if (superseded.has(call.id)) {
    return { ...base, keepCall: 0, keepResult: 0, action: 'drop_call', reason: 'call_dropped' };
  }
  const keptRead = call.tool === 'Read' && call.resultChars <= KEEP_READ_UP_TO_CHARS;
  if (!edited.has(call.id) && !keptRead && call.resultChars > resultThreshold) {
    return { ...base, keepCall: 1, keepResult: 0, action: 'drop_result', reason: 'result_dropped' };
  }
  return { ...base, keepCall: 1, keepResult: 1, action: 'keep', reason: 'kept' };
}

/**
 * Compacts a transcript with fixed rules instead of asking a backend. The rules
 * only produce `drop_result` / `drop_call` decisions for calls outside the
 * protected messages, so `applyDecisions` keeps the verbatim guarantees.
 */
export function compactByRules(
  messages: readonly Message[],
  options: CompactOptions = {},
): CompactResult {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  const resultThreshold = Math.max(
    SHORTEN_ABOVE_CHARS,
    resolved.truncateHeadChars + NOTE_ALLOWANCE_CHARS,
  );
  const superseded = supersededReads(calls);
  const edited = editedReads(calls);
  const decisions = calls.map((call) => decide(call, superseded, edited, resultThreshold));

  const kept = applyDecisions(messages, decisions, calls, resolved.truncateHeadChars);
  const count = (reason: CallDecision['reason']): number =>
    decisions.filter((decision) => decision.reason === reason).length;
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      calls: calls.length,
      kept: count('kept'),
      resultsDropped: count('result_dropped'),
      callsDropped: count('call_dropped'),
      pinned: count('pinned'),
      stateTokens: 0,
      stateStage: 'rules',
      requests: 0,
      ms: Date.now() - started,
    },
  };
}
