import { applyDecisions, messageChars, resolveOptions } from './compact.js';
import { collectToolCalls } from './state.js';
import type { CallDecision, CompactOptions, CompactResult, Message, ToolCall } from './types.js';

/** Results at or below this many characters are never shortened. */
const SHORTEN_ABOVE_CHARS = 2000;
/** `truncatedResultText` leaves a result alone unless it saves more than this plus the head. */
const NOTE_ALLOWANCE_CHARS = 120;

const FILE_TOOLS = new Set(['Read', 'Edit', 'Write']);

function filePathKey(call: ToolCall): string | undefined {
  if (!FILE_TOOLS.has(call.tool)) return undefined;
  const path = call.input['file_path'];
  return typeof path === 'string' ? path.replaceAll('\\', '/') : undefined;
}

function isFullRead(call: ToolCall): boolean {
  return call.input['offset'] == null && call.input['limit'] == null;
}

/**
 * Ids of the unprotected reads that a later successful call makes stale: a full
 * read by any later read, edit or write of the path, a partial read only by a
 * later full read. A failed call supersedes nothing.
 */
function supersededReads(calls: readonly ToolCall[]): Set<string> {
  const later = new Map<string, { touched: boolean; fullRead: boolean }>();
  const superseded = new Set<string>();
  for (let index = calls.length - 1; index >= 0; index--) {
    const call = calls[index]!;
    const key = filePathKey(call);
    if (key === undefined) continue;
    const seen = later.get(key) ?? { touched: false, fullRead: false };
    if (call.tool === 'Read' && !call.pinned && (isFullRead(call) ? seen.touched : seen.fullRead)) {
      superseded.add(call.id);
    }
    if (!call.isError) {
      seen.touched = true;
      if (call.tool === 'Read' && isFullRead(call)) seen.fullRead = true;
      later.set(key, seen);
    }
  }
  return superseded;
}

function decide(call: ToolCall, superseded: ReadonlySet<string>, resultThreshold: number): CallDecision {
  const base = { id: call.id, tool: call.tool };
  if (call.pinned) return { ...base, keepCall: 1, keepResult: 1, action: 'keep', reason: 'pinned' };
  if (superseded.has(call.id)) {
    return { ...base, keepCall: 0, keepResult: 0, action: 'drop_call', reason: 'call_dropped' };
  }
  if (call.resultChars > resultThreshold) {
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
  const decisions = calls.map((call) => decide(call, superseded, resultThreshold));

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
