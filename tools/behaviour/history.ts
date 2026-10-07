import { compactByRules } from '../../src/index.js';
import type { Message, ToolUse } from '../../src/index.js';
import { candidatesOf, withoutLineNumbers } from '../replay/replay.js';
import type { LostPoint } from './select.js';

/** The history the assistant had before the recorded step, uncompacted. */
export function controlHistory(messages: readonly Message[], k: number): Message[] {
  return messages.slice(0, k);
}

/** The same history after rule-based compaction with default options, as the replay judged it. */
export function compactedHistory(messages: readonly Message[], k: number): Message[] {
  return compactByRules(controlHistory(messages, k)).messages;
}

const INTRO =
  'Below is the conversation so far between a user and an assistant working in a software project, ' +
  'including the tool calls the assistant made and what they returned. Parts of long tool results ' +
  'may have been left out and marked as such.';

const OUTRO =
  'Continue as the assistant: take the next step now by calling one of your tools (Read, Grep, Glob, ' +
  'Edit or Bash, which may be listed with an mcp__stub__ prefix). Do not describe what you would do.';

/**
 * The messages as a labelled text transcript. The CLI cannot take earlier tool-call turns as
 * structured input, so a history reaches the model as text (research R3); both arms are
 * rendered the same way.
 */
export function renderTranscript(messages: readonly Message[]): string {
  const names = new Map<string, string>();
  const parts: string[] = [];
  for (const message of messages) {
    if (message.text !== '') parts.push(`=== ${message.role} ===\n${message.text}`);
    for (const use of message.toolUses) {
      names.set(use.tool_use_id, use.tool);
      parts.push(`--- assistant tool call: ${use.tool} ---\n${JSON.stringify(use.input)}`);
    }
    for (const result of message.toolResults ?? []) {
      const name = names.get(result.tool_use_id) ?? 'tool';
      parts.push(`--- tool result: ${name}${result.isError === true ? ' (error)' : ''} ---\n${result.text}`);
    }
  }
  return parts.join('\n\n');
}

const framed = (body: string): string => `${INTRO}\n\n${body}\n\n${OUTRO}`;

/** The prompt of an arm that shows a history: the transcript between the intro and the request. */
export function flattenHistory(messages: readonly Message[]): string {
  return framed(renderTranscript(messages));
}

/** The fixed request that produces the approximate built-in-summary arm. */
export const SUMMARY_REQUEST =
  'Summarize the conversation so far so that work can continue from your summary alone: the goal, ' +
  'what has been done, the files involved, and what remains.';

/** The prompt of the tool-less call that writes the summary of a history. */
export function summaryPrompt(messages: readonly Message[]): string {
  return `${renderTranscript(messages)}\n\n${SUMMARY_REQUEST}`;
}

/** The summary arm's prompt: the summary stands in for the whole history. */
export function summaryHistory(summary: string): string {
  return framed(`=== summary of the conversation so far ===\n${summary}`);
}

export interface ToolCallRequest {
  tool: string;
  input: Record<string, unknown>;
}

const FAMILY: Readonly<Record<string, string>> = {
  Read: 'read',
  Edit: 'write',
  MultiEdit: 'write',
  Write: 'write',
  Bash: 'bash',
};

/** Tools that count as the same kind of step (an Edit stands in for a recorded Write). */
export function familyOf(tool: string): string | undefined {
  return FAMILY[tool];
}

const WRITING_TOOLS = new Set(['Edit', 'MultiEdit', 'Write']);

const slashed = (value: unknown): string | undefined =>
  typeof value === 'string' ? value.replace(/\\/g, '/') : undefined;

const collapsed = (value: unknown): string | undefined =>
  typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : undefined;

function stringLeaves(value: unknown, out: string[]): void {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) for (const item of value) stringLeaves(item, out);
  else if (typeof value === 'object' && value !== null) {
    for (const item of Object.values(value)) stringLeaves(item, out);
  }
}

/** Every text of the messages, the same haystack the replay searches. */
function textsOf(messages: readonly Message[]): string[] {
  const texts: string[] = [];
  for (const message of messages) {
    texts.push(message.text);
    for (const use of message.toolUses) {
      stringLeaves(use.input, texts);
      if (use.text !== undefined) texts.push(use.text);
    }
    for (const result of message.toolResults ?? []) texts.push(result.text, withoutLineNumbers(result.text));
  }
  return texts;
}

const occurs = (texts: readonly string[], value: string): boolean => texts.some((text) => text.includes(value));

function lookupKey(tool: string, input: Record<string, unknown>): string | undefined {
  if (tool === 'Read') {
    const path = slashed(input['file_path']);
    return path === undefined ? undefined : `Read\0${path}`;
  }
  if (tool === 'Bash') {
    const command = collapsed(input['command']);
    return command === undefined ? undefined : `Bash\0${command}`;
  }
  if (tool === 'Grep' || tool === 'Glob') {
    return `${tool}\0${JSON.stringify(Object.entries(input).sort(([a], [b]) => (a < b ? -1 : 1)))}`;
  }
  return undefined;
}

/** What the stub tools can answer: recorded results of the uncompacted prefix (research R4). */
export interface Lookup {
  /** The recorded result for the call, or `undefined` when the test cannot answer it. */
  serve(call: ToolCallRequest): string | undefined;
  /** Every result `serve` could return. */
  servable: readonly string[];
  /** The label for a call `serve` cannot answer: the tool name, with a reason for a Read. */
  miss(call: ToolCallRequest): string;
}

/** Whether two paths differ only in case, a drive prefix or a relative-versus-absolute spelling. */
function respelled(a: string, b: string): boolean {
  const [x, y] = [a.toLowerCase().replace(/^\.\//, ''), b.toLowerCase().replace(/^\.\//, '')];
  return x === y || x.endsWith(`/${y}`) || y.endsWith(`/${x}`);
}

export function buildLookup(prefix: readonly Message[]): Lookup {
  const uses = new Map<string, ToolUse>();
  const entries = new Map<string, { text: string; index: number; path: string | undefined }>();
  const lastWrite = new Map<string, number>();

  prefix.forEach((message, index) => {
    for (const use of message.toolUses) uses.set(use.tool_use_id, use);
    for (const result of message.toolResults ?? []) {
      const use = uses.get(result.tool_use_id);
      if (use === undefined || result.isError === true) continue;
      const path = slashed(use.input['file_path']);
      if (WRITING_TOOLS.has(use.tool) && path !== undefined) lastWrite.set(path, index);
      const key = lookupKey(use.tool, use.input);
      if (key !== undefined) entries.set(key, { text: result.text, index, path: use.tool === 'Read' ? path : undefined });
    }
  });

  const current = (entry: { index: number; path: string | undefined }): boolean => {
    const written = entry.path === undefined ? undefined : lastWrite.get(entry.path);
    return written === undefined || written < entry.index;
  };
  const servable = [...entries.values()].filter(current).flatMap((entry) => [entry.text, withoutLineNumbers(entry.text)]);

  return {
    servable,
    serve(call) {
      const key = lookupKey(call.tool, call.input);
      const entry = key === undefined ? undefined : entries.get(key);
      return entry !== undefined && current(entry) ? entry.text : undefined;
    },
    miss(call) {
      if (call.tool !== 'Read') return call.tool;
      const path = slashed(call.input['file_path']);
      const key = lookupKey(call.tool, call.input);
      if (key !== undefined && entries.has(key)) return 'Read (stale)';
      const recorded = [...entries.values()].some((entry) => path !== undefined && entry.path !== undefined && respelled(path, entry.path));
      return recorded ? 'Read (respelled)' : 'Read (never read)';
    },
  };
}

/** Everything a point's arms share: the recorded step, its uncompacted history, the lookups. */
export interface PointContext {
  step: ToolUse;
  prefix: readonly Message[];
  lookup: Lookup;
  /** Every lost value occurs in a result the stubs can re-serve (research R5). */
  reachable: boolean;
}

export function pointContext(point: Pick<LostPoint, 'messages' | 'messageIndex' | 'toolUseId'>): PointContext {
  const step = point.messages[point.messageIndex]?.toolUses.find((use) => use.tool_use_id === point.toolUseId);
  if (step === undefined) throw new Error('the recorded step is missing from its session');
  const prefix = controlHistory(point.messages, point.messageIndex);
  const kept = textsOf(compactedHistory(point.messages, point.messageIndex));
  const all = textsOf(prefix);
  const lost = candidatesOf(step.tool, step.input).filter((c) => occurs(all, c.value) && !occurs(kept, c.value));
  const lookup = buildLookup(prefix);
  return { step, prefix, lookup, reachable: lost.every((c) => occurs(lookup.servable, c.value)) };
}

function recordedOldStrings(step: ToolUse): string[] {
  const found: string[] = [];
  const one = step.input['old_string'];
  if (typeof one === 'string') found.push(one);
  const edits = step.input['edits'];
  if (Array.isArray(edits)) {
    for (const edit of edits) {
      const old = typeof edit === 'object' && edit !== null ? (edit as Record<string, unknown>)['old_string'] : undefined;
      if (typeof old === 'string') found.push(old);
    }
  }
  return found;
}

/**
 * Whether a proposed call is the recorded step (research R5): same path after slash
 * normalisation, same command after whitespace collapse, and an edit target that occurs in
 * a recorded text that held the recorded target.
 */
export function matchesRecorded(call: ToolCallRequest, step: ToolUse, prefix: readonly Message[]): boolean {
  if (familyOf(call.tool) !== familyOf(step.tool)) return false;
  const path = slashed(step.input['file_path']);
  if (path !== undefined && slashed(call.input['file_path']) !== path) return false;
  const command = collapsed(step.input['command']);
  if (command !== undefined && collapsed(call.input['command']) !== command) return false;
  const recorded = recordedOldStrings(step);
  if (recorded.length === 0) return true;
  const target = call.input['old_string'];
  if (typeof target !== 'string' || target === '') return false;
  return textsOf(prefix).some((text) => text.includes(target) && recorded.some((old) => text.includes(old)));
}
