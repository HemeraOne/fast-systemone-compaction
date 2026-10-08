import type { Message, ToolUse } from '../../src/index.js';

/**
 * Answers a lookup from the recorded content of a file instead of from a recorded call
 * (spec 007). Pure: nothing here reads a real file or runs a command. Anything the recorded
 * text cannot fully determine, or a form this module cannot reproduce exactly, is not answered.
 */

export const slashed = (value: unknown): string | undefined =>
  typeof value === 'string' ? value.replace(/\\/g, '/') : undefined;

/** Whether two paths differ only in case, a drive prefix or a relative-versus-absolute spelling. */
export function respelled(a: string, b: string): boolean {
  const [x, y] = [a.toLowerCase().replace(/^\.\//, ''), b.toLowerCase().replace(/^\.\//, '')];
  return x === y || x.endsWith(`/${y}`) || y.endsWith(`/${x}`);
}

const WRITING_TOOLS = new Set(['Edit', 'MultiEdit', 'Write']);

/** The Read tool's own cut-offs: it shows at most this many rows, each at most this long. */
const READ_ROWS = 2000;
const READ_ROW_CHARS = 2000;

/** Why a file has no usable recorded content; the fixed causes of the report. */
export type FileCause = 'never read' | 'respelled' | 'partial' | 'stale';

export type FileStatus = { lines: string[] } | { cause: FileCause };

/** The rows of a complete Read result without their numbers, or `undefined` when it is not one. */
function completeRows(text: string): string[] | undefined {
  if (text.includes('\r')) return undefined;
  const rows = text.split('\n');
  if (rows[rows.length - 1] === '') rows.pop();
  if (rows.length === 0 || rows.length >= READ_ROWS) return undefined;
  const lines: string[] = [];
  for (const [i, row] of rows.entries()) {
    const match = /^ *(\d+)[\t→:-](.*)$/.exec(row);
    if (match === null || Number(match[1]) !== i + 1 || match[2]!.length >= READ_ROW_CHARS) return undefined;
    lines.push(match[2]!);
  }
  return lines;
}

/**
 * The recorded content of files in a history: a file counts only when a successful Read
 * without `offset` or `limit` returned all of it, and nothing later in the history wrote it.
 */
export function recordedFiles(prefix: readonly Message[]): (path: string) => FileStatus {
  const uses = new Map<string, ToolUse>();
  const reads = new Map<string, { index: number; lines: string[] | undefined }[]>();
  const lastWrite = new Map<string, number>();

  prefix.forEach((message, index) => {
    for (const use of message.toolUses) uses.set(use.tool_use_id, use);
    for (const result of message.toolResults ?? []) {
      const use = uses.get(result.tool_use_id);
      const path = use === undefined ? undefined : slashed(use.input['file_path']);
      if (use === undefined || path === undefined || result.isError === true) continue;
      if (WRITING_TOOLS.has(use.tool)) lastWrite.set(path, index);
      if (use.tool !== 'Read') continue;
      const whole = use.input['offset'] === undefined && use.input['limit'] === undefined;
      const list = reads.get(path) ?? [];
      list.push({ index, lines: whole ? completeRows(result.text) : undefined });
      reads.set(path, list);
    }
  });

  return (path) => {
    const made = reads.get(path);
    if (made === undefined) {
      return { cause: [...reads.keys()].some((other) => respelled(path, other)) ? 'respelled' : 'never read' };
    }
    const latest = made.filter((read) => read.lines !== undefined).pop();
    if (latest === undefined) return { cause: 'partial' };
    const written = lastWrite.get(path);
    if (written !== undefined && written > latest.index) return { cause: 'stale' };
    return { lines: latest.lines! };
  };
}

/** A form this module cannot reproduce exactly. */
export const UNSUPPORTED = Symbol('unsupported');
export type Derived = string | typeof UNSUPPORTED;

const NON_ASCII = /[^\x00-\x7f]/;

/**
 * A JS `RegExp` for a pattern whose meaning is the same in ripgrep's dialect: literals, `.`,
 * `|`, groups, classes, `^`, `$`, the usual quantifiers and the escapes `\w \s \d \b`.
 * Anything else, and the Unicode-sensitive forms over non-ASCII text, is not reproducible.
 */
function matcherFor(pattern: string, ignoreCase: boolean, lines: readonly string[]): RegExp | undefined {
  if (pattern === '' || NON_ASCII.test(pattern)) return undefined;
  const asciiOnly = !lines.some((line) => NON_ASCII.test(line));
  let usesClass = false;
  let inClass = false;
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]!;
    if (ch === '\\') {
      const next = pattern[++i];
      if (next === undefined) return undefined;
      if ('wsdb'.includes(next)) usesClass = true;
      else if (!'\\.^$*+?()[]{}|/-'.includes(next)) return undefined;
    } else if (inClass) {
      if (ch === '[') return undefined;
      if (ch === ']') inClass = false;
    } else if (ch === '[') {
      if (pattern[i + 1] === ':') return undefined;
      inClass = true;
    } else if (ch === '(' && pattern[i + 1] === '?') {
      return undefined;
    } else if (ch === '{') {
      const quantifier = /^\{\d+(,\d*)?\}/.exec(pattern.slice(i));
      if (quantifier === null) return undefined;
      i += quantifier[0].length - 1;
    }
  }
  if (inClass || ((usesClass || ignoreCase) && !asciiOnly)) return undefined;
  try {
    return new RegExp(pattern, ignoreCase ? 'iu' : 'u');
  } catch {
    return undefined;
  }
}

/** The tool's text for a search that found nothing, per output mode (the same in every recorded result). */
const NO_MATCHES = 'No matches found';
const NO_FILES = 'No files found';

/**
 * The tool replaces a very long matching line with a note. Its limit is not recorded; no kept
 * line in the recorded results was longer than this, so a longer one is not reproduced.
 */
const LONG_LINE = 250;

const GREP_KEYS = new Set(['pattern', 'path', 'output_mode', '-n', '-i', '-A', '-B', '-C', 'context', 'head_limit']);
const DEFAULT_HEAD_LIMIT = 250;

const count = (value: unknown): number | undefined | typeof UNSUPPORTED =>
  value === undefined ? undefined : typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : UNSUPPORTED;

/** What a Grep over one file with these recorded lines returns, in the Grep tool's own format. */
export function deriveGrep(lines: readonly string[], input: Record<string, unknown>, path: string): Derived {
  if (Object.keys(input).some((key) => !GREP_KEYS.has(key))) return UNSUPPORTED;
  const { pattern } = input;
  if (typeof pattern !== 'string') return UNSUPPORTED;
  const mode = input['output_mode'] ?? 'files_with_matches';
  if (mode !== 'content' && mode !== 'files_with_matches') return UNSUPPORTED;
  if (input['-n'] !== undefined && input['-n'] !== true) return UNSUPPORTED;
  if (input['-i'] !== undefined && typeof input['-i'] !== 'boolean') return UNSUPPORTED;
  const [after, before, around, limit] = [count(input['-A']), count(input['-B']), count(input['-C'] ?? input['context']), count(input['head_limit'])];
  if ([after, before, around, limit].includes(UNSUPPORTED)) return UNSUPPORTED;
  const matcher = matcherFor(pattern, input['-i'] === true, lines);
  if (matcher === undefined) return UNSUPPORTED;

  const hits = lines.flatMap((line, i) => (matcher.test(line) ? [i] : []));
  const cap = (limit as number | undefined) ?? DEFAULT_HEAD_LIMIT;
  if (hits.length === 0) return mode === 'content' ? NO_MATCHES : NO_FILES;
  // The tool prints a matching file's path relative to its own working directory, which the history does not show.
  if (mode === 'files_with_matches') return UNSUPPORTED;

  const [a, b] = [(after as number | undefined) ?? (around as number | undefined) ?? 0, (before as number | undefined) ?? (around as number | undefined) ?? 0];
  const shown = new Set<number>();
  for (const hit of hits) for (let i = Math.max(0, hit - b); i <= Math.min(lines.length - 1, hit + a); i++) shown.add(i);
  const matched = new Set(hits);
  const out: string[] = [];
  let previous = -2;
  for (const i of [...shown].sort((x, y) => x - y)) {
    if (previous >= 0 && i !== previous + 1 && (a > 0 || b > 0)) out.push('--');
    out.push(`${i + 1}${matched.has(i) ? ':' : '-'}${lines[i]}`);
    previous = i;
  }
  if ((cap !== 0 && out.length > cap) || out.some((line) => line.length > LONG_LINE)) return UNSUPPORTED;
  return out.join('\n');
}

/** The Bash tool cuts what a command prints at this many characters. */
const BASH_OUTPUT_CHARS = 30_000;

const SHOW = /^(cat|head|tail)(?:\s+(?:-n\s*(\d+)|-(\d+)))?\s+(?:"([^"]+)"|'([^']+)'|([^\s"']+))$/;

/** The file a command only prints (all or part of), with how to print it; `undefined` for any other command. */
export function fileShowing(command: string): { path: string; show: (lines: readonly string[]) => Derived } | undefined {
  const text = command.replace(/\s+/g, ' ').trim();
  if (/[|&;<>`$*?\\\n]/.test(text)) return undefined;
  const match = SHOW.exec(text);
  if (match === null) return undefined;
  const [, program, nOption, nShort, doubleQuoted, singleQuoted, bare] = match;
  const path = doubleQuoted ?? singleQuoted ?? bare;
  const given = nOption ?? nShort;
  if (path === undefined || path.startsWith('-') || (program === 'cat' && given !== undefined)) return undefined;
  const n = given === undefined ? 10 : Number(given);
  return {
    path,
    show: (lines) => {
      const shown = program === 'cat' ? lines : program === 'head' ? lines.slice(0, n) : n === 0 ? [] : lines.slice(-n);
      // The Bash tool trims the end of what a command prints; a file that starts with a blank line is not shown the same way.
      const out = shown.join('\n').trimEnd();
      return lines[0] === '' || out.length > BASH_OUTPUT_CHARS ? UNSUPPORTED : out;
    },
  };
}
