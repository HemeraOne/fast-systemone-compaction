import { describe, expect, it } from 'vitest';
import type { Message } from '../src/index.js';
import { recordedFiles } from '../tools/behaviour/derive.js';
import { buildLookup } from '../tools/behaviour/history.js';
import type { Answer } from '../tools/behaviour/history.js';

// Derived answers (spec 007): lookups that are not exact repeats but whose result recorded
// file content fully determines. Histories are built in memory; nothing is read or run.

let nextId = 0;

function pair(tool: string, input: Record<string, unknown>, output: string, isError = false): Message[] {
  const id = `tu${++nextId}`;
  return [
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool, input }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: output, isError }] },
  ];
}

/** A Read result as the tool prints it: every row numbered from 1. */
const numbered = (lines: readonly string[]): string => lines.map((line, i) => `${String(i + 1).padStart(6)}→${line}`).join('\n');

const CODE = ['export function total() {', '  return 1;', '}', '', 'export const tax = 2;', 'const total2 = total();'];

const readOf = (path: string, lines: readonly string[] = CODE, extra: Record<string, unknown> = {}): Message[] =>
  pair('Read', { file_path: path, ...extra }, numbered(lines));

const edit = (path: string, isError = false): Message[] =>
  pair('Edit', { file_path: path, old_string: 'o', new_string: 'n' }, 'ok', isError);

const answerOf = (prefix: Message[], tool: string, input: Record<string, unknown>): Answer => buildLookup(prefix).answer({ tool, input });
const grep = (prefix: Message[], input: Record<string, unknown>): Answer => answerOf(prefix, 'Grep', input);
const bash = (prefix: Message[], command: string): Answer => answerOf(prefix, 'Bash', { command });

const derived = (text: string): Answer => ({ kind: 'derived', text });
const miss = (label: string): Answer => ({ kind: 'miss', label });

describe('recorded content of a file', () => {
  const lines = (prefix: Message[], path = 'src/a.ts') => recordedFiles(prefix)(path);

  it('takes the rows of a complete Read without their numbers', () => {
    expect(lines(readOf('src/a.ts'))).toEqual({ lines: CODE });
    expect(lines(readOf('src\\a.ts').map((m) => m), 'src/a.ts')).toEqual({ lines: CODE });
  });

  it('refuses a Read with an offset or a limit', () => {
    expect(lines(readOf('src/a.ts', CODE, { offset: 2 }))).toEqual({ cause: 'partial' });
    expect(lines(readOf('src/a.ts', CODE, { limit: 3 }))).toEqual({ cause: 'partial' });
  });

  it('refuses a result whose rows are not numbered consecutively from 1', () => {
    expect(lines(pair('Read', { file_path: 'src/a.ts' }, '     1→a\n     3→b'))).toEqual({ cause: 'partial' });
    expect(lines(pair('Read', { file_path: 'src/a.ts' }, '     2→a\n     3→b'))).toEqual({ cause: 'partial' });
    expect(lines(pair('Read', { file_path: 'src/a.ts' }, 'a result with a trailing note'))).toEqual({ cause: 'partial' });
  });

  it('refuses a result at the Read tool\'s own limits and a file with carriage returns', () => {
    const rows = (n: number): string[] => Array.from({ length: n }, (_, i) => `l${i}`);
    expect(lines(readOf('src/a.ts', rows(1999)))).toMatchObject({ lines: expect.any(Array) });
    expect(lines(readOf('src/a.ts', rows(2000)))).toEqual({ cause: 'partial' });
    expect(lines(readOf('src/a.ts', ['x'.repeat(2000)]))).toEqual({ cause: 'partial' });
    expect(lines(readOf('src/a.ts', ['a\r', 'b']))).toEqual({ cause: 'partial' });
  });

  it('uses the latest complete Read, and an earlier one when a later Read is partial', () => {
    expect(lines([...readOf('src/a.ts', ['old']), ...readOf('src/a.ts', ['new'])])).toEqual({ lines: ['new'] });
    expect(lines([...readOf('src/a.ts', ['old']), ...readOf('src/a.ts', ['new'], { limit: 1 })])).toEqual({ lines: ['old'] });
  });

  it('is stale after a later successful write, not after a failed one, and fresh after a Read that follows the write', () => {
    expect(lines([...readOf('src/a.ts'), ...edit('src/a.ts')])).toEqual({ cause: 'stale' });
    expect(lines([...readOf('src/a.ts'), ...edit('src/a.ts', true)])).toEqual({ lines: CODE });
    expect(lines([...readOf('src/a.ts', ['old']), ...pair('Write', { file_path: 'src/a.ts', content: 'x' }, 'ok'), ...readOf('src/a.ts', ['new'])])).toEqual({
      lines: ['new'],
    });
  });

  it('says never read, or respelled when only another spelling of the path was read', () => {
    expect(lines(readOf('src/a.ts'), 'src/b.ts')).toEqual({ cause: 'never read' });
    expect(lines(readOf('src/a.ts'), './src/a.ts')).toEqual({ cause: 'respelled' });
    expect(lines(readOf('C:/repo/src/a.ts'), 'src/a.ts')).toEqual({ cause: 'respelled' });
  });
});

describe('derived Grep', () => {
  const prefix = readOf('src/a.ts');
  const content = (input: Record<string, unknown>) => grep(prefix, { path: 'src/a.ts', output_mode: 'content', ...input });

  it('returns the matching lines with their numbers, as the real tool does for one file', () => {
    expect(content({ pattern: 'total' })).toEqual(derived('1:export function total() {\n6:const total2 = total();'));
    expect(content({ pattern: 'total', '-n': true })).toEqual(content({ pattern: 'total' }));
  });

  it('folds ASCII case only with -i', () => {
    expect(content({ pattern: 'TAX' })).toEqual(derived('No matches found'));
    expect(content({ pattern: 'TAX', '-i': true })).toEqual(derived('5:export const tax = 2;'));
  });

  it('adds context lines with a dash and separates groups that are not adjacent', () => {
    expect(content({ pattern: 'tax', '-C': 1 })).toEqual(derived('4-\n5:export const tax = 2;\n6-const total2 = total();'));
    expect(content({ pattern: 'return', '-A': 1 })).toEqual(derived('2:  return 1;\n3-}'));
    expect(content({ pattern: 'function|total2', '-B': 1, '-A': 0 })).toEqual(derived('1:export function total() {\n--\n5-export const tax = 2;\n6:const total2 = total();'));
    expect(content({ pattern: 'tax', context: 1 })).toEqual(content({ pattern: 'tax', '-C': 1 }));
  });

  it('prints the tool\'s no-match text per output mode', () => {
    expect(content({ pattern: 'zzz' })).toEqual(derived('No matches found'));
    expect(grep(prefix, { path: 'src/a.ts', pattern: 'zzz' })).toEqual(derived('No files found'));
    expect(grep(prefix, { path: 'src/a.ts', pattern: 'zzz', output_mode: 'files_with_matches' })).toEqual(derived('No files found'));
  });

  it('applies head_limit, and does not answer when the limit would cut the output', () => {
    expect(content({ pattern: '.', head_limit: 0 })).toEqual(derived('1:export function total() {\n2:  return 1;\n3:}\n5:export const tax = 2;\n6:const total2 = total();'));
    expect(content({ pattern: '.', head_limit: 3 })).toEqual(miss('Grep'));
    const long = Array.from({ length: 300 }, (_, i) => `line ${i}`);
    expect(grep(readOf('src/long.ts', long), { path: 'src/long.ts', output_mode: 'content', pattern: 'line' })).toEqual(miss('Grep'));
    expect(grep(readOf('src/long.ts', long), { path: 'src/long.ts', output_mode: 'content', pattern: 'line 29', head_limit: 250 })).toMatchObject({ kind: 'derived' });
  });

  it('matches whole words and classes in the supported subset', () => {
    expect(content({ pattern: '\\btotal\\b' })).toEqual(derived('1:export function total() {\n6:const total2 = total();'));
    expect(content({ pattern: '^export (const|function)\\s+\\w+' })).toEqual(derived('1:export function total() {\n5:export const tax = 2;'));
    expect(content({ pattern: 'tot[a-z]{2}2' })).toEqual(derived('6:const total2 = total();'));
  });
});

describe('derived Bash', () => {
  const prefix = readOf('src/a.ts');
  const twelve = Array.from({ length: 12 }, (_, i) => `row ${i + 1}`);

  it('prints a file, its first lines or its last lines, as the Bash tool trims them', () => {
    expect(bash(prefix, 'cat src/a.ts')).toEqual(derived(CODE.join('\n')));
    expect(bash(prefix, 'head -n 2 src/a.ts')).toEqual(derived(CODE.slice(0, 2).join('\n')));
    expect(bash(prefix, 'head -2 src/a.ts')).toEqual(derived(CODE.slice(0, 2).join('\n')));
    expect(bash(prefix, 'tail -n 2 src/a.ts')).toEqual(derived(CODE.slice(-2).join('\n')));
    expect(bash(prefix, 'cat   "src/a.ts"')).toEqual(derived(CODE.join('\n')));
    expect(bash(readOf('src/t.ts', twelve), 'head src/t.ts')).toEqual(derived(twelve.slice(0, 10).join('\n')));
    expect(bash(readOf('src/t.ts', twelve), 'tail src/t.ts')).toEqual(derived(twelve.slice(2).join('\n')));
    expect(bash(readOf('src/e.ts', ['a', 'b', '', '']), 'cat src/e.ts')).toEqual(derived('a\nb'));
  });
});

describe('what stays unanswerable', () => {
  const prefix = readOf('src/a.ts');
  const partial = readOf('src/p.ts', CODE, { limit: 3 });
  const content = { output_mode: 'content' };

  it('labels a file that was never read, a directory, a partial read, a stale file and a respelled path', () => {
    expect(grep(prefix, { pattern: 'x', path: 'src/b.ts', ...content })).toEqual(miss('Grep (never read)'));
    expect(grep(prefix, { pattern: 'x', path: 'src', ...content })).toEqual(miss('Grep (never read)'));
    expect(grep(partial, { pattern: 'x', path: 'src/p.ts', ...content })).toEqual(miss('Grep (partial)'));
    expect(grep([...prefix, ...edit('src/a.ts')], { pattern: 'x', path: 'src/a.ts', ...content })).toEqual(miss('Grep (stale)'));
    expect(grep(prefix, { pattern: 'x', path: './src/a.ts', ...content })).toEqual(miss('Grep (respelled)'));
    expect(bash(prefix, 'cat src/b.ts')).toEqual(miss('Bash (never read)'));
    expect(bash(partial, 'head -n 2 src/p.ts')).toEqual(miss('Bash (partial)'));
  });

  it('keeps the bare tool name for a form the recorded text cannot answer', () => {
    expect(grep(prefix, { pattern: 'x' })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', glob: '*.ts', path: 'src/a.ts' })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', type: 'ts', path: 'src/a.ts' })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', '-o': true, path: 'src/a.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', multiline: true, path: 'src/a.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', '-n': false, path: 'src/a.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'x', output_mode: 'count', path: 'src/a.ts' })).toEqual(miss('Grep'));
    expect(grep(prefix, { pattern: 'tax', path: 'src/a.ts' })).toEqual(miss('Grep'));
  });

  it('does not answer a pattern outside the shared dialect, or one whose class or case rule depends on Unicode', () => {
    for (const pattern of ['(?=x)', '(?i)tax', 'a\\1', '\\p{L}', '[[:alpha:]]', 'a{', 'tax\\', '\\d+\\h', '[a', '']) {
      expect(grep(prefix, { pattern, path: 'src/a.ts', ...content })).toEqual(miss('Grep'));
    }
    const accented = readOf('src/u.ts', ['const café = 1;', 'const cafe = 2;']);
    expect(grep(accented, { pattern: '\\w+', path: 'src/u.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(accented, { pattern: 'CAFE', '-i': true, path: 'src/u.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(accented, { pattern: 'cafe', path: 'src/u.ts', ...content })).toEqual(derived('2:const cafe = 2;'));
  });

  it('does not answer a line the tool would shorten', () => {
    const wide = readOf('src/w.ts', ['x'.repeat(300), 'short']);
    expect(grep(wide, { pattern: 'x', path: 'src/w.ts', ...content })).toEqual(miss('Grep'));
    expect(grep(wide, { pattern: 'short', path: 'src/w.ts', ...content })).toEqual(derived('2:short'));
  });

  it('does not answer a command that does more than print one recorded file', () => {
    for (const command of [
      'ls',
      'ls src',
      'git status',
      'cd src && cat a.ts',
      'cat src/a.ts | head',
      'cat src/a.ts > out.txt',
      'cat src/a.ts src/b.ts',
      'cat -n src/a.ts',
      'cat $HOME/a.ts',
      'cat src/*.ts',
      'head -n 2 src\\a.ts',
      'grep tax src/a.ts',
      'sed -n 1,2p src/a.ts',
      'head -c 5 src/a.ts',
    ]) {
      expect(bash(prefix, command)).toEqual(miss('Bash'));
    }
  });

  it('does not answer a Glob that is not an exact repeat, and not a file that starts with a blank line', () => {
    expect(answerOf(prefix, 'Glob', { pattern: 'src/*.ts' })).toEqual(miss('Glob'));
    expect(bash(readOf('src/b.ts', ['', 'x']), 'cat src/b.ts')).toEqual(miss('Bash'));
    expect(bash(readOf('src/big.ts', ['x'.repeat(1900), ...Array.from({ length: 20 }, () => 'x'.repeat(1900))]), 'cat src/big.ts')).toEqual(miss('Bash'));
  });
});

describe('order and identity of answers', () => {
  it('answers an exact repeat as recorded, before any derivation', () => {
    const prefix = [...readOf('src/a.ts'), ...pair('Grep', { path: 'src/a.ts', pattern: 'tax', output_mode: 'content' }, 'RECORDED')];
    expect(grep(prefix, { output_mode: 'content', pattern: 'tax', path: 'src/a.ts' })).toEqual({ kind: 'repeat', text: 'RECORDED' });
    expect(buildLookup(prefix).serve({ tool: 'Grep', input: { output_mode: 'content', pattern: 'tax', path: 'src/a.ts' } })).toBe('RECORDED');
  });

  it('gives one lookup one answer, so every arm sees the same text', () => {
    const lookup = buildLookup(readOf('src/a.ts'));
    const call = { tool: 'Grep', input: { path: 'src/a.ts', output_mode: 'content', pattern: 'tax' } };
    expect(lookup.answer(call)).toEqual(lookup.answer(call));
    expect(lookup.serve(call)).toBe('5:export const tax = 2;');
    expect(lookup.miss({ tool: 'Bash', input: { command: 'ls' } })).toBe('Bash');
  });

  it('leaves a Read exactly as before: whole recorded text for a repeat, never derived', () => {
    const lookup = buildLookup(readOf('src/a.ts'));
    expect(lookup.answer({ tool: 'Read', input: { file_path: 'src/a.ts', offset: 3 } })).toEqual({ kind: 'repeat', text: numbered(CODE) });
    expect(lookup.answer({ tool: 'Read', input: { file_path: './src/a.ts' } })).toEqual(miss('Read (respelled)'));
  });
});
