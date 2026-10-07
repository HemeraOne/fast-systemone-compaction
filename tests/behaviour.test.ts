import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Message, ToolUse } from '../src/index.js';
import {
  buildLookup,
  compactedHistory,
  controlHistory,
  flattenHistory,
  matchesRecorded,
  pointContext,
  renderTranscript,
  SUMMARY_REQUEST,
  summaryHistory,
  summaryPrompt,
} from '../tools/behaviour/history.js';
import { buildArgs, parseStream, quoteArg, spawnChild, stubFailed } from '../tools/behaviour/model.js';
import type { ChildRequest, ChildResult, ChildRunner, Scratch, StreamLine } from '../tools/behaviour/model.js';
import { formatReport, summarize } from '../tools/behaviour/report.js';
import type { PointRecord, RunInfo } from '../tools/behaviour/report.js';
import { run } from '../tools/behaviour/run.js';
import { choose, keptPoints, lostPoints, sample } from '../tools/behaviour/select.js';
import type { LostPoint } from '../tools/behaviour/select.js';
import { runPoint, runText } from '../tools/behaviour/session.js';
import type { Outcome, SessionDeps } from '../tools/behaviour/session.js';
import { createStub, MAX_LOOKUPS, NOT_AVAILABLE, STOP, STUB_TOOLS } from '../tools/behaviour/stub.js';

// --- synthetic sessions (same shapes as tests/replay.test.ts) --------------------

let nextId = 0;

function pair(tool: string, input: Record<string, unknown>, output: string, isError = false): Message[] {
  const id = `tu${++nextId}`;
  return [
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool, input }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: output, isError }] },
  ];
}

function chat(role: 'user' | 'assistant', text: string): Message {
  return { role, text, toolUses: [] };
}

const around = (before: number, value: string, after: number): string =>
  `${'a'.repeat(before)}${value}${'b'.repeat(after)}`;

const VALUE = 'return computeInvoiceTotal(lineItems, taxRate);';

function session(middle: Message[], final: { tool: string; input: Record<string, unknown> }): Message[] {
  const filler = Math.max(8, 20 - 1 - middle.length);
  return [
    chat('user', 'please fix the invoice total'),
    ...middle,
    ...Array.from({ length: filler }, (_, i) => chat(i % 2 === 0 ? 'assistant' : 'user', `chat ${i}`)),
    ...pair(final.tool, final.input, 'done'),
  ];
}

const editing = (oldString: string, file = 'src/billing.ts') => ({
  tool: 'Edit',
  input: { file_path: file, old_string: oldString, new_string: 'x' },
});

/** The edit target sits only in a big Read result that rule 1 cuts; a Read can bring it back. */
const reachableSession = (): Message[] =>
  session(pair('Read', { file_path: 'src/billing.ts' }, around(5000, VALUE, 5000)), editing(VALUE));

/** The edit target sits only in a Read of a file written afterwards, which no stub can re-serve. */
const unreachableSession = (): Message[] =>
  session(
    [
      ...pair('Read', { file_path: 'src/legacy.ts' }, around(2500, VALUE, 2500)),
      ...pair('Write', { file_path: 'src/legacy.ts', content: 'rewritten' }, 'ok'),
    ],
    editing(VALUE, 'src/elsewhere.ts'),
  );

/**
 * The edit target sits in a small Read result that compaction keeps verbatim, so nothing is
 * lost; a big unrelated result gives the session the reduction the replay requires.
 */
const keptSession = (): Message[] =>
  session(
    [
      ...pair('Read', { file_path: 'src/other.ts' }, 'c'.repeat(20_000)),
      ...pair('Read', { file_path: 'src/billing.ts' }, around(100, VALUE, 100)),
    ],
    editing(VALUE),
  );

function toJsonl(messages: readonly Message[]): string {
  return messages
    .map((message, i) => {
      if (message.role === 'assistant') {
        const content: unknown[] = [];
        if (message.text !== '') content.push({ type: 'text', text: message.text });
        for (const use of message.toolUses) {
          content.push({ type: 'tool_use', id: use.tool_use_id, name: use.tool, input: use.input });
        }
        return JSON.stringify({ type: 'assistant', message: { id: `m${i}`, role: 'assistant', content } });
      }
      const content =
        message.toolResults === undefined
          ? message.text
          : message.toolResults.map((r) => ({
              type: 'tool_result',
              tool_use_id: r.tool_use_id,
              is_error: r.isError === true,
              content: r.text,
            }));
      return JSON.stringify({ type: 'user', message: { role: 'user', content } });
    })
    .join('\n');
}

// --- temp corpus, scripted fake child runner, fake scratch ------------------------

const dirs: string[] = [];

function corpus(sessions: Record<string, Message[]>): string {
  const root = mkdtempSync(join(tmpdir(), 'behaviour-test-'));
  dirs.push(root);
  for (const [name, messages] of Object.entries(sessions)) {
    const [project, file] = name.split('/') as [string, string];
    mkdirSync(join(root, project), { recursive: true });
    writeFileSync(join(root, project, file), toJsonl(messages));
  }
  return root;
}

afterEach(() => {
  // A killed child can hold its directory for a moment on Windows, so retry.
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
});

function pointOf(messages: Message[]): LostPoint {
  const { points } = lostPoints(corpus({ 'p/s.jsonl': messages }));
  if (points.length !== 1) throw new Error(`expected one lost point, got ${points.length}`);
  return points[0]!;
}

const textOf = (value: unknown): string => JSON.stringify(value);

const event = (value: unknown, at: number): StreamLine => ({ text: JSON.stringify(value), at });

interface StreamOptions {
  calls?: { tool: string; input: Record<string, unknown> }[];
  /** Tokens as `usage`; `null` leaves usage out. */
  usage?: { input_tokens: number; output_tokens: number } | null;
  init?: unknown;
  result?: boolean;
  isError?: boolean;
  text?: string;
  subtype?: string;
}

/** A child's event stream: init, one assistant event per call (one second apart), the result. */
function stream(options: StreamOptions = {}): StreamLine[] {
  const lines: StreamLine[] = [];
  lines.push(event({ type: 'system', subtype: 'init', ...(options.init === undefined ? { mcp_servers: [{ name: 'stub', status: 'connected' }], tools: ['mcp__stub__Read', 'mcp__stub__Edit'] } : options.init as object) }, 1500));
  (options.calls ?? []).forEach((call, i) => {
    lines.push(
      event(
        { type: 'assistant', message: { content: [{ type: 'tool_use', id: `c${++nextId}`, name: `mcp__stub__${call.tool}`, input: call.input }] } },
        2000 + i * 1000,
      ),
    );
  });
  if (options.result !== false) {
    lines.push(
      event(
        {
          type: 'result',
          is_error: options.isError === true,
          ...(options.subtype === undefined ? {} : { subtype: options.subtype }),
          result: options.text ?? 'done',
          ...(options.usage === null ? {} : { usage: options.usage ?? { input_tokens: 100, output_tokens: 10 } }),
        },
        9000,
      ),
    );
  }
  return lines;
}

const ok = (lines: StreamLine[], code: number | null = 0): ChildResult => ({ ok: true, lines, code });

/** A runner that answers from a function of the request and remembers every request. */
function fakeRunner(answer: (request: ChildRequest, count: number) => ChildResult) {
  const requests: ChildRequest[] = [];
  const impl = vi.fn(async (request: ChildRequest) => {
    requests.push(request);
    return answer(request, requests.length);
  });
  return { runner: impl as ChildRunner, requests, impl };
}

/** A runner that plays the given results in order. */
function scripted(results: ChildResult[]) {
  return fakeRunner((_request, count) => results[count - 1] ?? { ok: false, reason: 'script ended' });
}

function fakeScratch(): Scratch & { cleanup: ReturnType<typeof vi.fn> } {
  return {
    cwd: '/scratch/work',
    mcpConfigFor: () => '/scratch/mcp.json',
    file: vi.fn((name: string) => `/scratch/${name}`),
    cleanup: vi.fn(),
  };
}

function clock(): () => number {
  let t = 0;
  return () => (t += 1000);
}

function deps(runner: ChildRunner): SessionDeps {
  return { runner, scratch: fakeScratch(), model: 'test-model', now: clock() };
}

const call = (tool: string, input: Record<string, unknown>) => ({ tool, input });
const edit = (old: string, file = 'src/billing.ts') => call('Edit', { file_path: file, old_string: old, new_string: 'n' });

// --- selection and histories ----------------------------------------------------

describe('selection', () => {
  it('orders lost points by session name then index, whatever the folder order', () => {
    const root = corpus({ 'p1/b.jsonl': reachableSession(), 'p2/a.jsonl': reachableSession() });
    const { points, sessions } = lostPoints(root);
    expect(sessions).toBe(2);
    expect(points.map((p) => p.session)).toEqual(['a.jsonl', 'b.jsonl']);
    expect(points[0]!.path).toBe(join(root, 'p2', 'a.jsonl'));
    expect(lostPoints(root).points.map((p) => p.messageIndex)).toEqual(points.map((p) => p.messageIndex));
  });

  it('ignores points without a loss and sessions that are not lost points', () => {
    const quiet = session([], { tool: 'Read', input: { file_path: 'src/billing.ts' } });
    expect(lostPoints(corpus({ 'p/q.jsonl': quiet })).points).toEqual([]);
  });

  it('selects kept points apart from lost ones, and only edit and command steps that needed an earlier value', () => {
    const quiet = session([], { tool: 'Read', input: { file_path: 'src/billing.ts' } });
    const root = corpus({ 'p/k.jsonl': keptSession(), 'p/l.jsonl': reachableSession(), 'p/q.jsonl': quiet });
    expect(keptPoints(root).points.map((p) => [p.session, p.losses.length])).toEqual([['k.jsonl', 0]]);
    // A Read step is left out: a re-read there is the final action, so doubt could not show.
    const reading = session(pair('Read', { file_path: 'src/billing.ts' }, around(100, VALUE, 100)), { tool: 'Read', input: { file_path: 'src/billing.ts' } });
    expect(keptPoints(corpus({ 'p/r.jsonl': [...pair('Read', { file_path: 'src/other.ts' }, 'c'.repeat(20_000)), ...reading] })).points).toEqual([]);
    expect(lostPoints(root).points.map((p) => p.session)).toEqual(['l.jsonl']);
  });

  it('continues in order after the skipped points when asked to skip, whatever the count', () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    expect(choose(items, 3, 4)).toEqual([4, 5, 6]);
    expect(choose(items, 3, 9)).toEqual([9]);
    expect(choose(items, 3, 10)).toEqual([]);
    expect(choose(items, 3, undefined)).toEqual(sample(items, 3));
  });

  it('takes evenly spaced entries, all when there are fewer, the same every time', () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    expect(sample(items, 3)).toEqual([0, 3, 6]);
    expect(sample(items, 3)).toEqual(sample(items, 3));
    expect(sample(items, 20)).toEqual(items);
  });
});

/** The recorded step is one of two parallel Edits in one message. */
const batchedSession = (): Message[] => {
  const base = reachableSession().slice(0, -2);
  return [
    ...base,
    {
      role: 'assistant',
      text: '',
      toolUses: [
        { tool_use_id: 'b1', tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'x' } },
        { tool_use_id: 'b2', tool: 'Edit', input: { file_path: 'src/other-file.ts', old_string: 'unrelated text', new_string: 'y' } },
      ],
    },
    {
      role: 'user',
      text: '',
      toolUses: [],
      toolResults: [
        { tool_use_id: 'b1', text: 'done', isError: false },
        { tool_use_id: 'b2', text: 'done', isError: false },
      ],
    },
  ];
};

describe('parallel calls', () => {
  it('skips a lost point whose step is one of several calls of its kind, and counts it', () => {
    const root = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': batchedSession() });
    const found = lostPoints(root);
    expect(found.points.map((p) => p.session)).toEqual(['a.jsonl']);
    expect(found.skipped).toBe(1);
    expect(lostPoints(corpus({ 'p/a.jsonl': reachableSession() })).skipped).toBe(0);
  });

  it('says so in the report, and in the refusal when nothing else is left', async () => {
    const mixed = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': batchedSession() });
    const result = await runWith([...ARGS, '--root', mixed], needsTheValue().runner);
    expect(result.output).toContain('1 lost points available (1 more skipped: one of several parallel calls of its kind)');
    const onlyBatched = await runWith([...ARGS, '--root', corpus({ 'p/b.jsonl': batchedSession() })], scripted([]).runner);
    expect(onlyBatched.code).toBe(1);
    expect(onlyBatched.output).toContain('1 skipped as parallel calls');
  });
});

describe('histories', () => {
  it('the compacted history lacks the lost value that the control history still has', () => {
    const point = pointOf(reachableSession());
    expect(textOf(controlHistory(point.messages, point.messageIndex))).toContain(VALUE);
    expect(textOf(compactedHistory(point.messages, point.messageIndex))).not.toContain(VALUE);
    expect(flattenHistory(controlHistory(point.messages, point.messageIndex))).toContain(VALUE);
    expect(flattenHistory(compactedHistory(point.messages, point.messageIndex))).not.toContain(VALUE);
  });

  it('renders a labelled transcript with roles, tool calls, named results and error marks', () => {
    const messages: Message[] = [
      chat('user', 'go'),
      ...pair('Read', { file_path: 'a.ts' }, 'boom', true),
      chat('assistant', 'done'),
      chat('assistant', ''),
    ];
    const text = renderTranscript(messages);
    expect(text).toBe(
      [
        '=== user ===\ngo',
        '--- assistant tool call: Read ---\n{"file_path":"a.ts"}',
        '--- tool result: Read (error) ---\nboom',
        '=== assistant ===\ndone',
      ].join('\n\n'),
    );
    const prompt = flattenHistory(messages);
    expect(prompt.startsWith('Below is the conversation so far')).toBe(true);
    expect(prompt).toContain(text);
    expect(prompt.endsWith('Do not describe what you would do.')).toBe(true);
  });

  it('builds the summary request and the summary arm prompt', () => {
    const messages = [chat('user', 'go'), chat('assistant', 'ok')];
    expect(summaryPrompt(messages)).toBe(`${renderTranscript(messages)}\n\n${SUMMARY_REQUEST}`);
    const arm = summaryHistory('the gist');
    expect(arm).toContain('=== summary of the conversation so far ===\nthe gist');
    expect(arm).toContain('Continue as the assistant');
  });
});

describe('lookups', () => {
  const read = [...pair('Read', { file_path: 'src/a.ts' }, 'BODY-A')];
  const rest = [
    ...pair('Bash', { command: 'npm   test' }, 'TEST OUT'),
    ...pair('Grep', { pattern: 'x', path: 'src' }, 'GREP OUT'),
  ];

  it('answers a Read by path, ignoring slashes and offset, and exact repeats of Bash and Grep', () => {
    const lookup = buildLookup([chat('user', 'go'), ...read, ...rest]);
    expect(lookup.serve({ tool: 'Read', input: { file_path: 'src\\a.ts', offset: 5 } })).toBe('BODY-A');
    expect(lookup.serve({ tool: 'Bash', input: { command: 'npm test' } })).toBe('TEST OUT');
    expect(lookup.serve({ tool: 'Grep', input: { path: 'src', pattern: 'x' } })).toBe('GREP OUT');
  });

  it('refuses anything that is not an exact repeat, other tools, and unknown files', () => {
    const lookup = buildLookup([chat('user', 'go'), ...read, ...rest]);
    expect(lookup.serve({ tool: 'Bash', input: { command: 'npm run test' } })).toBeUndefined();
    expect(lookup.serve({ tool: 'Grep', input: { pattern: 'y', path: 'src' } })).toBeUndefined();
    expect(lookup.serve({ tool: 'Glob', input: { pattern: '*.ts' } })).toBeUndefined();
    expect(lookup.serve({ tool: 'Read', input: { file_path: 'src/b.ts' } })).toBeUndefined();
    expect(lookup.serve({ tool: 'Edit', input: { file_path: 'src/a.ts', old_string: 'o' } })).toBeUndefined();
  });

  it('refuses a Read after a later successful Edit of that file, not after a failed one', () => {
    const changed = (isError: boolean): Message[] =>
      pair('Edit', { file_path: 'src/a.ts', old_string: 'o', new_string: 'n' }, 'r', isError);
    expect(buildLookup([...read, ...changed(false)]).serve({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBeUndefined();
    expect(buildLookup([...read, ...changed(true)]).serve({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBe('BODY-A');
  });

  it('labels an unanswerable Read as never read, respelled or stale, and other tools by name', () => {
    const lookup = buildLookup([chat('user', 'go'), ...read, ...rest]);
    expect(lookup.miss({ tool: 'Read', input: { file_path: 'src/b.ts' } })).toBe('Read (never read)');
    expect(lookup.miss({ tool: 'Read', input: { file_path: 'C:/repo/SRC/a.ts' } })).toBe('Read (respelled)');
    expect(lookup.miss({ tool: 'Read', input: { file_path: './src/a.ts' } })).toBe('Read (respelled)');
    expect(lookup.miss({ tool: 'Bash', input: { command: 'ls' } })).toBe('Bash');
    const written = pair('Edit', { file_path: 'src/a.ts', old_string: 'o', new_string: 'n' }, 'r');
    expect(buildLookup([...read, ...written]).miss({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBe('Read (stale)');
  });

  it('reports a point as reachable only when a lost value sits in a result the stubs can serve', () => {
    expect(pointContext(pointOf(reachableSession())).reachable).toBe(true);
    expect(pointContext(pointOf(unreachableSession())).reachable).toBe(false);
  });
});

describe('matchesRecorded', () => {
  const step = (tool: string, input: Record<string, unknown>): ToolUse => ({ tool_use_id: 'rec', tool, input });
  const prefix = [chat('user', `see ${around(10, VALUE, 10)}`)];

  it('matches an edit whose target spans lines when the earlier Read result is numbered', () => {
    const target = 'first line of the target\n    second line of the target';
    const numbered = ['src/billing.ts', '10\tfirst line of the target', '11\t    second line of the target'].join('\n');
    const read = pair('Read', { file_path: 'src/billing.ts' }, numbered);
    const recorded = step('Edit', { file_path: 'src/billing.ts', old_string: target, new_string: 'a' });
    const proposed = { tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: target, new_string: 'b' } };
    expect(matchesRecorded(proposed, recorded, read)).toBe(true);
    expect(matchesRecorded({ ...proposed, input: { ...proposed.input, old_string: 'not in the file\nat all' } }, recorded, read)).toBe(false);
  });

  it('matches a path after slash normalisation and the same tool family', () => {
    const recorded = step('Read', { file_path: 'src/billing.ts' });
    expect(matchesRecorded({ tool: 'Read', input: { file_path: 'src\\billing.ts' } }, recorded, [])).toBe(true);
    expect(matchesRecorded({ tool: 'Read', input: { file_path: 'src/other.ts' } }, recorded, [])).toBe(false);
    expect(matchesRecorded({ tool: 'Edit', input: { file_path: 'src/billing.ts' } }, recorded, [])).toBe(false);
  });

  it('matches a command after whitespace collapse', () => {
    const recorded = step('Bash', { command: 'npm  run   typecheck' });
    expect(matchesRecorded({ tool: 'Bash', input: { command: 'npm run typecheck' } }, recorded, [])).toBe(true);
    expect(matchesRecorded({ tool: 'Bash', input: { command: 'npm run test' } }, recorded, [])).toBe(false);
  });

  it('matches an edit target that occurs in a recorded text holding the recorded target', () => {
    const recorded = step('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'y' });
    const proposed = (old: unknown) => ({ tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: old } });
    expect(matchesRecorded(proposed(VALUE), recorded, prefix)).toBe(true);
    expect(matchesRecorded(proposed('computeInvoiceTotal(lineItems'), recorded, prefix)).toBe(true);
    expect(matchesRecorded(proposed('somewhere else entirely'), recorded, prefix)).toBe(false);
    expect(matchesRecorded(proposed(''), recorded, prefix)).toBe(false);
    expect(matchesRecorded({ tool: 'Edit', input: { file_path: 'src/billing.ts' } }, recorded, prefix)).toBe(false);
    expect(matchesRecorded({ tool: 'Edit', input: { file_path: 'src/x.ts', old_string: VALUE } }, recorded, prefix)).toBe(false);
  });
});

// --- the child: arguments, stream, quoting ----------------------------------------

describe('buildArgs', () => {
  it('isolates the child, keeps the subscription login usable, and names no key', () => {
    const args = buildArgs({ model: 'sonnet', mcpConfig: '/scratch/mcp.json' });
    for (const flag of ['-p', '--disable-slash-commands', '--no-session-persistence', '--strict-mcp-config', '--verbose']) {
      expect(args).toContain(flag);
    }
    expect(args).not.toContain('--bare');
    expect(args).not.toContain('--safe-mode');
    expect(args.slice(args.indexOf('--setting-sources'), args.indexOf('--setting-sources') + 2)).toEqual(['--setting-sources', 'project']);
    expect(args.slice(args.indexOf('--model'), args.indexOf('--model') + 2)).toEqual(['--model', 'sonnet']);
    expect(args.slice(args.indexOf('--tools'), args.indexOf('--tools') + 2)).toEqual(['--tools', '']);
    expect(args[args.indexOf('--mcp-config') + 1]).toBe('/scratch/mcp.json');
    expect(args[args.indexOf('--allowedTools') + 1]).toBe(
      'mcp__stub__Read,mcp__stub__Grep,mcp__stub__Glob,mcp__stub__Edit,mcp__stub__Bash',
    );
    expect(textOf(args).toLowerCase()).not.toContain('api');
  });

  it('leaves the stub out for the summary call', () => {
    const args = buildArgs({ model: 'sonnet' });
    expect(args).not.toContain('--mcp-config');
    expect(args).not.toContain('--allowedTools');
    expect(args).toContain('--strict-mcp-config');
  });

  it('quotes only what the Windows shell would split', () => {
    expect(quoteArg('plain')).toBe('plain');
    expect(quoteArg('')).toBe('""');
    expect(quoteArg('two words')).toBe('"two words"');
    expect(quoteArg('say "hi"')).toBe('"say \\"hi\\""');
  });
});

describe('parseStream', () => {
  it('reads calls with their times, strips the stub prefix, and sums every reported token', () => {
    const lines = stream({ calls: [call('Read', { file_path: 'a.ts' }), edit(VALUE)] });
    lines[lines.length - 1] = event(
      {
        type: 'result',
        is_error: false,
        result: 'done',
        usage: { input_tokens: 100, output_tokens: 10, cache_creation_input_tokens: 5, cache_read_input_tokens: 7 },
      },
      9000,
    );
    const parsed = parseStream(lines);
    expect(parsed.calls.map((c) => [c.tool, c.at])).toEqual([['Read', 2000], ['Edit', 3000]]);
    expect(parsed).toMatchObject({ stubStatus: 'connected', tokens: 122, sawResult: true, isError: false, resultText: 'done' });
  });

  it('skips lines that are not JSON, counts a repeated call once, and reads a missing usage as unknown', () => {
    const one = stream({ calls: [edit(VALUE)], usage: null });
    const parsed = parseStream([{ text: 'not json', at: 1 }, ...one, one[1]!]);
    expect(parsed.calls).toHaveLength(1);
    expect(parsed.tokens).toBeUndefined();
  });

  it('reports stub tools that were not offered, and a compaction inside the child', () => {
    expect(parseStream(stream({ init: { mcp_servers: [{ name: 'stub', status: 'connected' }], tools: ['Bash'] } })).toolsOffered).toBe(false);
    expect(parseStream(stream({ init: { tools: ['mcp__stub__Read'] } })).toolsOffered).toBe(true);
    expect(parseStream(stream({ init: {} })).toolsOffered).toBeUndefined();
    const lines = [...stream({ calls: [edit(VALUE)] }), event({ type: 'system', subtype: 'compact_boundary' }, 5000)];
    expect(parseStream(lines).compacted).toBe(true);
    expect(parseStream(stream({ calls: [edit(VALUE)] })).compacted).toBe(false);
  });

  it('reads the stub status from the init event: a fixed vocabulary, not listed, or unknown', () => {
    const statusOf = (servers: unknown) => parseStream(stream({ init: { mcp_servers: servers } })).stubStatus;
    expect(statusOf([{ name: 'stub', status: 'connected' }])).toBe('connected');
    expect(statusOf([{ name: 'stub', status: 'pending' }])).toBe('pending');
    expect(statusOf([{ name: 'stub', status: 'failed' }])).toBe('failed');
    expect(statusOf([{ name: 'stub', status: 'needs-auth' }])).toBe('needs-auth');
    expect(statusOf([{ name: 'stub', status: 'some text we do not know' }])).toBe('unknown');
    expect(statusOf([{ name: 'other', status: 'connected' }])).toBe('not listed');
    expect(statusOf([])).toBe('not listed');
    expect(parseStream(stream({ init: {} })).stubStatus).toBeUndefined();
  });

  it('treats only a definitive status as a failure: pending may still connect', () => {
    expect(stubFailed('failed') && stubFailed('not listed') && stubFailed('needs-auth') && stubFailed('unknown')).toBe(true);
    expect(stubFailed('connected') || stubFailed('pending') || stubFailed(undefined)).toBe(false);
  });
});

// --- the stub -------------------------------------------------------------------------

describe('spawnChild', () => {
  it('kills a child as soon as its init event says the stub failed, keeping what it printed', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'behaviour-spawn-'));
    dirs.push(dir);
    const script = join(dir, 'fake-claude.mjs');
    const init = JSON.stringify({ type: 'system', subtype: 'init', mcp_servers: [{ name: 'stub', status: 'failed' }] });
    writeFileSync(script, `console.log(${JSON.stringify(init)});\nsetInterval(() => {}, 1000);\n`);
    const started = Date.now();
    const result = await spawnChild(Date.now, process.execPath)({ args: [script, '--mcp-config', 'x'], prompt: 'x', cwd: dir, timeoutMs: 60_000 });
    expect(Date.now() - started).toBeLessThan(30_000);
    expect(result).toMatchObject({ ok: true, code: null });
    expect(result.ok && result.lines).toHaveLength(1);
  }, 40_000);

  it('does not throw for a command that cannot be started', async () => {
    const result = await spawnChild(Date.now, 'definitely-not-a-real-command-xyz')({ args: [], prompt: '', cwd: tmpdir(), timeoutMs: 10_000 });
    expect(result.ok === false || result.code !== 0).toBe(true);
  });
});

describe('stub', () => {
  const make = () => {
    const point = pointOf(reachableSession());
    return createStub(point);
  };
  const tool = (id: number, name: string, args: Record<string, unknown>) => ({
    jsonrpc: '2.0',
    id,
    method: 'tools/call',
    params: { name, arguments: args },
  });
  const replyText = (response: ReturnType<ReturnType<typeof make>['handle']>) =>
    ((response?.result as { content: { text: string }[] }).content[0]!.text);

  it('speaks the handshake: initialize, ping, tools/list, and silence for notifications', () => {
    const stub = make();
    expect(stub.handle({ id: 1, method: 'initialize', params: { protocolVersion: '2025-03-26' } })?.result).toMatchObject({
      protocolVersion: '2025-03-26',
      serverInfo: { name: 'stub' },
    });
    expect(stub.handle({ method: 'notifications/initialized' })).toBeUndefined();
    expect(stub.handle({ id: 2, method: 'ping' })?.result).toEqual({});
    const listed = (stub.handle({ id: 3, method: 'tools/list' })?.result as { tools: { name: string }[] }).tools;
    expect(listed.map((t) => t.name)).toEqual(['Read', 'Grep', 'Glob', 'Edit', 'Bash']);
    expect(listed).toEqual(STUB_TOOLS);
    expect(stub.handle({ id: 4, method: 'resources/list' })?.error?.code).toBe(-32601);
  });

  it('answers a lookup from the recorded history, and says not available for anything else', () => {
    const stub = make();
    expect(replyText(stub.handle(tool(1, 'Read', { file_path: 'src\\billing.ts' })))).toContain(VALUE);
    const missing = stub.handle(tool(2, 'Read', { file_path: 'src/none.ts' }));
    expect(replyText(missing)).toBe(NOT_AVAILABLE);
    expect((missing?.result as { isError: boolean }).isError).toBe(true);
    expect(replyText(stub.handle(tool(3, 'Grep', { pattern: 'nothing recorded' })))).toBe(NOT_AVAILABLE);
  });

  it('ends the conversation at the terminal action and after the lookup limit', () => {
    const stub = make();
    expect(replyText(stub.handle(tool(1, 'Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'n' })))).toBe(STOP);
    for (let i = 0; i < MAX_LOOKUPS; i++) {
      expect(replyText(stub.handle(tool(10 + i, 'Grep', { pattern: `p${i}` })))).toBe(NOT_AVAILABLE);
    }
    expect(replyText(stub.handle(tool(20, 'Grep', { pattern: 'one too many' })))).toBe(STOP);
  });
});

// --- classification ----------------------------------------------------------------------

describe('runPoint', () => {
  const setup = (messages: Message[]) => {
    const point = pointOf(messages);
    return { point, context: pointContext(point) };
  };

  it('same: the matching call at once, no lookups, timed to the call', async () => {
    const { point, context } = setup(reachableSession());
    const model = scripted([ok(stream({ calls: [edit(VALUE)] }))]);
    const outcome = await runPoint(point, context, 'the prompt', deps(model.runner));
    expect(outcome).toEqual({ class: 'same', lookups: 0, seconds: 1, tokens: 110 });
    expect(model.requests[0]).toMatchObject({ prompt: 'the prompt', cwd: '/scratch/work' });
    expect(model.requests[0]!.args).toContain('/scratch/mcp.json');
  });

  it('recovered: a Read lookup, then the matching call', async () => {
    const { point, context } = setup(reachableSession());
    const model = scripted([ok(stream({ calls: [call('Read', { file_path: 'src\\billing.ts' }), edit(VALUE)] }))]);
    expect(await runPoint(point, context, 'p', deps(model.runner))).toEqual({ class: 'recovered', lookups: 1, seconds: 2, tokens: 110 });
  });

  it('is not harness-limited when the stub answered a lookup, or when there were none', async () => {
    const { point, context } = setup(reachableSession());
    const answered = stream({ calls: [call('Read', { file_path: 'src/billing.ts' }), call('Grep', { pattern: 'x' })] });
    expect(await runPoint(point, context, 'p', deps(scripted([ok(answered)]).runner))).toMatchObject({ class: 'gave-up', lookups: 2 });
    expect(await runPoint(point, context, 'p', deps(scripted([ok(answered)]).runner))).not.toHaveProperty('harnessLimited');
    expect(await runPoint(point, context, 'p', deps(scripted([ok(stream())]).runner))).not.toHaveProperty('harnessLimited');
  });

  it('names the tools of lookups the stub cannot answer, and not those it can', async () => {
    const { point, context } = setup(reachableSession());
    const lookups = [call('Read', { file_path: 'src/billing.ts' }), call('Bash', { command: 'ls' }), call('Grep', { pattern: 'x' })];
    const model = scripted([ok(stream({ calls: [...lookups, edit(VALUE)] }))]);
    expect(await runPoint(point, context, 'p', deps(model.runner))).toMatchObject({
      class: 'recovered',
      lookups: 3,
      unanswered: ['Bash', 'Grep'],
    });
  });

  it('wrong: a call that is not the recorded step on a reachable point', async () => {
    const { point, context } = setup(reachableSession());
    const outcome = await runPoint(point, context, 'p', deps(scripted([ok(stream({ calls: [edit('something unrelated')] }))]).runner));
    expect(outcome.class).toBe('wrong');
  });

  it('unreachable: a non-matching call when no stub could return the lost value', async () => {
    const { point, context } = setup(unreachableSession());
    const stream1 = stream({ calls: [edit('something unrelated', 'src/elsewhere.ts')] });
    expect((await runPoint(point, context, 'p', deps(scripted([ok(stream1)]).runner))).class).toBe('unreachable');
  });

  it('gave-up, with its reason: no tool call, stopped after lookups, and the sixth lookup', async () => {
    const { point, context } = setup(reachableSession());
    expect(await runPoint(point, context, 'p', deps(scripted([ok(stream())]).runner))).toMatchObject({
      class: 'gave-up',
      reason: 'no tool call',
    });
    const stopped = stream({ calls: [call('Grep', { pattern: 'a' }), call('Grep', { pattern: 'b' })] });
    expect(await runPoint(point, context, 'p', deps(scripted([ok(stopped)]).runner))).toMatchObject({
      class: 'gave-up',
      lookups: 2,
      unanswered: ['Grep', 'Grep'],
      harnessLimited: true,
      reason: 'no final action',
    });
    const lookups = Array.from({ length: 6 }, (_, i) => call('Grep', { pattern: `p${i}` }));
    expect(await runPoint(point, context, 'p', deps(scripted([ok(stream({ calls: lookups }))]).runner))).toMatchObject({
      class: 'gave-up',
      lookups: 6,
      unanswered: ['Grep', 'Grep', 'Grep', 'Grep', 'Grep'],
      harnessLimited: true,
      reason: 'lookup limit',
    });
  });

  it('failed, with a fixed reason: cannot start, timeout, exit without result, stub not connected, error result', async () => {
    const { point, context } = setup(reachableSession());
    const reasonOf = async (result: ChildResult) => runPoint(point, context, 'p', deps(scripted([result]).runner));
    expect(await reasonOf({ ok: false, reason: 'cannot start claude' })).toMatchObject({
      class: 'failed',
      reason: 'cannot start claude',
      tokens: 0,
      unmetered: true,
    });
    expect(await reasonOf({ ok: false, reason: 'timeout' })).toMatchObject({ class: 'failed', reason: 'timeout' });
    expect(await reasonOf(ok(stream({ result: false }), 1))).toMatchObject({ class: 'failed', reason: 'exit 1' });
    expect(await reasonOf(ok(stream({ result: false }), 0))).toMatchObject({ class: 'failed', reason: 'no result' });
    expect(await reasonOf(ok(stream({ init: { mcp_servers: [] } })))).toMatchObject({ class: 'failed', reason: 'stub not connected (not listed)' });
    expect(await reasonOf(ok(stream({ init: { mcp_servers: [{ name: 'stub', status: 'failed' }] } })))).toMatchObject({
      class: 'failed',
      reason: 'stub not connected (failed)',
    });
    expect(await reasonOf(ok(stream({ init: { mcp_servers: [{ name: 'stub', status: 'pending' }] } })))).toMatchObject({
      class: 'failed',
      reason: 'stub still pending at start',
    });
    expect(await reasonOf(ok(stream({ isError: true })))).toMatchObject({ class: 'failed', reason: 'error result' });
    expect(await reasonOf(ok(stream({ init: { mcp_servers: [{ name: 'stub', status: 'connected' }], tools: ['Bash'] } })))).toMatchObject({
      class: 'failed',
      reason: 'stub tools not offered',
    });
    const compacted = [...stream({ calls: [edit(VALUE)] }), event({ type: 'system', subtype: 'compact_boundary' }, 5000)];
    expect(await reasonOf(ok(compacted))).toMatchObject({ class: 'failed', reason: 'auto-compacted' });
  });

  it('classifies a child that began with the stub pending but then used it', async () => {
    const { point, context } = setup(reachableSession());
    const pending = { mcp_servers: [{ name: 'stub', status: 'pending' }] };
    const lines = stream({ init: pending, calls: [edit(VALUE)] });
    expect((await runPoint(point, context, 'p', deps(scripted([ok(lines)]).runner))).class).toBe('same');
  });

  it('marks an outcome without reported usage as unmetered, but still classifies it', async () => {
    const { point, context } = setup(reachableSession());
    const outcome = await runPoint(point, context, 'p', deps(scripted([ok(stream({ calls: [edit(VALUE)], usage: null }))]).runner));
    expect(outcome).toMatchObject({ class: 'same', tokens: 0, unmetered: true });
  });

  it('classifies a proposed edit and writes nothing', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'behaviour-real-'));
    dirs.push(dir);
    const real = join(dir, 'billing.ts');
    writeFileSync(real, VALUE);
    const { point, context } = setup(reachableSession());
    await runPoint(point, context, 'p', deps(scripted([ok(stream({ calls: [edit(VALUE, real)] }))]).runner));
    expect(readFileSync(real, 'utf8')).toBe(VALUE);
    expect(readdirSync(dir)).toEqual(['billing.ts']);
  });
});

describe('runText', () => {
  it('returns the result text and its cost, and says why when there is none', async () => {
    const good = await runText('prompt', deps(scripted([ok(stream({ text: 'the gist' }))]).runner));
    expect(good).toEqual({ text: 'the gist', tokens: 110, unmetered: false });
    expect((await runText('p', deps(scripted([ok(stream({ text: '' }))]).runner))).reason).toBe('empty summary');
    expect((await runText('p', deps(scripted([ok(stream({ isError: true }))]).runner))).reason).toBe('error result');
    expect((await runText('p', deps(scripted([{ ok: false, reason: 'timeout' }]).runner))).reason).toBe('timeout');
  });
});

// --- report ------------------------------------------------------------------------

const outcome = (c: Outcome['class'], lookups = 0, seconds = 1, reason?: string, unanswered?: string[]): Outcome => ({
  class: c,
  lookups,
  seconds,
  tokens: 10,
  ...(reason === undefined ? {} : { reason }),
  ...(unanswered === undefined ? {} : { unanswered }),
});

const record = (control: Outcome, compacted: Outcome, session = 's.jsonl'): PointRecord => ({
  session,
  messageIndex: 21,
  tool: 'Edit',
  kinds: ['editTarget', 'path'],
  rules: ['rule1'],
  outcomes: { control, compacted },
});

const info = (extra: Partial<RunInfo> = {}): RunInfo => ({
  model: 'test-model',
  sessions: 3,
  available: 5,
  skipped: 0,
  tried: 2,
  tokens: 40,
  tokenCap: 1000,
  stopped: undefined,
  summaryRan: false,
  ...extra,
});

describe('report', () => {
  it('prints counts beside rates, leaves failed runs out of the rates, and gives recovered medians', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('same'), outcome('recovered', 2, 3)),
          record(outcome('same'), outcome('failed', 0, 1, 'timeout')),
          record(outcome('wrong'), outcome('recovered', 4, 5)),
        ],
        info(),
      ),
    );
    expect(text).toContain('Arm: compacted (2 judged, 1 failed to run)');
    expect(text).toMatch(/recovered\s+2\/2 \(100%\)/);
    expect(text).toMatch(/same\s+2\/3 \(67%\)/);
    expect(text).toContain('recovered: median 3 extra lookups, 4.0 s to the final action');
  });

  it('shows the doubt line only for kept points', () => {
    const records = [record(outcome('same'), outcome('recovered', 1, 2))];
    expect(formatReport(summarize(records, info({ kept: true })))).toContain('doubt (lookups although nothing was lost): compacted 1, control 0');
    expect(formatReport(summarize(records, info()))).not.toContain('doubt');
  });

  it('lists the fixed reasons of failed runs per arm, and why the run stopped', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('failed', 0, 1, 'exit 1'), outcome('failed', 0, 1, 'stub not connected')),
          record(outcome('failed', 0, 1, 'exit 1'), outcome('failed', 0, 1, 'stub not connected')),
        ],
        info({ stopped: 'every arm failed to run' }),
      ),
    );
    expect(text).toContain('failed to run: exit 1 x2');
    expect(text).toContain('failed to run: stub not connected x2');
    expect(text).toContain('stopped early (2 point(s) done): every arm failed to run');
  });

  it('says the difference is within the control arm when compacted does not exceed it', () => {
    const text = formatReport(summarize([record(outcome('wrong'), outcome('wrong'))], info()));
    expect(text).toContain('wrong: compacted 0, control 0 (left out: 1 point(s) wrong in the control arm too, so not decided by the history; compacted wrong there: 1) - within what the control arm shows');
    const worse = formatReport(summarize([record(outcome('same'), outcome('wrong'))], info()));
    expect(worse).toContain('wrong: compacted 1, control 0 - more than the control arm shows');
  });

  it('leaves points the control arm got wrong out of the wrong comparison, but still counts a compacted-only miss', () => {
    const text = formatReport(
      summarize([record(outcome('wrong'), outcome('same')), record(outcome('wrong'), outcome('wrong')), record(outcome('same'), outcome('wrong'))], info()),
    );
    expect(text).toContain('wrong: compacted 1, control 0 (left out: 2 point(s) wrong in the control arm too, so not decided by the history; compacted wrong there: 1) - more than the control arm shows');
  });

  it('states that the summary arm was not run, prints no figure for it, and names the flattening caveat', () => {
    const text = formatReport(summarize([record(outcome('same'), outcome('same'))], info()));
    expect(text).toContain('Summary arm not run');
    expect(text).not.toContain('Arm: summary');
    expect(text).toContain('reaches the model as text');
  });

  it('says why an arm gave up, with the fixed reasons and their counts', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('gave-up', 0, 1, 'no tool call'), outcome('gave-up', 6, 1, 'lookup limit')),
          record(outcome('gave-up', 0, 1, 'no tool call'), outcome('same')),
        ],
        info(),
      ),
    );
    expect(text).toMatch(/gave-up\s+2\/2 \(100%\) - no tool call x2/);
    expect(text).toMatch(/gave-up\s+1\/2 \(50%\) - lookup limit \(6 lookups, 0 unanswerable\) x1/);
  });

  it('leaves harness-limited gave-ups out of the control comparison and says so', () => {
    const limited = { ...outcome('gave-up', 3, 1, 'no final action', ['Bash', 'Grep', 'Read']), harnessLimited: true };
    const text = formatReport(summarize([record(limited, outcome('recovered', 2, 2)), record(outcome('gave-up', 0, 1, 'no tool call'), outcome('same'))], info()));
    expect(text).toContain('harness-limited: 1 of the gave-up (the stub answered none of their lookups)');
    expect(text).toContain('gave-up: compacted 0, control 1 (harness-limited left out: compacted 0, control 1) - within what the control arm shows');
  });

  it('shows how many lookups were unanswerable and which tools, for gave-up and recovered', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('gave-up', 3, 1, 'no final action', ['Bash', 'Grep']), outcome('recovered', 2, 2, undefined, ['Bash'])),
          record(outcome('gave-up', 3, 1, 'no final action', ['Bash', 'Grep']), outcome('recovered', 2, 2, undefined, ['Bash'])),
        ],
        info(),
      ),
    );
    expect(text).toMatch(/gave-up\s+2\/2 \(100%\) - no final action \(3 lookups, 2 unanswerable: Bash, Grep\) x2/);
    expect(text).toContain('recovered: median 2 extra lookups, 2.0 s to the final action; unanswerable lookups: Bash x2');
  });

  it('shows the time to a final action per arm and the paired extra effort of compacted over control', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('same', 0, 10), outcome('recovered', 3, 70)),
          record(outcome('same', 0, 20), outcome('recovered', 1, 50)),
          record(outcome('gave-up', 0, 5, 'no tool call'), outcome('same', 0, 9)),
        ],
        info(),
      ),
    );
    expect(text).toContain('time to final action: median 15.0 s over 2 outcome(s)');
    expect(text).toContain('time to final action: median 50.0 s over 3 outcome(s)');
    expect(text).toContain('extra effort of compacted over control: median +45.0 s and +2 lookups, over 2 point(s) where both arms reached a final action');
  });

  it('says there is nothing to compare when no point reached a final action in both arms', () => {
    const text = formatReport(summarize([record(outcome('gave-up', 0, 1, 'no tool call'), outcome('same'))], info()));
    expect(text).toContain('extra effort: no point reached a final action in both arms');
  });

  it('is a pure function of its input', () => {
    const records = [record(outcome('same'), outcome('wrong'))];
    expect(formatReport(summarize(records, info()))).toBe(formatReport(summarize(records, info())));
  });
});

// --- the runner -----------------------------------------------------------------------------

const ARGS = ['--model', 'test-model', '--max-points', '2', '--token-cap', '100000'];

/** Acts like a model that needs the lost value: right with it in view, a guess without it. */
const needsTheValue = () =>
  fakeRunner((request) =>
    ok(stream({ calls: [edit(request.prompt.includes(VALUE) ? VALUE : 'a guess at the code')] })),
  );

const runWith = (argv: string[], runner: ChildRunner, extra: Parameters<typeof run>[1] = {}) =>
  run(argv, { runner, createScratch: fakeScratch, now: clock(), ...extra });

describe('run', () => {
  it('runs both arms end to end over a corpus and prints a report with both', async () => {
    const root = corpus({ 'p/s.jsonl': reachableSession() });
    const model = needsTheValue();
    const scratch = fakeScratch();
    const result = await runWith([...ARGS, '--root', root], model.runner, { createScratch: () => scratch });
    expect(result.code).toBe(0);
    expect(result.output).toContain('Arm: control (1 judged, 0 failed to run)');
    expect(result.output).toContain('Arm: compacted (1 judged, 0 failed to run)');
    expect(result.output).toContain('wrong: compacted 1, control 0 - more than the control arm shows');
    expect(model.requests).toHaveLength(2);
    expect(scratch.cleanup).toHaveBeenCalledTimes(1);
  });

  it('gives the same report twice for the same corpus and scripted model', async () => {
    const root = corpus({ 'p/s.jsonl': reachableSession(), 'p/t.jsonl': unreachableSession() });
    const once = async () => (await runWith([...ARGS, '--root', root], needsTheValue().runner)).output;
    expect(await once()).toBe(await once());
  });

  it('keeps transcript text out of the report and out of every argument list', async () => {
    const secretValue = 'return decryptVault("hunter2-super-secret-token");';
    const secretFile = 'src/secret-path-xyz.ts';
    const secretCommand = 'curl https://internal.example/secret-command-xyz';
    const messages = session(
      [
        ...pair('Bash', { command: secretCommand }, 'ok'),
        ...pair('Read', { file_path: secretFile }, around(5000, secretValue, 5000)),
      ],
      editing(secretValue, secretFile),
    );
    const root = corpus({ 'p/s.jsonl': messages });
    const model = fakeRunner(() => ok(stream({ calls: [call('Edit', { file_path: secretFile, old_string: secretValue, new_string: 'secret-new-text' })] })));
    const result = await runWith([...ARGS, '--root', root], model.runner);
    expect(result.code).toBe(0);
    for (const secret of ['hunter2', 'secret-path-xyz', 'secret-command-xyz', 'secret-new-text', 'curl']) {
      expect(result.output).not.toContain(secret);
      expect(textOf(model.requests.map((r) => r.args))).not.toContain(secret);
    }
  });

  it('refuses without every required input, naming each, before any corpus read or child process', async () => {
    const model = scripted([]);
    const reader = vi.fn(lostPoints);
    const scratch = vi.fn(fakeScratch);
    const result = await runWith([], model.runner, { lostPoints: reader, createScratch: scratch });
    expect(result.code).toBe(1);
    for (const name of ['--model', '--max-points', '--token-cap']) expect(result.output).toContain(name);
    const partial = await runWith(['--model', 'm', '--max-points', '1'], model.runner, { lostPoints: reader, createScratch: scratch });
    expect(partial.output).toContain('--token-cap');
    expect(partial.output).not.toContain('--model');
    expect(model.impl).not.toHaveBeenCalled();
    expect(reader).not.toHaveBeenCalled();
    expect(scratch).not.toHaveBeenCalled();
  });

  it('refuses a missing root and a corpus with no lost point, starting nothing', async () => {
    const model = scripted([]);
    const scratch = vi.fn(fakeScratch);
    const missing = await runWith([...ARGS, '--root', join(tmpdir(), 'behaviour-no-such-root')], model.runner, { createScratch: scratch });
    expect(missing.code).toBe(1);
    const empty = await runWith([...ARGS, '--root', corpus({})], model.runner, { createScratch: scratch });
    expect(empty.code).toBe(1);
    expect(empty.output).toContain('No lost point');
    expect(model.impl).not.toHaveBeenCalled();
    expect(scratch).not.toHaveBeenCalled();
  });

  it('with --kept runs on points where nothing was lost and reports doubt', async () => {
    const root = corpus({ 'p/k.jsonl': keptSession(), 'p/l.jsonl': reachableSession() });
    // The control edits at once; the compacted arm looks the file up first.
    const model = fakeRunner((_request, count) =>
      ok(stream({ calls: count % 2 === 1 ? [edit(VALUE)] : [call('Read', { file_path: 'src/billing.ts' }), edit(VALUE)] })),
    );
    const result = await runWith([...ARGS, '--root', root, '--kept'], model.runner);
    expect(result.code).toBe(0);
    expect(result.output).toContain('where compaction lost nothing');
    expect(result.output).toContain('1 kept points available');
    expect(result.output).toContain('doubt (lookups although nothing was lost): compacted 1, control 0');
    expect(model.requests).toHaveLength(2);
  });

  it('refuses a corpus with no kept point with --kept, starting nothing', async () => {
    const model = scripted([]);
    const result = await runWith([...ARGS, '--root', corpus({ 'p/l.jsonl': reachableSession() }), '--kept'], model.runner);
    expect(result.code).toBe(1);
    expect(result.output).toContain('No kept point');
    expect(model.impl).not.toHaveBeenCalled();
  });

  it('refuses arguments that are not positive whole numbers', async () => {
    const result = await runWith(['--model', 'm', '--max-points', '0', '--token-cap', 'lots'], scripted([]).runner);
    expect(result.code).toBe(1);
  });

  describe('selecting points', () => {
    // a.jsonl has about 10k characters of history before its point, b.jsonl about 5k
    const root = () => corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': unreachableSession() });

    it('runs only points whose history fits --max-prefix-chars, and says so', async () => {
      const model = needsTheValue();
      const result = await runWith([...ARGS, '--root', root(), '--max-prefix-chars', '8000'], model.runner);
      expect(result.code).toBe(0);
      expect(model.requests).toHaveLength(2);
      expect(result.output).toContain('b.jsonl');
      expect(result.output).not.toContain('a.jsonl');
      expect(result.output).toContain('Selection: history at most 8000 characters (1 of 2 points)');
    });

    it('continues in order after --skip, so a later run does not repeat an earlier one', async () => {
      const one = ['--model', 'test-model', '--max-points', '1', '--token-cap', '100000', '--root', root()];
      const first = await runWith([...one, '--skip', '0'], needsTheValue().runner);
      const next = await runWith([...one, '--skip', '1'], needsTheValue().runner);
      expect(first.output).toContain('a.jsonl');
      expect(first.output).not.toContain('b.jsonl');
      expect(next.output).toContain('b.jsonl');
      expect(next.output).not.toContain('a.jsonl');
      expect(next.output).toContain('Selection: first 1 skipped, then in order');
    });

    it('prints no selection line without either flag', async () => {
      const result = await runWith([...ARGS, '--root', root()], needsTheValue().runner);
      expect(result.output).not.toContain('Selection:');
    });

    it('refuses, starting nothing, when the size limit or the skip leaves no point', async () => {
      const model = scripted([]);
      const scratch = vi.fn(fakeScratch);
      for (const extra of [['--max-prefix-chars', '10'], ['--skip', '2']]) {
        const result = await runWith([...ARGS, '--root', root(), ...extra], model.runner, { createScratch: scratch });
        expect(result.code).toBe(1);
        expect(result.output).toContain('No lost point left');
      }
      expect(model.impl).not.toHaveBeenCalled();
      expect(scratch).not.toHaveBeenCalled();
    });

    it('refuses a size limit or skip that is not a whole number', async () => {
      for (const extra of [['--max-prefix-chars', '0'], ['--max-prefix-chars', 'big'], ['--skip', '-1'], ['--skip', 'some']]) {
        expect((await runWith([...ARGS, '--root', root(), ...extra], scripted([]).runner)).code).toBe(1);
      }
    });
  });

  it('finishes the point in progress at the token cap and starts no more', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': reachableSession() });
    const model = needsTheValue();
    const result = await runWith(['--model', 'm', '--max-points', '2', '--token-cap', '1', '--root', root], model.runner);
    expect(result.code).toBe(0);
    expect(model.requests).toHaveLength(2);
    expect(result.output).toContain('stopped early (1 point(s) done): token cap reached');
    expect(result.output).toContain('Tokens: 220 used, cap 1');
  });

  it('stops after a point whose usage was not reported, because the cap cannot be enforced', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': reachableSession() });
    const model = fakeRunner(() => ok(stream({ calls: [edit(VALUE)], usage: null })));
    const result = await runWith([...ARGS, '--root', root], model.runner);
    expect(model.requests).toHaveLength(2);
    expect(result.output).toContain('usage not reported, so the cap cannot be enforced');
  });

  it('stops when every arm failed, and lists the reasons', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': reachableSession() });
    const model = fakeRunner(() => ({ ok: false, reason: 'cannot start claude' }));
    const result = await runWith([...ARGS, '--root', root], model.runner);
    expect(model.requests).toHaveLength(2);
    expect(result.output).toContain('failed to run: cannot start claude x1');
    expect(result.output).toContain('every arm failed to run');
  });

  it('removes the scratch directory even when a child cannot be started', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const scratch = fakeScratch();
    const boom = fakeRunner(() => {
      throw new Error('spawn exploded');
    });
    await expect(runWith([...ARGS, '--root', root], boom.runner, { createScratch: () => scratch })).rejects.toThrow('spawn exploded');
    expect(scratch.cleanup).toHaveBeenCalledTimes(1);
  });
});

describe('run with the summary arm', () => {
  const SUMMARY = 'SUMMARY-TEXT-ONLY: the invoice total needs fixing';
  /** A request without the stub is the summary call; every other request gets the right edit. */
  const summarising = () =>
    fakeRunner((request) =>
      request.args.includes('--mcp-config') ? ok(stream({ calls: [edit(VALUE)] })) : ok(stream({ text: SUMMARY })),
    );

  it('adds a third, approximate arm built from one extra call, counted toward the tokens', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const model = summarising();
    const result = await runWith([...ARGS, '--root', root, '--summary'], model.runner);
    expect(result.code).toBe(0);
    expect(result.output).toContain('Arm: summary (approximate');
    expect(result.output).toContain('summary=same');
    expect(model.requests).toHaveLength(4);
    expect(result.output).toContain('Tokens: 440 used');
    const armPrompt = model.requests[3]!.prompt;
    expect(armPrompt).toContain(SUMMARY);
    expect(armPrompt).not.toContain(VALUE);
    expect(model.requests[2]!.prompt).toContain(SUMMARY_REQUEST);
  });

  it('counts a failed summary call as a failed run of that arm, with its reason', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const model = fakeRunner((request) =>
      request.args.includes('--mcp-config') ? ok(stream({ calls: [edit(VALUE)] })) : ok(stream({ isError: true })),
    );
    const result = await runWith([...ARGS, '--root', root, '--summary'], model.runner);
    expect(result.output).toContain('(0 judged, 1 failed to run)');
    expect(result.output).toContain('failed to run: error result x1');
  });

  it('says the summary arm was not run, and prints no figure for it, without --summary', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const result = await runWith([...ARGS, '--root', root], summarising().runner);
    expect(result.output).toContain('Summary arm not run');
    expect(result.output).not.toContain('Arm: summary');
    expect(result.output).not.toContain('summary=');
  });
});

describe('run --check', () => {
  const CHECK = ['--check', '--model', 'sonnet'];

  it('reports OK when the model called a stub tool, sends only the synthetic prompt, and needs no cap', async () => {
    const model = fakeRunner(() => ok(stream({ calls: [call('Read', { file_path: 'check.txt' })] })));
    const scratch = fakeScratch();
    const result = await runWith(CHECK, model.runner, { createScratch: () => scratch });
    expect(result.code).toBe(0);
    expect(result.output).toContain('Result: OK');
    expect(result.output).toContain('Stub tool calls seen: 1');
    expect(result.output).toContain('Tokens used: 110');
    expect(result.output).not.toContain('Child result');
    expect(model.requests).toHaveLength(1);
    expect(model.requests[0]!.prompt).toContain('connectivity check');
    expect(scratch.file).toHaveBeenCalledTimes(1);
    expect(scratch.cleanup).toHaveBeenCalledTimes(1);
  });

  it('reports a problem with its reason when the child cannot use the stub', async () => {
    const notListed = await runWith(CHECK, fakeRunner(() => ok(stream({ init: { mcp_servers: [] } }))).runner);
    expect(notListed.code).toBe(1);
    expect(notListed.output).toContain('PROBLEM');
    expect(notListed.output).toContain('stub not connected (not listed)');
    const rejected = await runWith(
      CHECK,
      fakeRunner(() => ok(stream({ isError: true, subtype: 'error_during_execution', text: 'Please run /login\nto sign in' }))).runner,
    );
    expect(rejected.code).toBe(1);
    expect(rejected.output).toContain('error result');
    expect(rejected.output).toContain('Child result (error_during_execution): Please run /login to sign in');
    const noResult = await runWith(CHECK, fakeRunner(() => ok(stream({ result: false }), 3)).runner);
    expect(noResult.output).toContain('Child exited with code 3 and printed no result');
    const silent = await runWith(CHECK, fakeRunner(() => ok(stream())).runner);
    expect(silent.code).toBe(1);
    expect(silent.output).toContain('the model made no stub tool call');
  });

  it('needs --model and nothing else, and refuses without it before starting anything', async () => {
    const model = scripted([]);
    const scratch = vi.fn(fakeScratch);
    const result = await runWith(['--check'], model.runner, { createScratch: scratch });
    expect(result.code).toBe(1);
    expect(result.output).toContain('--model');
    expect(model.impl).not.toHaveBeenCalled();
    expect(scratch).not.toHaveBeenCalled();
  });
});
