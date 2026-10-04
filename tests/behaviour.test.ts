import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Message, ToolUse } from '../src/index.js';
import {
  buildLookup,
  compactedHistory,
  controlHistory,
  matchesRecorded,
  pointContext,
  toApiMessages,
} from '../tools/behaviour/history.js';
import type { ApiBlock } from '../tools/behaviour/history.js';
import { formatReport, summarize } from '../tools/behaviour/report.js';
import type { PointRecord, RunInfo } from '../tools/behaviour/report.js';
import { run } from '../tools/behaviour/run.js';
import { lostPoints, sample } from '../tools/behaviour/select.js';
import type { LostPoint } from '../tools/behaviour/select.js';
import { runPoint } from '../tools/behaviour/session.js';
import type { Outcome, SessionDeps } from '../tools/behaviour/session.js';

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

// --- temp corpus and scripted fake fetch -----------------------------------------

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
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

const toolUse = (name: string, input: Record<string, unknown>, id = `call${++nextId}`): ApiBlock => ({
  type: 'tool_use',
  id,
  name,
  input,
});

const okBody = (content: ApiBlock[], input = 100, output = 10): Response =>
  new Response(JSON.stringify({ content, stop_reason: 'tool_use', usage: { input_tokens: input, output_tokens: output } }));

interface Recorded {
  messages: { role: string; content: unknown }[];
  tools?: unknown;
}

/** A fetch that answers from a function of the parsed request and remembers every request. */
function fakeFetch(answer: (request: Recorded, count: number) => Response) {
  const requests: Recorded[] = [];
  const impl = vi.fn(async (_url: unknown, init?: { body?: unknown }) => {
    const request = JSON.parse(String(init?.body)) as Recorded;
    requests.push(request);
    return answer(request, requests.length);
  });
  return { fetch: impl as unknown as typeof fetch, requests, impl };
}

/** A fetch that plays the given replies in order. */
function scripted(replies: (ApiBlock[] | Response)[]) {
  return fakeFetch((_request, count) => {
    const reply = replies[count - 1];
    if (reply === undefined) return new Response('no more replies', { status: 500 });
    return reply instanceof Response ? reply : okBody(reply);
  });
}

function clock(): () => number {
  let t = 0;
  return () => (t += 1000);
}

function deps(fetchImpl: typeof fetch): SessionDeps {
  return { fetch: fetchImpl, apiKey: 'sk-ant-test-key', model: 'test-model', now: clock() };
}

function pointOf(messages: Message[]): LostPoint {
  const root = corpus({ 'p/s.jsonl': messages });
  const { points } = lostPoints(root);
  if (points.length !== 1) throw new Error(`expected one lost point, got ${points.length}`);
  return points[0]!;
}

const textOf = (value: unknown): string => JSON.stringify(value);

// --- selection and histories ----------------------------------------------------

describe('selection', () => {
  it('orders lost points by session name then index, whatever the folder order', () => {
    const root = corpus({ 'p1/b.jsonl': reachableSession(), 'p2/a.jsonl': reachableSession() });
    const { points, sessions } = lostPoints(root);
    expect(sessions).toBe(2);
    expect(points.map((p) => p.session)).toEqual(['a.jsonl', 'b.jsonl']);
    expect(lostPoints(root).points.map((p) => p.messageIndex)).toEqual(points.map((p) => p.messageIndex));
  });

  it('ignores points without a loss and sessions that are not lost points', () => {
    const quiet = session([], { tool: 'Read', input: { file_path: 'src/billing.ts' } });
    expect(lostPoints(corpus({ 'p/q.jsonl': quiet })).points).toEqual([]);
  });

  it('takes evenly spaced entries, all when there are fewer, the same every time', () => {
    const items = Array.from({ length: 10 }, (_, i) => i);
    expect(sample(items, 3)).toEqual([0, 3, 6]);
    expect(sample(items, 3)).toEqual(sample(items, 3));
    expect(sample(items, 20)).toEqual(items);
  });
});

describe('histories', () => {
  it('the compacted history lacks the lost value that the control history still has', () => {
    const point = pointOf(reachableSession());
    expect(textOf(controlHistory(point.messages, point.messageIndex))).toContain(VALUE);
    expect(textOf(compactedHistory(point.messages, point.messageIndex))).not.toContain(VALUE);
  });

  it('maps messages to API messages, keeps tool pairs, drops empty text, merges same-role runs', () => {
    const messages: Message[] = [
      chat('user', 'go'),
      { role: 'assistant', text: '', toolUses: [{ tool_use_id: 't1', tool: 'Read', input: { file_path: 'a.ts' } }] },
      { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 't1', text: 'boom', isError: true }] },
      chat('assistant', ''),
      chat('user', 'and then?'),
      chat('assistant', 'done'),
    ];
    expect(toApiMessages(messages)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'go' }] },
      { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'a.ts' } }] },
      {
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 't1', content: 'boom', is_error: true },
          { type: 'text', text: 'and then?' },
        ],
      },
      { role: 'assistant', content: [{ type: 'text', text: 'done' }] },
    ]);
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
    const edit = (isError: boolean): Message[] =>
      pair('Edit', { file_path: 'src/a.ts', old_string: 'o', new_string: 'n' }, 'r', isError);
    expect(buildLookup([...read, ...edit(false)]).serve({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBeUndefined();
    expect(buildLookup([...read, ...edit(true)]).serve({ tool: 'Read', input: { file_path: 'src/a.ts' } })).toBe('BODY-A');
  });

  it('reports a point as reachable only when a lost value sits in a result the stubs can serve', () => {
    expect(pointContext(pointOf(reachableSession())).reachable).toBe(true);
    expect(pointContext(pointOf(unreachableSession())).reachable).toBe(false);
  });
});

describe('matchesRecorded', () => {
  const step = (tool: string, input: Record<string, unknown>): ToolUse => ({ tool_use_id: 'rec', tool, input });
  const prefix = [chat('user', `see ${around(10, VALUE, 10)}`)];

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
    const edit = (old: unknown) => ({ tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: old } });
    expect(matchesRecorded(edit(VALUE), recorded, prefix)).toBe(true);
    expect(matchesRecorded(edit('computeInvoiceTotal(lineItems'), recorded, prefix)).toBe(true);
    expect(matchesRecorded(edit('somewhere else entirely'), recorded, prefix)).toBe(false);
    expect(matchesRecorded(edit(''), recorded, prefix)).toBe(false);
    expect(matchesRecorded({ tool: 'Edit', input: { file_path: 'src/billing.ts' } }, recorded, prefix)).toBe(false);
    expect(matchesRecorded({ tool: 'Edit', input: { file_path: 'src/x.ts', old_string: VALUE } }, recorded, prefix)).toBe(false);
  });
});

// --- classification --------------------------------------------------------------

describe('runPoint', () => {
  const edit = (old: string, file = 'src/billing.ts') => toolUse('Edit', { file_path: file, old_string: old, new_string: 'n' });

  it('same: the matching call at once, no lookups', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const model = scripted([[edit(VALUE)]]);
    const outcome = await runPoint(context, [{ role: 'user', content: [{ type: 'text', text: 'go' }] }], deps(model.fetch));
    expect(outcome).toEqual({ class: 'same', lookups: 0, seconds: 1, tokens: 110 });
  });

  it('recovered: a Read lookup is answered from the record, then the matching call', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const model = scripted([[toolUse('Read', { file_path: 'src\\billing.ts' }, 'r1')], [edit(VALUE)]]);
    const outcome = await runPoint(context, [{ role: 'user', content: [{ type: 'text', text: 'go' }] }], deps(model.fetch));
    expect(outcome).toEqual({ class: 'recovered', lookups: 1, seconds: 1, tokens: 220 });
    const answer = model.requests[1]!.messages.at(-1)!;
    expect(answer.role).toBe('user');
    expect(textOf(answer.content)).toContain(VALUE);
  });

  it('wrong: a call that is not the recorded step on a reachable point', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const outcome = await runPoint(context, [], deps(scripted([[edit('something unrelated')]]).fetch));
    expect(outcome.class).toBe('wrong');
  });

  it('unreachable: a non-matching call when no stub could return the lost value', async () => {
    const context = pointContext(pointOf(unreachableSession()));
    const outcome = await runPoint(context, [], deps(scripted([[edit('something unrelated', 'src/elsewhere.ts')]]).fetch));
    expect(outcome.class).toBe('unreachable');
  });

  it('gave-up: no tool call', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const outcome = await runPoint(context, [], deps(scripted([[{ type: 'text', text: 'I will stop here' }]]).fetch));
    expect(outcome.class).toBe('gave-up');
  });

  it('gave-up: the sixth lookup', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const lookups = Array.from({ length: 6 }, (_, i) => [toolUse('Grep', { pattern: `p${i}` })]);
    const model = scripted(lookups);
    const outcome = await runPoint(context, [], deps(model.fetch));
    expect(outcome).toMatchObject({ class: 'gave-up', lookups: 6 });
    expect(model.requests).toHaveLength(6);
  });

  it('failed: a model call that does not succeed', async () => {
    const context = pointContext(pointOf(reachableSession()));
    const outcome = await runPoint(context, [], deps(scripted([new Response('no', { status: 500 })]).fetch));
    expect(outcome).toMatchObject({ class: 'failed', tokens: 0 });
    const thrown = vi.fn(async () => {
      throw new Error('socket sk-ant-test-key');
    }) as unknown as typeof fetch;
    expect((await runPoint(context, [], deps(thrown))).class).toBe('failed');
  });

  it('classifies a proposed edit and writes nothing', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'behaviour-real-'));
    dirs.push(dir);
    const real = join(dir, 'billing.ts');
    writeFileSync(real, VALUE);
    const context = pointContext(pointOf(reachableSession()));
    const model = scripted([[toolUse('Edit', { file_path: real, old_string: VALUE, new_string: 'changed' })]]);
    await runPoint(context, [], deps(model.fetch));
    expect(readFileSync(real, 'utf8')).toBe(VALUE);
    expect(readdirSync(dir)).toEqual(['billing.ts']);
  });
});

// --- report ------------------------------------------------------------------------

const outcome = (c: Outcome['class'], lookups = 0, seconds = 1): Outcome => ({ class: c, lookups, seconds, tokens: 10 });

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
  tried: 2,
  tokens: 40,
  tokenCap: 1000,
  stoppedEarly: false,
  summaryRan: false,
  ...extra,
});

describe('report', () => {
  it('prints counts beside rates, leaves failed runs out of the rates, and gives recovered medians', () => {
    const text = formatReport(
      summarize(
        [
          record(outcome('same'), outcome('recovered', 2, 3)),
          record(outcome('same'), outcome('failed')),
          record(outcome('wrong'), outcome('recovered', 4, 5)),
        ],
        info(),
      ),
    );
    expect(text).toContain('Arm: compacted (2 judged, 1 failed to run)');
    expect(text).toMatch(/recovered\s+2\/2 \(100%\)/);
    expect(text).toMatch(/same\s+2\/3 \(67%\)/);
    expect(text).toContain('recovered: median 3 extra lookups, 4.0 s');
  });

  it('says the difference is within the control arm when compacted does not exceed it', () => {
    const text = formatReport(summarize([record(outcome('wrong'), outcome('wrong'))], info()));
    expect(text).toContain('wrong: compacted 1, control 1 - within what the control arm shows');
    const worse = formatReport(summarize([record(outcome('same'), outcome('wrong'))], info()));
    expect(worse).toContain('wrong: compacted 1, control 0 - more than the control arm shows');
  });

  it('states that the summary arm was not run, and prints no figure for it', () => {
    const text = formatReport(summarize([record(outcome('same'), outcome('same'))], info()));
    expect(text).toContain('Summary arm not run');
    expect(text).not.toContain('Arm: summary');
  });

  it('is a pure function of its input', () => {
    const records = [record(outcome('same'), outcome('wrong'))];
    expect(formatReport(summarize(records, info()))).toBe(formatReport(summarize(records, info())));
  });
});

// --- the runner ---------------------------------------------------------------------

const ARGS = ['--model', 'test-model', '--max-points', '2', '--token-cap', '100000'];
const ENV = { ANTHROPIC_API_KEY: 'sk-ant-test-key' };

/** Acts like a model that needs the lost value: right with it in view, a guess without it. */
const needsTheValue = () =>
  fakeFetch((request) => {
    const answer = textOf(request.messages).includes(VALUE) ? VALUE : 'a guess at the code';
    return okBody([toolUse('Edit', { file_path: 'src/billing.ts', old_string: answer, new_string: 'n' })]);
  });

describe('run', () => {
  it('runs both arms end to end over a corpus and prints a report with both', async () => {
    const root = corpus({ 'p/s.jsonl': reachableSession() });
    const model = needsTheValue();
    const result = await run([...ARGS, '--root', root], ENV, { fetch: model.fetch, now: clock() });
    expect(result.code).toBe(0);
    expect(result.output).toContain('Arm: control (1 judged, 0 failed to run)');
    expect(result.output).toContain('Arm: compacted (1 judged, 0 failed to run)');
    expect(result.output).toContain('wrong: compacted 1, control 0 - more than the control arm shows');
    expect(model.requests).toHaveLength(2);
  });

  it('gives the same report twice for the same corpus and scripted model', async () => {
    const root = corpus({ 'p/s.jsonl': reachableSession(), 'p/t.jsonl': unreachableSession() });
    const once = async () => (await run([...ARGS, '--root', root], ENV, { fetch: needsTheValue().fetch, now: clock() })).output;
    expect(await once()).toBe(await once());
  });

  it('keeps transcript text and the key out of the report', async () => {
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
    const model = fakeFetch(() =>
      okBody([toolUse('Edit', { file_path: secretFile, old_string: secretValue, new_string: 'secret-new-text' })]),
    );
    const result = await run([...ARGS, '--root', root], { ANTHROPIC_API_KEY: 'sk-ant-the-real-key' }, { fetch: model.fetch, now: clock() });
    expect(result.code).toBe(0);
    for (const secret of ['hunter2', 'secret-path-xyz', 'secret-command-xyz', 'secret-new-text', 'the-real-key', 'curl']) {
      expect(result.output).not.toContain(secret);
    }
  });

  it('refuses without every required input, naming each, before any corpus read or request', async () => {
    const model = scripted([]);
    const reader = vi.fn(lostPoints);
    const result = await run([], {}, { fetch: model.fetch, lostPoints: reader });
    expect(result.code).toBe(1);
    for (const name of ['--model', '--max-points', '--token-cap', 'ANTHROPIC_API_KEY']) {
      expect(result.output).toContain(name);
    }
    const onlyKey = await run(['--model', 'm', '--max-points', '1'], ENV, { fetch: model.fetch, lostPoints: reader });
    expect(onlyKey.output).toContain('--token-cap');
    expect(onlyKey.output).not.toContain('--model');
    expect(model.impl).not.toHaveBeenCalled();
    expect(reader).not.toHaveBeenCalled();
  });

  it('refuses a missing root and a corpus with no lost point, sending nothing', async () => {
    const model = scripted([]);
    const missing = await run([...ARGS, '--root', join(tmpdir(), 'behaviour-no-such-root')], ENV, { fetch: model.fetch });
    expect(missing.code).toBe(1);
    const empty = await run([...ARGS, '--root', corpus({})], ENV, { fetch: model.fetch });
    expect(empty.code).toBe(1);
    expect(empty.output).toContain('No lost point');
    expect(model.impl).not.toHaveBeenCalled();
  });

  it('refuses arguments that are not positive whole numbers', async () => {
    const result = await run(['--model', 'm', '--max-points', '0', '--token-cap', 'lots'], ENV, { fetch: scripted([]).fetch });
    expect(result.code).toBe(1);
  });

  it('finishes the point in progress at the token cap and starts no more', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession(), 'p/b.jsonl': reachableSession() });
    const model = needsTheValue();
    const result = await run(
      ['--model', 'm', '--max-points', '2', '--token-cap', '1', '--root', root],
      ENV,
      { fetch: model.fetch, now: clock() },
    );
    expect(result.code).toBe(0);
    expect(model.requests).toHaveLength(2);
    expect(result.output).toContain('stopped early (1 point(s) done)');
    expect(result.output).toContain('Tokens: 220 used, cap 1');
  });

  it('counts the tokens of every call, lookups included', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const model = fakeFetch((request) => {
      const last = request.messages.at(-1)!;
      const answered = Array.isArray(last.content) && textOf(last.content).includes('tool_result');
      return okBody(
        answered
          ? [toolUse('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'n' })]
          : [toolUse('Read', { file_path: 'src/billing.ts' })],
      );
    });
    const result = await run([...ARGS, '--root', root], ENV, { fetch: model.fetch, now: clock() });
    expect(model.requests.length).toBeGreaterThanOrEqual(4);
    expect(result.output).toContain(`Tokens: ${model.requests.length * 110} used`);
  });

  it('never puts the key into output or errors, even when requests fail', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const key = 'sk-ant-must-not-leak';
    const broken = vi.fn(async () => {
      throw new Error(`connect failed with ${key}`);
    }) as unknown as typeof fetch;
    const result = await run([...ARGS, '--root', root], { ANTHROPIC_API_KEY: key }, { fetch: broken, now: clock() });
    expect(result.code).toBe(0);
    expect(result.output).not.toContain(key);
    expect(result.output).toContain('failed to run');
    const refused = await run(['--model', 'm'], { ANTHROPIC_API_KEY: key });
    expect(refused.output).not.toContain(key);
  });
});

describe('run with the summary arm', () => {
  const SUMMARY = 'SUMMARY-TEXT-ONLY: the invoice total needs fixing';
  /** Answers a request without tools with a summary, and every other request with the right edit. */
  const summarising = () =>
    fakeFetch((request) =>
      request.tools === undefined
        ? okBody([{ type: 'text', text: SUMMARY }])
        : okBody([toolUse('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'n' })]),
    );

  it('adds a third, approximate arm built from one extra call, counted toward the tokens', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const model = summarising();
    const result = await run([...ARGS, '--root', root, '--summary'], ENV, { fetch: model.fetch, now: clock() });
    expect(result.code).toBe(0);
    expect(result.output).toContain('Arm: summary (approximate');
    expect(result.output).toContain('summary=same');
    expect(model.requests).toHaveLength(4);
    expect(result.output).toContain('Tokens: 440 used');
    const armRequest = model.requests[3]!;
    expect(armRequest.messages).toHaveLength(1);
    expect(textOf(armRequest.messages)).toContain(SUMMARY);
    expect(textOf(armRequest.messages)).not.toContain(VALUE);
  });

  it('counts a failed summary call as a failed run of that arm', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const model = fakeFetch((request) =>
      request.tools === undefined
        ? new Response('no', { status: 500 })
        : okBody([toolUse('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'n' })]),
    );
    const result = await run([...ARGS, '--root', root, '--summary'], ENV, { fetch: model.fetch, now: clock() });
    expect(result.output).toContain('Arm: summary (approximate: does not re-attach recently read files) (0 judged, 1 failed to run)');
  });

  it('says the summary arm was not run, and prints no figure for it, without --summary', async () => {
    const root = corpus({ 'p/a.jsonl': reachableSession() });
    const result = await run([...ARGS, '--root', root], ENV, { fetch: summarising().fetch, now: clock() });
    expect(result.output).toContain('Summary arm not run');
    expect(result.output).not.toContain('Arm: summary');
    expect(result.output).not.toContain('summary=');
  });
});

