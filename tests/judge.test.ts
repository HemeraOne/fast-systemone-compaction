import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnChild } from '../tools/behaviour/model.js';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { Message } from '../src/index.js';
import {
  CALIBRATION_MALFORMED,
  CALIBRATION_UNREADABLE,
  calibrate,
  JUDGE_PROMPT,
  JudgeBudget,
  judgeIfOpen,
  loadCalibrationSet,
  parseCalibrationSet,
  parseVerdict,
} from '../tools/behaviour/judge.js';
import type { CalibrationPair } from '../tools/behaviour/judge.js';
import type { ChildRequest, ChildResult, ChildRunner, Scratch, StreamLine } from '../tools/behaviour/model.js';
import { run } from '../tools/behaviour/run.js';
import { lostPoints } from '../tools/behaviour/select.js';
import type { Action, SessionDeps } from '../tools/behaviour/session.js';

// --- synthetic sessions and fakes (the same shapes as tests/behaviour.test.ts) -------------

let nextId = 0;

function pair(tool: string, input: Record<string, unknown>, output: string): Message[] {
  const id = `tu${++nextId}`;
  return [
    { role: 'assistant', text: '', toolUses: [{ tool_use_id: id, tool, input }] },
    { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: output, isError: false }] },
  ];
}

const chat = (role: 'user' | 'assistant', text: string): Message => ({ role, text, toolUses: [] });

const around = (before: number, value: string, after: number): string => `${'a'.repeat(before)}${value}${'b'.repeat(after)}`;

const VALUE = 'return computeInvoiceTotal(lineItems, taxRate);';

/** The edit target sits only in a big Read result that rule 1 cuts; a Read can bring it back. */
function reachableSession(): Message[] {
  const middle = pair('Read', { file_path: 'src/billing.ts' }, around(5000, VALUE, 5000));
  const filler = Math.max(8, 20 - 1 - middle.length);
  return [
    chat('user', 'please fix the invoice total'),
    ...middle,
    ...Array.from({ length: filler }, (_, i) => chat(i % 2 === 0 ? 'assistant' : 'user', `chat ${i}`)),
    ...pair('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'x' }, 'done'),
  ];
}

function toJsonl(messages: readonly Message[]): string {
  return messages
    .map((message, i) => {
      if (message.role === 'assistant') {
        const content: unknown[] = [];
        if (message.text !== '') content.push({ type: 'text', text: message.text });
        for (const use of message.toolUses) content.push({ type: 'tool_use', id: use.tool_use_id, name: use.tool, input: use.input });
        return JSON.stringify({ type: 'assistant', message: { id: `m${i}`, role: 'assistant', content } });
      }
      const content =
        message.toolResults === undefined
          ? message.text
          : message.toolResults.map((r) => ({ type: 'tool_result', tool_use_id: r.tool_use_id, is_error: r.isError === true, content: r.text }));
      return JSON.stringify({ type: 'user', message: { role: 'user', content } });
    })
    .join('\n');
}

const dirs: string[] = [];

function corpus(sessions: Record<string, Message[]>): string {
  const root = mkdtempSync(join(tmpdir(), 'judge-test-'));
  dirs.push(root);
  for (const [name, messages] of Object.entries(sessions)) {
    const [project, file] = name.split('/') as [string, string];
    mkdirSync(join(root, project), { recursive: true });
    writeFileSync(join(root, project, file), toJsonl(messages));
  }
  return root;
}

afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
});

const event = (value: unknown, at: number): StreamLine => ({ text: JSON.stringify(value), at });

interface StreamOptions {
  calls?: { tool: string; input: Record<string, unknown> }[];
  /** Tokens as `usage`; `null` leaves usage out. */
  usage?: { input_tokens: number; output_tokens: number } | null;
  text?: string;
}

/** A child's event stream: init, one assistant event per call, the result (110 tokens by default). */
function stream(options: StreamOptions = {}): StreamLine[] {
  const lines: StreamLine[] = [
    event({ type: 'system', subtype: 'init', mcp_servers: [{ name: 'stub', status: 'connected' }], tools: ['mcp__stub__Read', 'mcp__stub__Edit'] }, 1500),
  ];
  (options.calls ?? []).forEach((call, i) => {
    lines.push(
      event({ type: 'assistant', message: { content: [{ type: 'tool_use', id: `c${++nextId}`, name: `mcp__stub__${call.tool}`, input: call.input }] } }, 2000 + i * 1000),
    );
  });
  lines.push(
    event(
      {
        type: 'result',
        is_error: false,
        result: options.text ?? 'done',
        ...(options.usage === null ? {} : { usage: options.usage ?? { input_tokens: 100, output_tokens: 10 } }),
      },
      9000,
    ),
  );
  return lines;
}

const ok = (lines: StreamLine[]): ChildResult => ({ ok: true, lines, code: 0 });

function fakeRunner(answer: (request: ChildRequest, count: number) => ChildResult) {
  const requests: ChildRequest[] = [];
  const impl = vi.fn(async (request: ChildRequest) => {
    requests.push(request);
    return answer(request, requests.length);
  });
  return { runner: impl as ChildRunner, requests, impl };
}

const isJudgeCall = (request: ChildRequest): boolean => request.prompt.startsWith(JUDGE_PROMPT);
const judgeCalls = (requests: readonly ChildRequest[]) => requests.filter(isJudgeCall);

function fakeScratch(): Scratch & { cleanup: ReturnType<typeof vi.fn> } {
  return { cwd: '/scratch/work', mcpConfigFor: () => '/scratch/mcp.json', file: vi.fn((name: string) => `/scratch/${name}`), cleanup: vi.fn() };
}

function clock(): () => number {
  let t = 0;
  return () => (t += 1000);
}

const GUESS = 'a guess at the code';
const edit = (old: string) => ({ tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: old, new_string: 'n' } });

/** A model that edits from a guess in every arm (so both arms are `wrong`) and judges by `verdicts`. */
function guessing(verdicts: (count: number, request: ChildRequest) => string | ChildResult) {
  let judged = 0;
  return fakeRunner((request) => {
    if (!isJudgeCall(request)) return ok(stream({ calls: [edit(GUESS)] }));
    const verdict = verdicts(++judged, request);
    return typeof verdict === 'string' ? ok(stream({ text: verdict })) : verdict;
  });
}

const BASE = ['--model', 'test-model', '--max-points', '2', '--token-cap', '100000'];
const JUDGE = [...BASE, '--judge', '--judge-token-cap', '100000'];

const runWith = (argv: string[], runner: ChildRunner, extra: Parameters<typeof run>[1] = {}) =>
  run(argv, { runner, createScratch: fakeScratch, now: clock(), ...extra });

function deps(runner: ChildRunner): SessionDeps {
  return { runner, scratch: fakeScratch(), model: 'test-model', now: clock() };
}

const action = (command: string): Action => ({ tool: 'Bash', input: { command } });

/** `n` calibration pairs; labels alternate, and the stand-in judge below answers by what the action looks like. */
function pairs(n: number): CalibrationPair[] {
  return Array.from({ length: n }, (_, i) =>
    i % 2 === 0
      ? { recorded: action(`c${i}`), action: action(`c${i}`), label: 'equivalent' as const }
      : { recorded: action(`c${i}`), action: action(`x${i}`), label: 'different' as const },
  );
}

/** A judge that gets every calibration pair right, except the first `wrong` of them, and calls behaviour edits different. */
function calibrationJudge(wrong = 0) {
  let n = 0;
  return fakeRunner((request) => {
    if (!isJudgeCall(request)) return ok(stream({ calls: [edit(GUESS)] }));
    const right = request.prompt.includes('"command":"x') ? 'different' : request.prompt.includes('"command":"c') ? 'equivalent' : 'different';
    const flipped = n++ < wrong;
    return ok(stream({ text: flipped ? (right === 'different' ? 'equivalent' : 'different') : right }));
  });
}

// --- verdicts ------------------------------------------------------------------------------

describe('parseVerdict', () => {
  it('takes the first word, ignoring case and punctuation', () => {
    expect(parseVerdict('equivalent')).toBe('equivalent');
    expect(parseVerdict('  Equivalent.\n')).toBe('equivalent');
    expect(parseVerdict('DIFFERENT - it edits another file')).toBe('different');
    expect(parseVerdict('"different"')).toBe('different');
  });

  it('maps anything else to undecided', () => {
    for (const text of ['', '   ', 'maybe', 'not equivalent', 'undecided', 'I think they are equivalent']) {
      expect(parseVerdict(text)).toBe('undecided');
    }
  });
});

describe('judgeIfOpen', () => {
  const recorded = action('npm test');
  const other = action('npm run test');

  it('asks once with both actions and nothing else, and returns the verdict', async () => {
    const model = fakeRunner(() => ok(stream({ text: 'equivalent' })));
    const budget = new JudgeBudget(1000);
    expect(await judgeIfOpen(budget, recorded, other, deps(model.runner))).toBe('equivalent');
    expect(model.requests).toHaveLength(1);
    const prompt = model.requests[0]!.prompt;
    expect(prompt).toBe(`${JUDGE_PROMPT}\n\nRecorded step:\n${JSON.stringify(recorded)}\n\nSecond call:\n${JSON.stringify(other)}`);
    expect(budget.used).toBe(110);
  });

  it('answers undecided for a failed child, an empty reply and a malformed reply', async () => {
    const cases: ChildResult[] = [{ ok: false, reason: 'timeout' }, ok(stream({ text: '' })), ok(stream({ text: 'hard to say' }))];
    for (const result of cases) {
      expect(await judgeIfOpen(new JudgeBudget(1000), recorded, other, deps(fakeRunner(() => result).runner))).toBe('undecided');
    }
  });

  it('does not call the model once the budget has reached its cap', async () => {
    const model = fakeRunner(() => ok(stream({ text: 'different' })));
    const budget = new JudgeBudget(110);
    expect(await judgeIfOpen(budget, recorded, other, deps(model.runner))).toBe('different');
    expect(await judgeIfOpen(budget, recorded, other, deps(model.runner))).toBeUndefined();
    expect(model.impl).toHaveBeenCalledTimes(1);
  });

  it('stops after a call that reports no usage, since the cap can no longer be enforced', async () => {
    const model = fakeRunner(() => ok(stream({ text: 'equivalent', usage: null })));
    const budget = new JudgeBudget(1_000_000);
    expect(await judgeIfOpen(budget, recorded, other, deps(model.runner))).toBe('equivalent');
    expect(await judgeIfOpen(budget, recorded, other, deps(model.runner))).toBeUndefined();
  });
});

// --- calibration ---------------------------------------------------------------------------

describe('calibrate', () => {
  const grade = (set: CalibrationPair[], wrong = 0, cap = 1_000_000) =>
    calibrate(set, new JudgeBudget(cap), deps(calibrationJudge(wrong).runner));

  it('passes at 19 of 20 and fails at 17 of 20 or with fewer than 20 pairs', async () => {
    expect(await grade(pairs(20), 1)).toMatchObject({ correct: 19, total: 20, passed: true });
    expect(await grade(pairs(20), 3)).toMatchObject({ correct: 17, total: 20, passed: false });
    expect(await grade(pairs(19), 0)).toMatchObject({ correct: 19, total: 19, passed: false });
    expect(await grade(pairs(20), 0)).toMatchObject({ correct: 20, passed: true, bar: 0.9, minimum: 20 });
  });

  it('counts an undecided reply and a pair left unjudged by the cap as misses', async () => {
    const model = fakeRunner(() => ok(stream({ text: 'maybe' })));
    expect(await calibrate(pairs(20), new JudgeBudget(1_000_000), deps(model.runner))).toMatchObject({ correct: 0, total: 20, passed: false });
    const capped = calibrationJudge();
    expect(await calibrate(pairs(20), new JudgeBudget(220), deps(capped.runner))).toMatchObject({ correct: 2, total: 20, passed: false });
    expect(capped.impl).toHaveBeenCalledTimes(2);
  });
});

describe('calibration files', () => {
  const good = JSON.stringify([{ recorded: action('a'), action: action('b'), label: 'equivalent' }]);

  it('parses a labelled set', () => {
    expect(parseCalibrationSet(good)).toEqual([{ recorded: action('a'), action: action('b'), label: 'equivalent' }]);
  });

  it('rejects anything else with one fixed message that never echoes the content', () => {
    const secret = 'secret-file-content-xyz';
    const bad = [
      '{',
      '[]',
      JSON.stringify({ [secret]: 1 }),
      JSON.stringify([{ recorded: action(secret), action: action('b'), label: 'maybe' }]),
      JSON.stringify([{ recorded: { tool: secret }, action: action('b'), label: 'different' }]),
    ];
    for (const json of bad) {
      let message = '';
      try {
        parseCalibrationSet(json);
      } catch (error) {
        message = (error as Error).message;
      }
      expect(message).toBe(CALIBRATION_MALFORMED);
      expect(message).not.toContain(secret);
    }
  });

  it('reports an unreadable file with a fixed message', () => {
    expect(() => loadCalibrationSet(join(tmpdir(), 'no-such-judge-calibration-file.json'))).toThrow(CALIBRATION_UNREADABLE);
  });
});

// --- the judge inside a run -------------------------------------------------------------------

describe('run with --judge', () => {
  const root = () => corpus({ 'p/s.jsonl': reachableSession() });

  it('keeps spec 005 counts and adds a judge line per arm for wrong results', async () => {
    const model = guessing((n) => (n === 1 ? 'equivalent' : 'different'));
    const result = await runWith([...JUDGE, '--root', root()], model.runner);
    expect(result.code).toBe(0);
    expect(result.output).toContain('wrong       1/1 (100%)');
    const judgeLines = result.output.split('\n').filter((line) => line.includes('judge (wrong results'));
    expect(judgeLines).toEqual([
      '  judge (wrong results: 1): equivalent 1, different 0, undecided 0, unjudged 0',
      '  judge (wrong results: 1): equivalent 0, different 1, undecided 0, unjudged 0',
    ]);
    expect(judgeCalls(model.requests)).toHaveLength(2);
  });

  it('judges only wrong results', async () => {
    // Right with the value in view (the control arm), a guess without it (the compacted arm).
    let judged = 0;
    const model = fakeRunner((request) => {
      if (isJudgeCall(request)) {
        judged++;
        return ok(stream({ text: 'different' }));
      }
      return ok(stream({ calls: [edit(request.prompt.includes(VALUE) ? VALUE : GUESS)] }));
    });
    const result = await runWith([...JUDGE, '--root', root()], model.runner);
    expect(result.code).toBe(0);
    expect(judged).toBe(1);
    const lines = result.output.split('\n').filter((line) => line.includes('judge (wrong results'));
    expect(lines).toEqual([
      '  judge (wrong results: 0): equivalent 0, different 0, undecided 0, unjudged 0',
      '  judge (wrong results: 1): equivalent 0, different 1, undecided 0, unjudged 0',
    ]);
  });

  it('prints the same report as a run without the judge once the judge lines are removed', async () => {
    const plain = await runWith([...BASE, '--root', root()], guessing(() => 'equivalent').runner);
    const judgedRun = guessing(() => 'equivalent');
    const judged = await runWith([...JUDGE, '--root', root()], judgedRun.runner);
    const withoutJudge = judged.output
      .split('\n')
      .filter((line) => !/^( {2}judge \(|Judge|Calibration)/.test(line))
      .join('\n');
    expect(withoutJudge).toBe(plain.output);
  });

  it('makes no judge call and prints no judge text when the judge is off', async () => {
    const model = guessing(() => 'equivalent');
    const result = await runWith([...BASE, '--root', root()], model.runner);
    expect(judgeCalls(model.requests)).toHaveLength(0);
    expect(result.output).not.toMatch(/judge \(|Judge|Calibration/);
  });

  it('shows the not-validated notice without calibration, and clears it only after a pass', async () => {
    const plain = await runWith([...JUDGE, '--root', root()], calibrationJudge().runner);
    expect(plain.output).toContain('Judge: not validated (calibration did not pass in this invocation)');
    expect(plain.output).not.toContain('Calibration:');

    const readCalibration = () => pairs(20);
    const passed = await runWith([...JUDGE, '--calibrate', 'set.json', '--root', root()], calibrationJudge().runner, { readCalibration });
    expect(passed.code).toBe(0);
    expect(passed.output).toContain('Calibration: 20/20 correct (100%), bar 90% over at least 20 pairs: passed');
    expect(passed.output).not.toContain('not validated');

    const failed = await runWith([...JUDGE, '--calibrate', 'set.json', '--root', root()], calibrationJudge(5).runner, { readCalibration });
    expect(failed.output).toContain('Calibration: 15/20 correct (75%), bar 90% over at least 20 pairs: not passed');
    expect(failed.output).toContain('Judge: not validated');
  });

  it('stops judging at the judge cap, reports the rest as unjudged, and still exits 0', async () => {
    const model = guessing(() => 'equivalent');
    const result = await runWith([...BASE, '--judge', '--judge-token-cap', '1', '--root', root()], model.runner);
    expect(result.code).toBe(0);
    expect(judgeCalls(model.requests)).toHaveLength(1);
    expect(result.output).toContain('  judge (wrong results: 1): equivalent 1, different 0, undecided 0, unjudged 0');
    expect(result.output).toContain('  judge (wrong results: 1): equivalent 0, different 0, undecided 0, unjudged 1');
    expect(result.output).toContain('Judge tokens: 110 used, cap 1;');
  });

  it('counts calibration tokens against the same cap', async () => {
    const model = calibrationJudge();
    const result = await runWith([...BASE, '--judge', '--judge-token-cap', '220', '--calibrate', 'set.json', '--root', root()], model.runner, {
      readCalibration: () => pairs(20),
    });
    expect(result.code).toBe(0);
    expect(judgeCalls(model.requests)).toHaveLength(2);
    expect(result.output).toContain('Judge tokens: 220 used, cap 220;');
    expect(result.output).toContain('unjudged 1');
  });

  it('keeps transcript text out of the report and the argument lists', async () => {
    const secretValue = 'return decryptVault("hunter2-super-secret-token");';
    const secretNew = 'secret-new-text-xyz';
    const middle = pair('Read', { file_path: 'src/secret-path-xyz.ts' }, around(5000, secretValue, 5000));
    const messages: Message[] = [
      chat('user', 'go'),
      ...middle,
      ...Array.from({ length: 17 }, (_, i) => chat(i % 2 === 0 ? 'assistant' : 'user', `chat ${i}`)),
      ...pair('Edit', { file_path: 'src/secret-path-xyz.ts', old_string: secretValue, new_string: 'x' }, 'done'),
    ];
    const model = fakeRunner((request) =>
      isJudgeCall(request)
        ? ok(stream({ text: 'equivalent' }))
        : ok(stream({ calls: [{ tool: 'Edit', input: { file_path: 'src/secret-path-xyz.ts', old_string: 'a guess', new_string: secretNew } }] })),
    );
    const result = await runWith([...JUDGE, '--calibrate', 'set.json', '--root', corpus({ 'p/s.jsonl': messages })], model.runner, {
      readCalibration: () => pairs(20),
    });
    expect(result.code).toBe(0);
    expect(judgeCalls(model.requests).length).toBeGreaterThan(20);
    for (const secret of ['hunter2', 'secret-path-xyz', secretNew, 'a guess', 'c0', 'x1']) {
      expect(result.output).not.toContain(secret);
      expect(JSON.stringify(model.requests.map((r) => r.args))).not.toContain(secret);
    }
  });
});

describe('run --calibrate on its own', () => {
  const ARGS = ['--calibrate', 'set.json', '--model', 'test-model', '--judge-token-cap', '100000'];

  it('prints the result and the spend, reads no corpus and runs no behaviour point', async () => {
    const reader = vi.fn(lostPoints);
    const model = calibrationJudge(1);
    const result = await runWith(ARGS, model.runner, { readCalibration: () => pairs(20), lostPoints: reader });
    expect(result.code).toBe(0);
    expect(result.output).toBe(
      [
        'Judge calibration',
        'Model: test-model',
        'Calibration: 19/20 correct (95%), bar 90% over at least 20 pairs: passed',
        'Judge tokens: 2200 used, cap 100000',
      ].join('\n'),
    );
    expect(reader).not.toHaveBeenCalled();
    expect(model.requests.every(isJudgeCall)).toBe(true);
  });

  it('says so when the set is too small', async () => {
    const result = await runWith(ARGS, calibrationJudge().runner, { readCalibration: () => pairs(19) });
    expect(result.output).toContain('Calibration: 19/19 correct (100%), bar 90% over at least 20 pairs: only 19 pairs, at least 20 needed: not passed');
  });

  it('exits 1 on a malformed file before any child process starts', async () => {
    const model = calibrationJudge();
    const scratch = vi.fn(fakeScratch);
    const result = await runWith(ARGS, model.runner, {
      readCalibration: () => {
        throw new Error(CALIBRATION_MALFORMED);
      },
      createScratch: scratch,
    });
    expect(result).toEqual({ code: 1, output: CALIBRATION_MALFORMED });
    expect(model.impl).not.toHaveBeenCalled();
    expect(scratch).not.toHaveBeenCalled();
  });
});

describe('judge options', () => {
  it('refuse --judge or --calibrate without a judge cap or model, before any corpus read or child process', async () => {
    const model = calibrationJudge();
    const reader = vi.fn(lostPoints);
    const scratch = vi.fn(fakeScratch);
    const extra = { lostPoints: reader, createScratch: scratch, readCalibration: () => pairs(20) };
    const noCap = await runWith([...BASE, '--judge'], model.runner, extra);
    expect(noCap).toEqual({ code: 1, output: 'Missing required input: --judge-token-cap <n>' });
    const noModel = await runWith(['--calibrate', 'set.json', '--judge-token-cap', '5'], model.runner, extra);
    expect(noModel).toEqual({ code: 1, output: 'Missing required input: --model <id>' });
    const noFile = await runWith([...BASE, '--judge', '--judge-token-cap', '5', '--calibrate'], model.runner, extra);
    expect(noFile).toEqual({ code: 1, output: 'Missing required input: --calibrate <file>' });
    for (const bad of ['0', '-3', 'many']) {
      const result = await runWith([...BASE, '--judge', '--judge-token-cap', bad], model.runner, extra);
      expect(result.code).toBe(1);
    }
    expect(model.impl).not.toHaveBeenCalled();
    expect(reader).not.toHaveBeenCalled();
    expect(scratch).not.toHaveBeenCalled();
  });
});

describe('documentation', () => {
  it('quotes the judge prompt verbatim in the README (FR-011)', () => {
    const readme = readFileSync(join(__dirname, '..', 'README.md'), 'utf8').replace(/\r\n/g, '\n');
    expect(readme).toContain(`> ${JUDGE_PROMPT}`);
  });
});

describe('a tool-less child', () => {
  it('runs to its result although its init event lists no stub server', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'judge-spawn-'));
    dirs.push(dir);
    const script = join(dir, 'fake-claude.mjs');
    const init = JSON.stringify({ type: 'system', subtype: 'init', mcp_servers: [], tools: [] });
    const result = JSON.stringify({ type: 'result', is_error: false, result: 'equivalent', usage: { input_tokens: 1, output_tokens: 1 } });
    writeFileSync(script, `console.log(${JSON.stringify(init)});\nsetTimeout(() => console.log(${JSON.stringify(result)}), 300);\n`);
    const child = await spawnChild(Date.now, process.execPath)({ args: [script], prompt: 'x', cwd: dir, timeoutMs: 60_000 });
    expect(child).toMatchObject({ ok: true, code: 0 });
    expect(child.ok && child.lines).toHaveLength(2);
  }, 40_000);
});
