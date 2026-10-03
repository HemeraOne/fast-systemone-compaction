import { describe, expect, it } from 'vitest';
import type { Message } from '../src/index.js';
import { formatReport, summarize } from '../tools/replay/report.js';
import { candidatesOf, replaySession } from '../tools/replay/replay.js';
import type { ExcludedReason, Loss, PointResult, Rule, SessionResult, ValueKind } from '../tools/replay/replay.js';
import { parseTranscript } from '../tools/replay/transcript.js';

const line = (value: unknown): string => JSON.stringify(value);

describe('parseTranscript', () => {
  it('maps user and assistant text, tool uses, and tool results', () => {
    const text = [
      line({ type: 'user', message: { role: 'user', content: 'fix the parser' } }),
      line({
        type: 'assistant',
        message: {
          id: 'm1',
          role: 'assistant',
          content: [
            { type: 'text', text: 'looking' },
            { type: 'tool_use', id: 'tu1', name: 'Read', input: { file_path: 'src/a.ts' } },
          ],
        },
      }),
      line({
        type: 'user',
        message: {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: 'tu1', content: 'file body' }],
        },
      }),
    ].join('\n');

    const { messages, malformedLines } = parseTranscript(text);
    expect(malformedLines).toBe(0);
    expect(messages).toEqual([
      { role: 'user', text: 'fix the parser', toolUses: [] },
      {
        role: 'assistant',
        text: 'looking',
        toolUses: [{ tool_use_id: 'tu1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
      },
      {
        role: 'user',
        text: '',
        toolUses: [],
        toolResults: [{ tool_use_id: 'tu1', text: 'file body', isError: false }],
      },
    ]);
  });

  it('reads result text from an array of text blocks and the error flag', () => {
    const text = line({
      type: 'user',
      message: {
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: 'tu2',
            is_error: true,
            content: [
              { type: 'text', text: 'first ' },
              { type: 'image', source: {} },
              { type: 'text', text: 'second' },
            ],
          },
        ],
      },
    });
    expect(parseTranscript(text).messages[0]?.toolResults).toEqual([
      { tool_use_id: 'tu2', text: 'first second', isError: true },
    ]);
  });

  it('merges consecutive assistant entries that share a message id', () => {
    const text = [
      line({
        type: 'assistant',
        message: { id: 'm1', role: 'assistant', content: [{ type: 'text', text: 'a' }] },
      }),
      line({
        type: 'assistant',
        message: {
          id: 'm1',
          role: 'assistant',
          content: [{ type: 'tool_use', id: 'tu1', name: 'Bash', input: { command: 'ls' } }],
        },
      }),
      line({
        type: 'assistant',
        message: { id: 'm2', role: 'assistant', content: [{ type: 'text', text: 'b' }] },
      }),
    ].join('\n');

    const { messages } = parseTranscript(text);
    expect(messages).toHaveLength(2);
    expect(messages[0]).toEqual({
      role: 'assistant',
      text: 'a',
      toolUses: [{ tool_use_id: 'tu1', tool: 'Bash', input: { command: 'ls' } }],
    });
    expect(messages[1]?.text).toBe('b');
  });

  it('ignores sidechain entries and non-message entries', () => {
    const text = [
      line({ type: 'summary', summary: 'x' }),
      line({ type: 'user', isSidechain: true, message: { role: 'user', content: 'agent talk' } }),
      line({ type: 'user', message: { role: 'user', content: 'real' } }),
    ].join('\n');
    const { messages, malformedLines } = parseTranscript(text);
    expect(messages.map((m) => m.text)).toEqual(['real']);
    expect(malformedLines).toBe(0);
  });

  it('counts malformed lines and skips them', () => {
    const text = ['{not json', line({ type: 'user', message: { role: 'user', content: 'ok' } }), ''].join('\n');
    const { messages, malformedLines } = parseTranscript(text);
    expect(messages).toHaveLength(1);
    expect(malformedLines).toBe(1);
  });

  it('returns no messages for empty text', () => {
    expect(parseTranscript('')).toEqual({ messages: [], malformedLines: 0 });
  });
});

// --- synthetic session helpers -------------------------------------------------

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

/** `before` filler characters, then the value, then `after` filler characters. */
function around(before: number, value: string, after: number): string {
  return `${'a'.repeat(before)}${value}${'b'.repeat(after)}`;
}

const VALUE = 'return computeInvoiceTotal(lineItems, taxRate);';

/**
 * A session whose last real action is `final`: a goal message, the `middle` messages, enough
 * plain chat to clear the minimum history and the protected recent messages, then the
 * call. Returns the messages and the index of the message holding the final call.
 */
function session(middle: Message[], final: { tool: string; input: Record<string, unknown> }) {
  const filler = Math.max(8, 20 - 1 - middle.length);
  const messages: Message[] = [
    chat('user', 'please fix the invoice total'),
    ...middle,
    ...Array.from({ length: filler }, (_, i) => chat(i % 2 === 0 ? 'assistant' : 'user', `chat ${i}`)),
  ];
  const finalIndex = messages.length;
  messages.push(...pair(final.tool, final.input, 'done'));
  return { messages, finalIndex };
}

function editing(oldString: string) {
  return { tool: 'Edit', input: { file_path: 'src/billing.ts', old_string: oldString, new_string: 'x' } };
}

/** The point result for the call at `index`. */
function pointAt(result: ReturnType<typeof replaySession>, index: number) {
  const point = result.points.find((p) => p.messageIndex === index);
  if (!point) throw new Error(`no point at ${index}`);
  return point;
}

describe('candidatesOf', () => {
  it('takes paths, commands, and edit targets of at least 12 characters', () => {
    expect(candidatesOf('Read', { file_path: 'src/billing.ts' })).toEqual([
      { kind: 'path', value: 'src/billing.ts', length: 14 },
    ]);
    expect(candidatesOf('Write', { file_path: 'src/billing.ts', content: 'brand new file body' })).toEqual([
      { kind: 'path', value: 'src/billing.ts', length: 14 },
    ]);
    expect(candidatesOf('Bash', { command: 'npm run typecheck' })).toEqual([
      { kind: 'command', value: 'npm run typecheck', length: 17 },
    ]);
    expect(candidatesOf('Edit', { file_path: 'src/billing.ts', old_string: VALUE, new_string: 'y' })).toEqual([
      { kind: 'path', value: 'src/billing.ts', length: 14 },
      { kind: 'editTarget', value: VALUE, length: VALUE.length },
    ]);
  });

  it('takes every old_string of a MultiEdit', () => {
    const first = 'const first = computeFirst();';
    const second = 'const second = computeSecond();';
    expect(
      candidatesOf('MultiEdit', {
        file_path: 'src/billing.ts',
        edits: [
          { old_string: first, new_string: 'a' },
          { old_string: second, new_string: 'b' },
        ],
      }).filter((c) => c.kind === 'editTarget'),
    ).toEqual([
      { kind: 'editTarget', value: first, length: first.length },
      { kind: 'editTarget', value: second, length: second.length },
    ]);
  });

  it('ignores other tools, other fields, and short values', () => {
    expect(candidatesOf('Grep', { pattern: 'computeInvoiceTotal' })).toEqual([]);
    expect(candidatesOf('Write', { content: 'brand new file body' })).toEqual([]);
    expect(candidatesOf('Bash', { command: 'ls' })).toEqual([]);
    expect(candidatesOf('Read', { file_path: 'a.ts' })).toEqual([]);
    expect(candidatesOf('Edit', { file_path: 'src/billing.ts', old_string: 'short' })).toEqual([
      { kind: 'path', value: 'src/billing.ts', length: 14 },
    ]);
  });
});

describe('replaySession verdicts', () => {
  it('attributes a value cut out of a large result to rule 1', () => {
    const { messages, finalIndex } = session(
      pair('Bash', { command: 'cat build.log' }, around(5000, VALUE, 5000)),
      editing(VALUE),
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.status).toBe('checked');
    expect(point.needed).toBe(1);
    expect(point.losses).toEqual([
      {
        kind: 'editTarget',
        rule: 'rule1',
        length: VALUE.length,
        locator: { session: 's', messageIndex: finalIndex, tool: 'Edit' },
      },
    ]);
  });

  it('attributes a value that lived only in a superseded read to rule 2', () => {
    const { messages, finalIndex } = session(
      [
        ...pair('Read', { file_path: 'src/legacy.ts' }, around(2500, VALUE, 2500)),
        ...pair('Write', { file_path: 'src/legacy.ts', content: 'rewritten' }, 'ok'),
      ],
      { tool: 'Edit', input: { file_path: 'src/elsewhere.ts', old_string: VALUE, new_string: 'y' } },
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.needed).toBe(1);
    expect(point.losses.map((l) => l.rule)).toEqual(['rule2']);
  });

  it('attributes a value whose occurrences were hit by both rules to both', () => {
    const { messages, finalIndex } = session(
      [
        ...pair('Bash', { command: 'cat build.log' }, around(5000, VALUE, 5000)),
        ...pair('Read', { file_path: 'src/legacy.ts' }, around(2500, VALUE, 2500)),
        ...pair('Write', { file_path: 'src/legacy.ts', content: 'rewritten' }, 'ok'),
      ],
      editing(VALUE),
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.losses.map((l) => l.rule)).toEqual(['both']);
  });

  const bigLog = (): Message[] => pair('Bash', { command: 'cat build.log' }, around(5000, VALUE, 5000));

  it.each([
    ['a user message', () => [chat('user', `use this: ${VALUE}`), ...bigLog()]],
    ['a short kept result', () => [...pair('Bash', { command: 'cat short.log' }, `ok ${VALUE}`), ...bigLog()]],
    ['the head of a truncated result', () => pair('Bash', { command: 'cat build.log' }, around(20, VALUE, 9000))],
    ['the tail of a truncated result', () => pair('Bash', { command: 'cat build.log' }, around(9000, VALUE, 20))],
  ])('does not count a value that survives in %s', (_name, middle) => {
    const { messages, finalIndex } = session(middle(), editing(VALUE));
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.status).toBe('checked');
    expect(point.needed).toBe(1);
    expect(point.losses).toEqual([]);
  });

  it('does not count a value that never appeared before the point', () => {
    const { messages, finalIndex } = session(
      pair('Bash', { command: 'cat build.log' }, around(5000, 'something else entirely', 5000)),
      editing(VALUE),
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.needed).toBe(0);
    expect(point.losses).toEqual([]);
  });

  it('counts a value cut in the middle by truncation as lost', () => {
    const long = `function process() {\n${'  doStep();\n'.repeat(25)}}`;
    const { messages, finalIndex } = session(
      pair('Bash', { command: 'cat build.log' }, around(100, long, 9000)),
      editing(long),
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.losses.map((l) => l.rule)).toEqual(['rule1']);
  });
});

describe('replaySession points', () => {
  it('counts a call in a short history as excluded', () => {
    const early = pair('Bash', { command: 'npm run typecheck' }, 'ok');
    const filler = Array.from({ length: 30 }, (_, i) => chat(i % 2 ? 'user' : 'assistant', `c${i}`));
    const point = pointAt(replaySession('s', [chat('user', 'go'), ...early, ...filler]), 1);
    expect(point.status).toBe('excluded');
    expect(point.excludedReason).toBe('history-too-short');
    expect(point.losses).toEqual([]);
  });

  it('counts a point where the rules would reduce too little as excluded', () => {
    const { messages, finalIndex } = session(
      pair('Bash', { command: 'cat small.log' }, `fine ${VALUE}`),
      editing(VALUE),
    );
    const point = pointAt(replaySession('s', messages), finalIndex);
    expect(point.status).toBe('excluded');
    expect(point.excludedReason).toBe('below-minimum-reduction');
  });

  it('takes at most 10 evenly spaced points per session, the same ones every run', () => {
    const middle = Array.from({ length: 25 }, (_, i) => pair('Bash', { command: `echo step number ${i}` }, 'ok')).flat();
    const messages: Message[] = [chat('user', 'go'), ...middle, chat('assistant', 'end')];
    const first = replaySession('s', messages);
    const taken = first.points.filter((p) => p.excludedReason !== 'history-too-short');
    // Calls sit at odd indexes 1, 3, ..., 49; those from 21 on have a long enough history.
    const eligible = Array.from({ length: 15 }, (_, i) => 21 + 2 * i);
    expect(taken.map((p) => p.messageIndex)).toEqual(
      Array.from({ length: 10 }, (_, i) => eligible[Math.floor((i * eligible.length) / 10)]),
    );
    expect(first.points.filter((p) => p.excludedReason === 'history-too-short')).toHaveLength(10);
    expect(replaySession('s', messages)).toEqual(first);
  });

  it('yields no points for a session without tool calls', () => {
    const messages = Array.from({ length: 30 }, (_, i) => chat(i % 2 ? 'assistant' : 'user', `c${i}`));
    expect(replaySession('s', messages).points).toEqual([]);
  });
});

// --- report -------------------------------------------------------------------

const noKinds = { path: 0, command: 0, editTarget: 0 };

function loss(session: string, messageIndex: number, tool: string, kind: ValueKind, rule: Rule, length: number): Loss {
  return { kind, rule, length, locator: { session, messageIndex, tool } };
}

function checked(messageIndex: number, tool: string, neededByKind: typeof noKinds, losses: Loss[]): PointResult {
  const needed = neededByKind.path + neededByKind.command + neededByKind.editTarget;
  return { messageIndex, toolUseId: `t${messageIndex}`, tool, status: 'checked', needed, neededByKind, losses };
}

function excludedPoint(messageIndex: number, excludedReason: ExcludedReason): PointResult {
  return { messageIndex, toolUseId: `t${messageIndex}`, tool: 'Bash', status: 'excluded', excludedReason, needed: 0, neededByKind: noKinds, losses: [] };
}

// Three points checked, needing 2 + 1 + 1 = 4 values; 2 of them lost (one to rule 1, one to both),
// each at its own point.
const results: SessionResult[] = [
  {
    session: 'b-session.jsonl',
    messages: 60,
    malformedLines: 1,
    skipped: false,
    points: [
      checked(30, 'Edit', { path: 1, command: 0, editTarget: 1 }, [loss('b-session.jsonl', 30, 'Edit', 'editTarget', 'rule1', 80)]),
      checked(40, 'Bash', { path: 0, command: 1, editTarget: 0 }, []),
      excludedPoint(5, 'history-too-short'),
      excludedPoint(50, 'below-minimum-reduction'),
    ],
  },
  {
    session: 'a-session.jsonl',
    messages: 45,
    malformedLines: 0,
    skipped: false,
    points: [checked(25, 'Edit', { path: 0, command: 0, editTarget: 1 }, [loss('a-session.jsonl', 25, 'Edit', 'editTarget', 'both', 120)])],
  },
  { session: 'c-empty.jsonl', messages: 0, malformedLines: 3, skipped: true, points: [] },
];

describe('summarize and formatReport', () => {
  it('counts sessions, points, needed and lost values', () => {
    const summary = summarize(results);
    expect(summary).toMatchObject({
      sessionsRead: 2,
      sessionsSkipped: 1,
      malformedLines: 4,
      pointsChecked: 3,
      excludedTooShort: 1,
      excludedLowReduction: 1,
      neededValues: 4,
      lostValues: 2,
      pointsWithLoss: 2,
    });
    expect(summary.byRule).toEqual({ rule1: 1, rule2: 0, both: 1 });
  });

  it('prints the headline and the per-rule table', () => {
    const text = formatReport(summarize(results));
    expect(text).toContain('Sessions read: 2   skipped: 1   malformed lines: 4');
    expect(text).toContain('Replay points checked: 3   excluded: 2 (history too short: 1, below minimum reduction: 1)');
    expect(text).toContain('Needed values: 4   lost: 2   loss rate: 50.0%   points with a loss: 2');
    expect(text).toMatch(/rule 1\s+1\s+25\.0%/);
    expect(text).toMatch(/rule 2\s+0\s+0\.0%/);
    expect(text).toMatch(/both\s+1\s+25\.0%/);
    expect(text).toContain('upper bound on harm');
  });

  it('prints n/a instead of a rate when there are no needed values', () => {
    const text = formatReport(summarize([]));
    expect(text).toContain('Needed values: 0   lost: 0   loss rate: n/a   points with a loss: 0');
    expect(text).toMatch(/rule 1\s+0\s+n\/a/);
  });

  it('gives byte-identical output for the same input, whatever the input order', () => {
    const forward = formatReport(summarize(results));
    expect(formatReport(summarize(results))).toBe(forward);
    expect(formatReport(summarize([...results].reverse()))).toBe(forward);
  });

  it('breaks needed and lost values down by kind', () => {
    const text = formatReport(summarize(results));
    expect(text).toMatch(/path\s+1\s+0\s+0\.0%/);
    expect(text).toMatch(/command\s+1\s+0\s+0\.0%/);
    expect(text).toMatch(/editTarget\s+2\s+2\s+100\.0%/);
  });

  it('lists each loss with a locator, sorted by session and position, and no text', () => {
    const text = formatReport(summarize(results));
    expect(text).toContain('Losses (2)');
    const lines = text.split('\n').filter((l) => /^ {2}\S+\.jsonl #/.test(l));
    expect(lines).toEqual([
      '  a-session.jsonl #25 Edit editTarget 120 chars both',
      '  b-session.jsonl #30 Edit editTarget 80 chars rule 1',
    ]);
  });

  it('never prints a path, command, or edit target from the session', () => {
    const marked = 'return MARKER_EDIT_TARGET_computeInvoiceTotal(items);';
    const { messages, finalIndex } = session(
      pair('Bash', { command: 'cat MARKER_COMMAND_build.log' }, around(5000, marked, 5000)),
      { tool: 'Edit', input: { file_path: 'src/MARKER_PATH_billing.ts', old_string: marked, new_string: 'x' } },
    );
    const result = replaySession('plain-name.jsonl', messages);
    expect(pointAt(result, finalIndex).losses).toHaveLength(1);
    const text = formatReport(summarize([result]));
    expect(text).toContain('plain-name.jsonl');
    expect(text).not.toContain('MARKER');
  });
});
