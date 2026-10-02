import { describe, expect, it } from 'vitest';
import { compactByRules, reductionRatio, type Message } from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

/** An assistant call and its user result, as the host records them. */
function pair(
  id: string,
  tool: string,
  input: Record<string, unknown>,
  text: string,
  isError = false,
): Message[] {
  return [
    message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] }),
    message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] }),
  ];
}

/** First message, the given middle messages, then `tail` plain messages that stay protected. */
function session(middle: Message[], tail = 2): Message[] {
  return [
    message('user', 'Fix the failing test.'),
    ...middle,
    ...Array.from({ length: tail }, (_, index) => message('assistant', `recent ${index}`)),
  ];
}

const OPTIONS = { preserveRecentMessages: 2 };

function resultText(messages: readonly Message[], id: string): string | undefined {
  for (const m of messages) {
    for (const r of m.toolResults ?? []) if (r.tool_use_id === id) return r.text;
  }
  return undefined;
}

function hasCall(messages: readonly Message[], id: string): boolean {
  return messages.some((m) => m.toolUses.some((t) => t.tool_use_id === id));
}

describe('rule 1: large old results', () => {
  it('leaves a result of exactly 2000 characters alone and shortens 2001', () => {
    const middle = [
      ...pair('exact', 'Bash', { command: 'a' }, 'x'.repeat(2000)),
      ...pair('over', 'Bash', { command: 'b' }, 'y'.repeat(2001)),
    ];
    const input = session(middle);
    const out = compactByRules(input, OPTIONS);

    expect(out.messages[2]).toBe(input[2]);
    expect(resultText(out.messages, 'exact')).toBe('x'.repeat(2000));
    expect(resultText(out.messages, 'over')).toBe(
      `${'y'.repeat(300)}\n[fast-jev-compaction truncated 1701 chars of this tool result; re-run the tool if needed]`,
    );
    expect(out.stats.resultsDropped).toBe(1);
    expect(out.stats.callsDropped).toBe(0);
  });

  it('keeps the call when it shortens the result', () => {
    const input = session(pair('big', 'Read', { file_path: 'a.ts' }, 'z'.repeat(10_000)));
    const out = compactByRules(input, OPTIONS);

    expect(hasCall(out.messages, 'big')).toBe(true);
    expect(resultText(out.messages, 'big')).toContain('truncated 9700 chars');
    expect(out.messages).toHaveLength(input.length);
  });

  it('does not touch results in the first message or the recent messages', () => {
    const big = 'q'.repeat(10_000);
    const first = message('assistant', 'start', {
      toolUses: [{ tool_use_id: 'first', tool: 'Bash', input: {}, text: big }],
    });
    const firstResult = message('user', '', {
      toolResults: [{ tool_use_id: 'first', text: big }],
    });
    const recent = pair('recent', 'Bash', { command: 'c' }, big);
    const input = [first, firstResult, message('assistant', 'middle'), ...recent];
    const out = compactByRules(input, OPTIONS);

    expect(out.messages).toHaveLength(input.length);
    out.messages.forEach((m, index) => expect(m).toBe(input[index]));
    expect(out.stats.resultsDropped).toBe(0);
    expect(out.stats.pinned).toBe(2);
  });

  it('raises the threshold to head + 120 for a large truncateHeadChars', () => {
    const middle = [
      ...pair('inside', 'Bash', { command: 'a' }, 'x'.repeat(2050)),
      ...pair('outside', 'Bash', { command: 'b' }, 'y'.repeat(2071)),
    ];
    const out = compactByRules(session(middle), { ...OPTIONS, truncateHeadChars: 1950 });

    expect(resultText(out.messages, 'inside')).toBe('x'.repeat(2050));
    expect(resultText(out.messages, 'outside')).toBe(
      `${'y'.repeat(1950)}\n[fast-jev-compaction truncated 121 chars of this tool result; re-run the tool if needed]`,
    );
    expect(out.stats.resultsDropped).toBe(1);
  });

  it('keeps only the note when truncateHeadChars is 0', () => {
    const out = compactByRules(session(pair('big', 'Bash', {}, 'x'.repeat(5000))), {
      ...OPTIONS,
      truncateHeadChars: 0,
    });
    expect(resultText(out.messages, 'big')).toBe(
      '[fast-jev-compaction truncated 5000 chars of this tool result; re-run the tool if needed]',
    );
  });

  it('shortens an error result and says so in the note', () => {
    const out = compactByRules(session(pair('err', 'Bash', {}, 'e'.repeat(3000), true)), OPTIONS);
    expect(resultText(out.messages, 'err')).toContain('truncated 2700 chars of this tool result (error)');
  });

  it('reports no reduction for a transcript without tool calls', () => {
    const input = session([message('assistant', 'thinking'), message('user', 'ok')]);
    const out = compactByRules(input, OPTIONS);

    expect(reductionRatio(out)).toBe(0);
    expect(out.stats.calls).toBe(0);
    out.messages.forEach((m, index) => expect(m).toBe(input[index]));
  });

  it('makes no request and labels the stage', () => {
    const out = compactByRules(session(pair('big', 'Bash', {}, 'x'.repeat(5000))), OPTIONS);
    expect(out.stats.requests).toBe(0);
    expect(out.stats.stateTokens).toBe(0);
    expect(out.stats.stateStage).toBe('rules');
    expect(reductionRatio(out)).toBeGreaterThan(0.5);
  });

  it('gives identical output on repeated runs', () => {
    const input = session([
      ...pair('a', 'Bash', { command: 'a' }, 'x'.repeat(4000)),
      ...pair('b', 'Read', { file_path: 'f.ts' }, 'y'.repeat(2500)),
    ]);
    const first = compactByRules(input, OPTIONS);
    const second = compactByRules(input, OPTIONS);

    expect(second.messages).toEqual(first.messages);
    expect(second.decisions).toEqual(first.decisions);
    expect({ ...second.stats, ms: 0 }).toEqual({ ...first.stats, ms: 0 });
  });
});

function read(
  id: string,
  path: unknown,
  text = 'file contents\n',
  extra: Record<string, unknown> = {},
  isError = false,
): Message[] {
  return pair(id, 'Read', path === undefined ? { ...extra } : { file_path: path, ...extra }, text, isError);
}

function edit(id: string, path: string, isError = false): Message[] {
  return pair(id, 'Edit', { file_path: path, old_string: 'a', new_string: 'b' }, 'ok', isError);
}

function write(id: string, path: string): Message[] {
  return pair(id, 'Write', { file_path: path, content: 'new' }, 'ok');
}

function removed(out: ReturnType<typeof compactByRules>, id: string): boolean {
  return !hasCall(out.messages, id) && resultText(out.messages, id) === undefined;
}

describe('rule 2: superseded reads', () => {
  it('removes a full read when a later read, edit or write of the same path exists', () => {
    const input = session([
      ...read('r1', 'src/a.ts'),
      ...read('r2', 'src/a.ts'),
      ...read('e1', 'src/b.ts'),
      ...edit('e2', 'src/b.ts'),
      ...read('w1', 'src/c.ts'),
      ...write('w2', 'src/c.ts'),
    ]);
    const out = compactByRules(input, OPTIONS);

    expect(removed(out, 'r1')).toBe(true);
    expect(removed(out, 'e1')).toBe(true);
    expect(removed(out, 'w1')).toBe(true);
    for (const id of ['r2', 'e2', 'w2']) expect(hasCall(out.messages, id)).toBe(true);
    expect(out.stats.callsDropped).toBe(3);
    expect(out.stats.resultsDropped).toBe(0);
  });

  it('keeps the latest read of a path and the same message objects', () => {
    const input = session(read('r1', 'src/a.ts'));
    const out = compactByRules(input, OPTIONS);

    expect(out.stats.callsDropped).toBe(0);
    out.messages.forEach((m, index) => expect(m).toBe(input[index]));
  });

  it('matches whole normalized paths only', () => {
    const input = session([
      ...read('a', 'src/a.ts'),
      ...edit('a2', 'src/a.tsx'),
      ...read('b', 'src/b'),
      ...read('b2', 'src/bc'),
      ...read('c', 'src\\c.ts'),
      ...read('c2', 'src/c.ts'),
    ]);
    const out = compactByRules(input, OPTIONS);

    expect(hasCall(out.messages, 'a')).toBe(true);
    expect(hasCall(out.messages, 'b')).toBe(true);
    expect(removed(out, 'c')).toBe(true);
    expect(out.stats.callsDropped).toBe(1);
  });

  it('keeps reads whose path is missing or not a string', () => {
    const input = session([
      ...read('n1', undefined),
      ...read('n2', undefined),
      ...read('x1', 42),
      ...read('x2', 42),
    ]);
    const out = compactByRules(input, OPTIONS);

    expect(out.stats.callsDropped).toBe(0);
    for (const id of ['n1', 'n2', 'x1', 'x2']) expect(hasCall(out.messages, id)).toBe(true);
  });

  it('keeps a partial read unless a later full read of the path exists', () => {
    for (const range of [{ offset: 10 }, { limit: 5 }, { offset: 10, limit: 5 }]) {
      const followedByPartial = compactByRules(
        session([...read('p', 'f.ts', 'part', range), ...read('q', 'f.ts', 'other', { offset: 99 })]),
        OPTIONS,
      );
      const followedByEdit = compactByRules(
        session([...read('p', 'f.ts', 'part', range), ...edit('e', 'f.ts')]),
        OPTIONS,
      );
      const followedByWrite = compactByRules(
        session([...read('p', 'f.ts', 'part', range), ...write('w', 'f.ts')]),
        OPTIONS,
      );
      const followedByFull = compactByRules(
        session([...read('p', 'f.ts', 'part', range), ...read('f', 'f.ts', 'whole')]),
        OPTIONS,
      );

      expect(hasCall(followedByPartial.messages, 'p')).toBe(true);
      expect(hasCall(followedByEdit.messages, 'p')).toBe(true);
      expect(hasCall(followedByWrite.messages, 'p')).toBe(true);
      expect(removed(followedByFull, 'p')).toBe(true);
    }
  });

  it('removes a full read when only a later partial read follows', () => {
    const out = compactByRules(
      session([...read('full', 'f.ts'), ...read('part', 'f.ts', 'part', { offset: 3 })]),
      OPTIONS,
    );
    expect(removed(out, 'full')).toBe(true);
    expect(hasCall(out.messages, 'part')).toBe(true);
  });

  it('lets a failed later call supersede nothing', () => {
    const failedRead = compactByRules(
      session([...read('r1', 'f.ts'), ...read('r2', 'f.ts', 'File does not exist.', {}, true)]),
      OPTIONS,
    );
    const failedEdit = compactByRules(
      session([...read('r1', 'f.ts'), ...edit('e', 'f.ts', true)]),
      OPTIONS,
    );
    const failedFull = compactByRules(
      session([...read('p', 'f.ts', 'part', { offset: 2 }), ...read('r2', 'f.ts', 'nope', {}, true)]),
      OPTIONS,
    );

    expect(hasCall(failedRead.messages, 'r1')).toBe(true);
    expect(hasCall(failedEdit.messages, 'r1')).toBe(true);
    expect(hasCall(failedFull.messages, 'p')).toBe(true);
  });

  it('removes a failed earlier read once a successful read of the path exists', () => {
    const out = compactByRules(
      session([...read('bad', 'f.ts', 'File does not exist.', {}, true), ...read('good', 'f.ts')]),
      OPTIONS,
    );
    expect(removed(out, 'bad')).toBe(true);
    expect(hasCall(out.messages, 'good')).toBe(true);
  });

  it('counts a superseding read that sits in the protected messages', () => {
    const input = [message('user', 'start'), ...read('old', 'f.ts'), ...read('new', 'f.ts')];
    const out = compactByRules(input, OPTIONS);

    expect(removed(out, 'old')).toBe(true);
    expect(hasCall(out.messages, 'new')).toBe(true);
  });

  it('never touches protected reads', () => {
    const wholeTail = [message('user', 'start'), ...read('a', 'f.ts'), ...read('b', 'f.ts')];
    const keptTail = compactByRules(wholeTail, { preserveRecentMessages: 4 });
    keptTail.messages.forEach((m, index) => expect(m).toBe(wholeTail[index]));

    const firstMessage = [
      message('assistant', 'start', {
        toolUses: [{ tool_use_id: 'first', tool: 'Read', input: { file_path: 'f.ts' }, text: 'x' }],
      }),
      message('user', '', { toolResults: [{ tool_use_id: 'first', text: 'x' }] }),
      ...read('later', 'f.ts'),
      message('assistant', 'recent 0'),
      message('assistant', 'recent 1'),
    ];
    const out = compactByRules(firstMessage, OPTIONS);
    expect(out.messages[0]).toBe(firstMessage[0]);
    expect(out.messages[1]).toBe(firstMessage[1]);
  });

  it('removes a read that is both superseded and large without counting it as shortened', () => {
    const out = compactByRules(
      session([...read('big', 'f.ts', 'z'.repeat(9000)), ...read('next', 'f.ts')]),
      OPTIONS,
    );
    expect(removed(out, 'big')).toBe(true);
    expect(out.stats.callsDropped).toBe(1);
    expect(out.stats.resultsDropped).toBe(0);
  });

  it('treats only Read, Edit and Write as touches, and only Read as removable', () => {
    const input = session([
      ...pair('g', 'Grep', { file_path: 'f.ts', pattern: 'a' }, 'hit'),
      ...read('r', 'f.ts'),
      ...pair('o', 'Glob', { file_path: 'f.ts' }, 'f.ts'),
      ...edit('e', 'f.ts'),
      ...edit('e2', 'f.ts'),
    ]);
    const out = compactByRules(input, OPTIONS);

    expect(hasCall(out.messages, 'g')).toBe(true);
    expect(hasCall(out.messages, 'o')).toBe(true);
    expect(hasCall(out.messages, 'e')).toBe(true);
    expect(removed(out, 'r')).toBe(true);
  });

  it('never leaves a result without its call and reports what it removed', () => {
    const input = session([
      ...read('r1', 'a.ts'),
      ...read('r2', 'a.ts'),
      ...read('p1', 'b.ts', 'part', { offset: 1, limit: 10 }),
      ...read('r3', 'b.ts'),
      ...pair('big', 'Bash', { command: 'x' }, 'y'.repeat(5000)),
    ]);
    const out = compactByRules(input, OPTIONS);

    const callIds = new Set(out.messages.flatMap((m) => m.toolUses.map((t) => t.tool_use_id)));
    for (const m of out.messages) {
      for (const r of m.toolResults ?? []) expect(callIds.has(r.tool_use_id)).toBe(true);
    }
    expect(out.stats.callsDropped).toBe(2);
    expect(out.stats.resultsDropped).toBe(1);
    expect(out.decisions.filter((d) => d.action === 'drop_call').map((d) => d.keepCall)).toEqual([0, 0]);
  });

  it('removes nothing when the real field names appear but no path repeats', () => {
    const input = session([
      ...read('a', '/repo/src/a.ts', 'a', { offset: 1, limit: 100 }),
      ...read('b', '/repo/src/b.ts'),
      ...read('c', 'C:\\repo\\src\\c.ts'),
    ]);
    const out = compactByRules(input, OPTIONS);
    expect(out.stats.callsDropped).toBe(0);
  });
});

describe('performance', () => {
  it('compacts a 10,000-message transcript in well under a second', () => {
    const middle: Message[] = [];
    for (let index = 0; index < 4999; index++) {
      middle.push(
        ...(index % 2 === 0
          ? read(`r${index}`, `src/file${index % 50}.ts`)
          : pair(`b${index}`, 'Bash', { command: 'c' }, 'o'.repeat(2500))),
      );
    }
    const input = session(middle);
    expect(input.length).toBeGreaterThanOrEqual(9998);

    const started = Date.now();
    const out = compactByRules(input, OPTIONS);
    expect(Date.now() - started).toBeLessThan(1000);
    expect(out.stats.callsDropped).toBeGreaterThan(0);
    expect(out.stats.resultsDropped).toBeGreaterThan(0);
  });
});
