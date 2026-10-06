import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  jevAsker,
  register,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyDecisions, backendMarker, collectToolCalls, decideCall, type Message } from '../src/index.js';
import type { PluginOptions } from 'claude-code';

type EventHandler = (
  $: Record<string, unknown>,
  event: Record<string, unknown>,
  next: (event: unknown) => unknown,
) => unknown;

/** Captures `register`'s `on(...)` calls so a handler can be invoked directly in a test. */
function registerHandlers(options: PluginOptions): Map<string, EventHandler> {
  const handlers = new Map<string, EventHandler>();
  const on = ((pattern: string, hook: EventHandler) => {
    handlers.set(pattern, hook);
  }) as unknown as Parameters<typeof register>[0];
  register(on, options);
  return handlers;
}

function fakeEngine(overrides: Record<string, unknown> = {}) {
  const logs: string[] = [];
  const toasts: string[] = [];
  return {
    logs,
    toasts,
    $: {
      ui: { log: (text: string) => logs.push(text), toast: (text: string) => toasts.push(text) },
      env: { get: async () => undefined },
      settings: { read: async () => ({}) },
      http: { fetch: async () => ({ status: 500, ok: false, text: 'unused' }) },
      ...overrides,
    },
  };
}

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    call('tool-2', 'Bash', { command: 'npm test' }, 'FAIL'),
    result('tool-2', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'Fixing now.', { handle: 'h-5' }),
    message('user', 'go ahead', { handle: 'h-6' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      mode: 'backend',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'jev-latest',
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      mode: 'backend',
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
    });
  });

  it('resolves a configured baseUrl onto the config', () => {
    expect(resolveHookConfig({ baseUrl: 'http://127.0.0.1:8000/v1/systemone' })).toEqual({
      mode: 'backend',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      baseUrl: 'http://127.0.0.1:8000/v1/systemone',
    });
  });

  it('rejects an invalid baseUrl onto the config instead of silently defaulting', () => {
    expect(resolveHookConfig({ baseUrl: 'localhost:8000/v1/systemone' })).toEqual({
      mode: 'backend',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      invalidBaseUrl: 'localhost:8000/v1/systemone',
    });
  });
});

describe('compaction mode config', () => {
  it('maps unset, empty, whitespace and backend to the backend mode', () => {
    for (const value of [undefined, '', '   ', 'backend', ' backend ']) {
      const config = resolveHookConfig(value === undefined ? {} : { compactionMode: value });
      expect(config.mode).toBe('backend');
      expect(config.invalidMode).toBeUndefined();
    }
  });

  it('maps rules, trimmed, to the rules mode', () => {
    for (const value of ['rules', ' rules ']) {
      const config = resolveHookConfig({ compactionMode: value });
      expect(config.mode).toBe('rules');
      expect(config.invalidMode).toBeUndefined();
    }
  });

  it('keeps any other value as invalid instead of selecting a mode', () => {
    for (const value of ['rulez', 'Rules', 'rules,backend']) {
      expect(resolveHookConfig({ compactionMode: value })).toMatchObject({
        mode: 'backend',
        invalidMode: value,
      });
    }
    expect(resolveHookConfig({ compactionMode: ' rulez ' }).invalidMode).toBe('rulez');
    expect(resolveHookConfig({ compactionMode: 3 }).invalidMode).toBeUndefined();
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    messages[1]!.toolUses[0]!.text = 'x'.repeat(2000);
    messages[2]!.toolResults![0]!.text = 'x'.repeat(2000);
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out).toHaveLength(messages.length);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]?.handle).toBeUndefined();
    const head150Tail150 = new RegExp(
      `^x{150}\\n\\[fast-systemone-compaction truncated 1700 chars[^\\]]*\\]\\nx{150}$`,
    );
    expect(out[1]?.toolUses[0]?.text).toMatch(head150Tail150);
    expect(out[2]?.handle).toBeUndefined();
    expect(out[2]?.toolResults?.[0]?.text).toMatch(head150Tail150);
    expect(out[2]?.toolResults?.[0]).toMatchObject({ tool_use_id: 'tool-1', isError: false });
    expect(out[3]).toBe(messages[3]);
    expect(out[4]).toBe(messages[4]);
  });

  it('preserves short dropped-result messages and their handles', () => {
    const messages = transcript();
    messages[1]!.toolUses[0]!.text = 'y'.repeat(100);
    messages[2]!.toolResults![0]!.text = 'y'.repeat(100);
    const calls = collectToolCalls(messages, 0);
    const decisions = [
      decideCall(calls[0]!, { keepCall: 0.9, keepResult: 0.1 }, { keepThreshold: 0.5 }),
      decideCall(calls[1]!, { keepCall: 0.9, keepResult: 0.9 }, { keepThreshold: 0.5 }),
    ];
    const out = toSessionMessages(messages, applyDecisions(messages, decisions, calls, 300));
    expect(out[1]).toBe(messages[1]);
    expect(out[2]).toBe(messages[2]);
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'call_t2' || name === 'result_t2' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(1);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['drop_call', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'r-tool-2', 'h-5', 'h-6']);
    expect(summarize(output)).toMatch(/^\d+% reduction; 1 kept, 1 call_dropped; state ~\d+ tokens \(full\) in 1 request\(s\)$/);
    expect(decisionLog(output)).toBe('t1:Read:drop_call/call=0.10/result=0.10 t2:Bash:keep/call=0.90/result=0.90');
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.1));
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([
      'decisions (1/2): t1:Read:drop_call/call=0.10/result=0.10',
      'decisions (2/2): t2:Bash:drop_call/call=0.10/result=0.10',
    ]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });

  it('sends requests to the configured baseUrl and model instead of TypeSafe', async () => {
    const urls: string[] = [];
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1, baseUrl: 'http://127.0.0.1:8000/v1/systemone', model: 'typed-decisions' }),
      apiKey: 'local',
    };
    await compactSession(transcript(), config, async (url, init) => {
      urls.push(url);
      const asker = jevFetch(() => 0.9);
      return asker(url, init);
    });
    expect(urls).toEqual(['http://127.0.0.1:8000/v1/systemone']);
  });

  it('pins the default TypeSafe URL when baseUrl is unset (regression guard)', async () => {
    const urls: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    await compactSession(transcript(), config, async (url, init) => {
      urls.push(url);
      const asker = jevFetch(() => 0.9);
      return asker(url, init);
    });
    expect(urls).toEqual(['https://api.typesafe.ai/v1/systemone']);
  });

  it('still falls back on a non-2xx response when baseUrl is configured', async () => {
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1, baseUrl: 'http://127.0.0.1:8000/v1/systemone' }),
      apiKey: 'local',
    };
    await expect(
      compactSession(transcript(), config, async () => ({ status: 503, ok: false, text: 'down' })),
    ).rejects.toThrow(/503/);
  });

  it('rejects an invalid baseUrl before making any request', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1, baseUrl: 'localhost:8000/v1/systemone', apiKey: 'k' });
    await expect(
      compactSession(transcript(), config, () => {
        throw new Error('fetch must not be called for an invalid baseUrl');
      }),
    ).rejects.toThrow(/invalid.*localhost:8000\/v1\/systemone/i);
  });
});

function noFetch(): never {
  throw new Error('fetch must not be called');
}

/** An old 10,000-character result, then two plain messages that stay protected. */
function rulesTranscript(): SessionMessage[] {
  const big = 'x'.repeat(10_000);
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Bash', { command: 'npm test' }, big),
    result('tool-1', big),
    message('assistant', 'Fixing now.', { handle: 'h-3' }),
    message('user', 'go ahead', { handle: 'h-4' }),
  ];
}

describe('compactSession in rules mode', () => {
  it('shortens old results without a key and without calling fetch', async () => {
    const config = resolveHookConfig({ compactionMode: 'rules', preserveRecentMessages: 2 });
    const { result: output, messages } = await compactSession(rulesTranscript(), config, noFetch);

    expect(output.stats.requests).toBe(0);
    expect(output.stats.resultsDropped).toBe(1);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, undefined, 'h-3', 'h-4']);
    expect(messages[2]?.toolResults?.[0]?.text).toContain('truncated 9700 chars');
  });

  it('ignores an invalid baseUrl', async () => {
    const config = resolveHookConfig({
      compactionMode: 'rules',
      baseUrl: 'localhost:8000/v1/systemone',
      preserveRecentMessages: 2,
    });
    expect(config.invalidBaseUrl).toBeDefined();
    const { result: output } = await compactSession(rulesTranscript(), config, noFetch);
    expect(output.stats.resultsDropped).toBe(1);
  });

  it('rejects an invalid mode before any request, key check or baseUrl check', async () => {
    const config = resolveHookConfig({
      compactionMode: 'rulez',
      baseUrl: 'localhost:8000/v1/systemone',
      apiKey: 'k',
    });
    await expect(compactSession(rulesTranscript(), config, noFetch)).rejects.toThrow(
      'invalid compactionMode: rulez (use backend or rules)',
    );
  });

  it('still uses the backend when the mode is unset', async () => {
    const urls: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    await compactSession(transcript(), config, async (url, init) => {
      urls.push(url);
      return jevFetch(() => 0.9)(url, init);
    });
    expect(urls).toEqual(['https://api.typesafe.ai/v1/systemone']);
  });
});

describe('register', () => {
  it('logs the invalid baseUrl once, at the first handled event', async () => {
    const handlers = registerHandlers({ baseUrl: 'localhost:8000/v1/systemone', apiKey: 'k' });
    const { $, logs } = fakeEngine({ session: { usage: async () => ({ context: { percent: 10 } }) } });
    const next = (event: unknown) => event;

    await handlers.get('turn.complete')!($, {}, next);
    await handlers.get('turn.complete')!($, {}, next);

    expect(logs.filter((line) => line.includes('invalid baseUrl'))).toHaveLength(1);
    expect(logs.some((line) => line.includes('localhost:8000/v1/systemone'))).toBe(true);
  });

  it('tags the fallback outcome with the raw invalid baseUrl instead of a host', async () => {
    const handlers = registerHandlers({ baseUrl: 'localhost:8000/v1/systemone', apiKey: 'k' });
    const { $, toasts } = fakeEngine();
    const next = (event: unknown) => event;

    await handlers.get('session.compact')!($, { messages: transcript() }, next);

    expect(toasts[0]).toContain('localhost:8000/v1/systemone');
  });

  it('tags a successful outcome with the configured backend host and model', async () => {
    const handlers = registerHandlers({
      baseUrl: 'http://127.0.0.1:8000/v1/systemone',
      model: 'typed-decisions',
      apiKey: 'local',
      preserveRecentMessages: 1,
    });
    const { $, toasts } = fakeEngine({
      http: { fetch: jevFetch(() => 0.9) },
    });
    const next = (event: unknown) => event;

    await handlers.get('session.compact')!($, { messages: transcript() }, next);

    expect(toasts[0]).toContain(backendMarker('http://127.0.0.1:8000/v1/systemone', 'typed-decisions'));
  });

  it('tags the default-endpoint outcome with the TypeSafe host', async () => {
    const handlers = registerHandlers({ apiKey: 'k', preserveRecentMessages: 1 });
    const { $, toasts } = fakeEngine({ http: { fetch: jevFetch(() => 0) } });
    const next = (event: unknown) => event;

    await handlers.get('session.compact')!($, { messages: transcript() }, next);

    expect(toasts[0]).toContain('api.typesafe.ai');
  });

  it('tags the below-minimum-reduction fallback outcome with the backend marker', async () => {
    const baseUrl = 'http://127.0.0.1:8000/v1/systemone';
    const handlers = registerHandlers({
      baseUrl,
      model: 'typed-decisions',
      apiKey: 'local',
      preserveRecentMessages: 1,
      minReductionRatio: 0.99,
    });
    const { $, toasts } = fakeEngine({ http: { fetch: jevFetch(() => 0.9) } });
    const next = (event: unknown) => event;

    await handlers.get('session.compact')!($, { messages: transcript() }, next);

    expect(toasts[0]).toContain('below');
    expect(toasts[0]).toContain(backendMarker(baseUrl, 'typed-decisions'));
  });

  it('tags the backend-error fallback outcome with the backend marker', async () => {
    const baseUrl = 'http://127.0.0.1:8000/v1/systemone';
    const handlers = registerHandlers({ baseUrl, model: 'typed-decisions', apiKey: 'local', preserveRecentMessages: 1 });
    const { $, toasts } = fakeEngine({
      http: { fetch: async () => ({ status: 503, ok: false, text: 'down' }) },
    });
    const next = (event: unknown) => event;

    await handlers.get('session.compact')!($, { messages: transcript() }, next);

    expect(toasts[0]).toContain('503');
    expect(toasts[0]).toContain(backendMarker(baseUrl, 'typed-decisions'));
  });
});

describe('register in rules mode', () => {
  const next = (event: unknown) => event;
  const throwing = {
    env: { get: async () => noFetch() },
    settings: { read: async () => noFetch() },
    http: { fetch: async () => noFetch() },
  };

  it('applies the rules, reports counts and marker, and never looks up a key', async () => {
    const handlers = registerHandlers({ compactionMode: 'rules', preserveRecentMessages: 2 });
    const { $, toasts } = fakeEngine(throwing);

    const out = (await handlers.get('session.compact')!($, { messages: rulesTranscript() }, next)) as {
      messages: unknown[];
    };

    expect(out.messages).toHaveLength(5);
    expect(toasts).toHaveLength(1);
    expect(toasts[0]).toMatch(
      /^kept 5\/5 messages, no summary \(\d+% reduction; 1 results shortened, 0 reads removed\) \[rules\]$/,
    );
  });

  it('falls back below the minimum and states achieved and required ratios', async () => {
    const handlers = registerHandlers({
      compactionMode: 'rules',
      preserveRecentMessages: 2,
      minReductionRatio: 0.99,
    });
    const { $, toasts } = fakeEngine(throwing);
    let fellBack = false;

    await handlers.get('session.compact')!($, { messages: rulesTranscript() }, () => {
      fellBack = true;
    });

    expect(fellBack).toBe(true);
    expect(toasts[0]).toMatch(
      /^fallback to built-in summary \(below 99% minimum: \d+% reduction; 1 results shortened, 0 reads removed\) \[rules\]$/,
    );
  });

  it('falls back on a transcript the rules cannot reduce', async () => {
    const handlers = registerHandlers({ compactionMode: 'rules' });
    const { $, toasts } = fakeEngine(throwing);
    let fellBack = false;

    await handlers.get('session.compact')!($, { messages: transcript() }, () => {
      fellBack = true;
    });

    expect(fellBack).toBe(true);
    expect(toasts[0]).toContain('below 25% minimum');
    expect(toasts[0]).toContain('[rules]');
  });

  it('does not log an invalid baseUrl in rules mode', async () => {
    const handlers = registerHandlers({
      compactionMode: 'rules',
      baseUrl: 'localhost:8000/v1/systemone',
    });
    const { $, logs } = fakeEngine({ ...throwing, session: { usage: async () => ({ context: { percent: 10 } }) } });

    await handlers.get('turn.complete')!($, {}, next);
    await handlers.get('session.compact')!($, { messages: rulesTranscript() }, next);

    expect(logs.filter((line) => line.includes('invalid baseUrl'))).toHaveLength(0);
  });

  it('falls back with the invalid-value message and makes no request', async () => {
    const handlers = registerHandlers({ compactionMode: 'rulez', apiKey: 'k' });
    const { $, toasts } = fakeEngine(throwing);
    let fellBack = false;

    await handlers.get('session.compact')!($, { messages: rulesTranscript() }, () => {
      fellBack = true;
    });

    expect(fellBack).toBe(true);
    expect(toasts[0]).toContain('invalid compactionMode: rulez (use backend or rules)');
  });

  it('gives lines from both modes a reduction and a bracketed marker', async () => {
    const backend = registerHandlers({ apiKey: 'k', preserveRecentMessages: 2 });
    const rules = registerHandlers({ compactionMode: 'rules', preserveRecentMessages: 2 });
    const backendEngine = fakeEngine({ http: { fetch: jevFetch(() => 0) } });
    const rulesEngine = fakeEngine(throwing);

    await backend.get('session.compact')!(backendEngine.$, { messages: rulesTranscript() }, next);
    await rules.get('session.compact')!(rulesEngine.$, { messages: rulesTranscript() }, next);

    for (const line of [backendEngine.toasts[0]!, rulesEngine.toasts[0]!]) {
      expect(line).toMatch(/\d+% reduction/);
      expect(line).toMatch(/\[[^\]]+\]$/);
    }
    expect(rulesEngine.toasts[0]).toMatch(/\d+ results shortened, \d+ reads removed/);
  });
});
