import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  compactedHistory,
  controlHistory,
  pointContext,
  summaryHistory,
  summaryRequestMessages,
  toApiMessages,
} from './history.js';
import { callModel } from './model.js';
import { formatReport, summarize } from './report.js';
import type { ApiMessage, PointContext } from './history.js';
import type { PointRecord } from './report.js';
import { lostPoints, sample } from './select.js';
import { runPoint } from './session.js';
import type { Arm, Outcome, SessionDeps } from './session.js';

export interface RunDeps {
  fetch: typeof fetch;
  now: () => number;
  lostPoints: typeof lostPoints;
  exists: (path: string) => boolean;
}

export interface RunResult {
  code: number;
  /** The report on success, the reason on failure. */
  output: string;
}

const VALUE_FLAGS = ['--model', '--max-points', '--token-cap', '--root'] as const;

function flagValue(args: readonly string[], flag: string): string | undefined {
  const value = args[args.indexOf(flag) + 1];
  return args.includes(flag) && value !== undefined && !value.startsWith('--') ? value : undefined;
}

const positiveInt = (value: string | undefined): number | undefined =>
  value !== undefined && /^\d+$/.test(value) && Number(value) > 0 ? Number(value) : undefined;

/** The summary arm: one extra call summarises the control history, then the point runs on it. */
async function summaryOutcome(
  context: PointContext,
  control: readonly ApiMessage[],
  session: SessionDeps,
): Promise<Outcome> {
  const started = session.now();
  const reply = await callModel(
    { apiKey: session.apiKey, model: session.model, messages: summaryRequestMessages(control) },
    session.fetch,
  );
  const summary = reply.ok
    ? reply.content.map((block) => (block['type'] === 'text' ? String(block['text']) : '')).join('')
    : '';
  if (!reply.ok || summary === '') {
    return { class: 'failed', lookups: 0, seconds: (session.now() - started) / 1000, tokens: reply.ok ? reply.inputTokens + reply.outputTokens : 0 };
  }
  const outcome = await runPoint(context, summaryHistory(summary), session);
  return {
    ...outcome,
    seconds: (session.now() - started) / 1000,
    tokens: outcome.tokens + reply.inputTokens + reply.outputTokens,
  };
}

/**
 * The whole run, with its effects injected. Nothing is read or sent until every required
 * input is present, and nothing is ever written.
 */
export async function run(
  argv: readonly string[],
  env: Readonly<Record<string, string | undefined>>,
  overrides: Partial<RunDeps> = {},
): Promise<RunResult> {
  const deps: RunDeps = {
    fetch: globalThis.fetch,
    now: Date.now,
    lostPoints,
    exists: existsSync,
    ...overrides,
  };
  const [model, maxPointsText, tokenCapText, rootArg] = VALUE_FLAGS.map((flag) => flagValue(argv, flag));
  const apiKey = env['ANTHROPIC_API_KEY'];

  const missing = [
    ...(model === undefined ? ['--model <id>'] : []),
    ...(maxPointsText === undefined ? ['--max-points <n>'] : []),
    ...(tokenCapText === undefined ? ['--token-cap <n>'] : []),
    ...(apiKey === undefined || apiKey === '' ? ['ANTHROPIC_API_KEY'] : []),
  ];
  if (missing.length > 0) return { code: 1, output: `Missing required input: ${missing.join(', ')}` };
  const maxPoints = positiveInt(maxPointsText);
  const tokenCap = positiveInt(tokenCapText);
  if (maxPoints === undefined || tokenCap === undefined) {
    return { code: 1, output: '--max-points and --token-cap each need a positive whole number' };
  }

  const root = rootArg ?? join(homedir(), '.claude', 'projects');
  if (!deps.exists(root)) return { code: 1, output: `Corpus root not found: ${root}` };
  const { points, sessions } = deps.lostPoints(root);
  if (points.length === 0) return { code: 1, output: 'No lost point found in the corpus; nothing was sent' };

  const session: SessionDeps = { fetch: deps.fetch, apiKey: apiKey!, model: model!, now: deps.now };
  const records: PointRecord[] = [];
  let tokens = 0;
  let stoppedEarly = false;
  const withSummary = argv.includes('--summary');

  for (const point of sample(points, maxPoints)) {
    if (tokens >= tokenCap) {
      stoppedEarly = true;
      break;
    }
    const context = pointContext(point);
    const k = point.messageIndex;
    const control = await runPoint(context, toApiMessages(controlHistory(point.messages, k)), session);
    const compacted = await runPoint(context, toApiMessages(compactedHistory(point.messages, k)), session);
    const outcomes: Partial<Record<Arm, Outcome>> = { control, compacted };
    if (withSummary) outcomes.summary = await summaryOutcome(context, toApiMessages(controlHistory(point.messages, k)), session);
    tokens += Object.values(outcomes).reduce((sum, outcome) => sum + outcome.tokens, 0);
    records.push({
      session: point.session,
      messageIndex: k,
      tool: point.tool,
      kinds: point.losses.map((loss) => loss.kind),
      rules: point.losses.map((loss) => loss.rule),
      outcomes,
    });
  }

  const report = summarize(records, {
    model: model!,
    sessions,
    available: points.length,
    tried: records.length,
    tokens,
    tokenCap,
    stoppedEarly,
    summaryRan: withSummary,
  });
  return { code: 0, output: formatReport(report) };
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  run(process.argv.slice(2), process.env).then((result) => {
    (result.code === 0 ? process.stdout : process.stderr).write(`${result.output}\n`);
    process.exitCode = result.code;
  });
}
