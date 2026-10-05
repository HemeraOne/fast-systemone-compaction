import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { compactedHistory, controlHistory, flattenHistory, pointContext, summaryHistory, summaryPrompt } from './history.js';
import { runCheck } from './check.js';
import { createScratch, spawnChild } from './model.js';
import type { ChildRunner, Scratch } from './model.js';
import { formatReport, summarize } from './report.js';
import type { PointRecord } from './report.js';
import { lostPoints, sample } from './select.js';
import type { LostPoint } from './select.js';
import { runPoint, runText } from './session.js';
import type { Arm, Outcome, SessionDeps } from './session.js';

export interface RunDeps {
  runner: ChildRunner;
  createScratch: () => Scratch;
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

/** The summary arm: one tool-less call summarises the control history, then the point runs on it. */
async function summaryOutcome(point: LostPoint, context: Parameters<typeof runPoint>[1], session: SessionDeps): Promise<Outcome> {
  const started = session.now();
  const summary = await runText(summaryPrompt(controlHistory(point.messages, point.messageIndex)), session);
  if (summary.reason !== undefined) {
    return {
      class: 'failed',
      lookups: 0,
      seconds: (session.now() - started) / 1000,
      tokens: summary.tokens,
      reason: summary.reason,
      ...(summary.unmetered ? { unmetered: true } : {}),
    };
  }
  const outcome = await runPoint(point, context, summaryHistory(summary.text), session);
  return {
    ...outcome,
    seconds: (session.now() - started) / 1000,
    tokens: outcome.tokens + summary.tokens,
    ...(outcome.unmetered || summary.unmetered ? { unmetered: true } : {}),
  };
}

/**
 * The whole run, with its effects injected. Nothing is read and no child process is started
 * until every required input is present; the only files written are in one scratch directory
 * that is removed at the end.
 */
export async function run(argv: readonly string[], overrides: Partial<RunDeps> = {}): Promise<RunResult> {
  const now = overrides.now ?? Date.now;
  const deps: RunDeps = {
    runner: spawnChild(now),
    createScratch,
    now,
    lostPoints,
    exists: existsSync,
    ...overrides,
  };
  const [model, maxPointsText, tokenCapText, rootArg] = VALUE_FLAGS.map((flag) => flagValue(argv, flag));

  if (argv.includes('--check')) {
    if (model === undefined) return { code: 1, output: 'Missing required input: --model <id>' };
    const scratch = deps.createScratch();
    try {
      return await runCheck({ runner: deps.runner, scratch, model, now: deps.now });
    } finally {
      scratch.cleanup();
    }
  }

  const missing = [
    ...(model === undefined ? ['--model <id>'] : []),
    ...(maxPointsText === undefined ? ['--max-points <n>'] : []),
    ...(tokenCapText === undefined ? ['--token-cap <n>'] : []),
  ];
  if (missing.length > 0) return { code: 1, output: `Missing required input: ${missing.join(', ')}` };
  const maxPoints = positiveInt(maxPointsText);
  const tokenCap = positiveInt(tokenCapText);
  if (maxPoints === undefined || tokenCap === undefined) {
    return { code: 1, output: '--max-points and --token-cap each need a positive whole number' };
  }

  const root = rootArg ?? join(homedir(), '.claude', 'projects');
  if (!deps.exists(root)) return { code: 1, output: `Corpus root not found: ${root}` };
  const { points, sessions, skipped } = deps.lostPoints(root);
  if (points.length === 0) {
    const note = skipped > 0 ? ` (${skipped} skipped as parallel calls)` : '';
    return { code: 1, output: `No lost point found in the corpus${note}; no child process was started` };
  }

  const scratch = deps.createScratch();
  try {
    const session: SessionDeps = { runner: deps.runner, scratch, model: model!, now: deps.now };
    const records: PointRecord[] = [];
    const withSummary = argv.includes('--summary');
    let tokens = 0;
    let stopped: string | undefined;

    for (const point of sample(points, maxPoints)) {
      if (tokens >= tokenCap) {
        stopped = 'token cap reached';
        break;
      }
      const context = pointContext(point);
      const k = point.messageIndex;
      const control = await runPoint(point, context, flattenHistory(controlHistory(point.messages, k)), session);
      const compacted = await runPoint(point, context, flattenHistory(compactedHistory(point.messages, k)), session);
      const outcomes: Partial<Record<Arm, Outcome>> = { control, compacted };
      if (withSummary) outcomes.summary = await summaryOutcome(point, context, session);
      const done = Object.values(outcomes);
      tokens += done.reduce((sum, outcome) => sum + outcome.tokens, 0);
      records.push({
        session: point.session,
        messageIndex: k,
        tool: point.tool,
        kinds: point.losses.map((loss) => loss.kind),
        rules: point.losses.map((loss) => loss.rule),
        outcomes,
      });
      if (done.every((outcome) => outcome.class === 'failed')) {
        stopped = 'every arm failed to run';
        break;
      }
      if (done.some((outcome) => outcome.unmetered)) {
        stopped = 'usage not reported, so the cap cannot be enforced';
        break;
      }
    }

    const report = summarize(records, {
      model: model!,
      sessions,
      available: points.length,
      skipped,
      tried: records.length,
      tokens,
      tokenCap,
      stopped,
      summaryRan: withSummary,
    });
    return { code: 0, output: formatReport(report) };
  } finally {
    scratch.cleanup();
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  run(process.argv.slice(2)).then((result) => {
    (result.code === 0 ? process.stdout : process.stderr).write(`${result.output}\n`);
    process.exitCode = result.code;
  });
}
