import { familyOf, matchesRecorded } from './history.js';
import type { PointContext } from './history.js';
import { buildArgs, CHILD_TIMEOUT_MS, parseStream, stubFailed } from './model.js';
import type { ChildRunner, Scratch } from './model.js';
import type { LostPoint } from './select.js';
import { MAX_LOOKUPS } from './stub.js';

export type Arm = 'control' | 'compacted' | 'summary';

/** A tool call: the final action of an arm, or the recorded step it is compared with. */
export interface Action {
  tool: string;
  input: Record<string, unknown>;
}

export type JudgeVerdict = 'equivalent' | 'different' | 'undecided';

export type OutcomeClass = 'same' | 'recovered' | 'wrong' | 'gave-up' | 'unreachable' | 'failed';

export interface Outcome {
  class: OutcomeClass;
  /** Lookups issued before the terminal action, answerable or not. */
  lookups: number;
  /** Tool names of the lookups the stub could not answer (not a repeat of recorded history); absent when none. */
  unanswered?: string[];
  seconds: number;
  /** Tokens the child reported (input, output, cache). */
  tokens: number;
  /** A `gave-up` after lookups of which the stub answered none: the harness, not the model, left it stuck. */
  harnessLimited?: boolean;
  /** Why a `failed` or `gave-up` outcome ended that way: a fixed, non-sensitive phrase. */
  reason?: string;
  /** The child reported no usage, so the cap cannot be enforced past this point. */
  unmetered?: boolean;
  /** The call that ended a `wrong` outcome. Kept in memory for the judge, never printed. */
  action?: Action;
  /** The judge's verdict on a `wrong` outcome; absent when the judge is off or the cap stopped it. */
  verdict?: JudgeVerdict;
}

export interface SessionDeps {
  runner: ChildRunner;
  scratch: Scratch;
  model: string;
  now: () => number;
}

/**
 * Lets the model take its next step from `prompt` and classifies it against the recorded
 * step. The first call in the recorded step's tool family is the terminal action; every
 * other call is a lookup the stub answers from recorded results. Nothing is written
 * anywhere: a proposed edit is only compared.
 */
export async function runPoint(
  point: LostPoint,
  context: PointContext,
  prompt: string,
  deps: SessionDeps,
): Promise<Outcome> {
  const started = deps.now();
  const seconds = (at: number): number => (at - started) / 1000;
  const child = await deps.runner({
    args: buildArgs({ model: deps.model, mcpConfig: deps.scratch.mcpConfigFor(point) }),
    prompt,
    cwd: deps.scratch.cwd,
    timeoutMs: CHILD_TIMEOUT_MS,
  });
  if (!child.ok) {
    return { class: 'failed', lookups: 0, seconds: seconds(deps.now()), tokens: 0, reason: child.reason, unmetered: true };
  }

  const stream = parseStream(child.lines);
  const tokens = stream.tokens ?? 0;
  const unmetered = stream.tokens === undefined;
  const lastAt = child.lines[child.lines.length - 1]?.at ?? started;
  const unanswered: string[] = [];
  const outcome = (cls: OutcomeClass, lookups: number, at: number): Outcome => ({
    class: cls,
    lookups,
    ...(unanswered.length > 0 ? { unanswered } : {}),
    seconds: seconds(at),
    tokens,
    ...(unmetered ? { unmetered } : {}),
  });
  const failed = (reason: string): Outcome => ({ ...outcome('failed', 0, lastAt), reason });
  const gaveUp = (lookups: number, at: number, reason: string): Outcome => ({
    ...outcome('gave-up', lookups, at),
    reason,
    // The stub checks at most MAX_LOOKUPS lookups; the one past the limit is never served.
    ...(lookups > 0 && unanswered.length === Math.min(lookups, MAX_LOOKUPS) ? { harnessLimited: true } : {}),
  });

  if (stubFailed(stream.stubStatus)) return failed(`stub not connected (${stream.stubStatus})`);
  if (stream.stubStatus === 'connected' && stream.toolsOffered === false) return failed('stub tools not offered');
  if (stream.compacted) return failed('auto-compacted');

  let lookups = 0;
  for (const call of stream.calls) {
    if (familyOf(call.tool) === familyOf(context.step.tool)) {
      if (matchesRecorded(call, context.step, context.prefix)) return outcome(lookups === 0 ? 'same' : 'recovered', lookups, call.at);
      if (!context.reachable) return outcome('unreachable', lookups, call.at);
      return { ...outcome('wrong', lookups, call.at), action: { tool: call.tool, input: call.input } };
    }
    lookups++;
    if (lookups > MAX_LOOKUPS) return gaveUp(lookups, call.at, 'lookup limit');
    if (context.lookup.serve({ tool: call.tool, input: call.input }) === undefined) unanswered.push(context.lookup.miss({ tool: call.tool, input: call.input }));
  }
  if (!stream.sawResult) return failed(child.code === 0 ? 'no result' : `exit ${child.code ?? 'signal'}`);
  if (stream.isError) return failed('error result');
  // A stub that was still starting when the child began may never have been offered to the model.
  if (stream.stubStatus === 'pending' && stream.calls.length === 0) return failed('stub still pending at start');
  return gaveUp(lookups, lastAt, stream.calls.length === 0 ? 'no tool call' : 'no final action');
}

/** The text a tool-less child writes for `prompt` (the summary arm), with what it cost. */
export async function runText(
  prompt: string,
  deps: SessionDeps,
): Promise<{ text: string; tokens: number; unmetered: boolean; reason?: string }> {
  const child = await deps.runner({
    args: buildArgs({ model: deps.model }),
    prompt,
    cwd: deps.scratch.cwd,
    timeoutMs: CHILD_TIMEOUT_MS,
  });
  if (!child.ok) return { text: '', tokens: 0, unmetered: true, reason: child.reason };
  const stream = parseStream(child.lines);
  const reason = !stream.sawResult ? 'no result' : stream.isError ? 'error result' : stream.resultText === '' ? 'empty summary' : undefined;
  return {
    text: stream.resultText,
    tokens: stream.tokens ?? 0,
    unmetered: stream.tokens === undefined,
    ...(reason === undefined ? {} : { reason }),
  };
}
