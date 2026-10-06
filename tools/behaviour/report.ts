import type { Rule, ValueKind } from '../replay/replay.js';
import type { Arm, Outcome, OutcomeClass } from './session.js';

/** One lost point and what each arm did. Identifiers and positions only, never text. */
export interface PointRecord {
  session: string;
  messageIndex: number;
  tool: string;
  kinds: ValueKind[];
  rules: Rule[];
  outcomes: Partial<Record<Arm, Outcome>>;
}

export interface RunInfo {
  model: string;
  sessions: number;
  /** Lost points left out because their step is one of several parallel calls of its kind. */
  skipped: number;
  available: number;
  tried: number;
  tokens: number;
  tokenCap: number;
  /** Why the run ended before the sample was done; `undefined` when it ran to the end. */
  stopped: string | undefined;
  summaryRan: boolean;
  /** How the points were narrowed (size filter, skip); `undefined` for the default even sample. */
  selection?: string;
}

export interface ArmSummary {
  arm: Arm;
  counts: Record<OutcomeClass, number>;
  /** Outcomes that are not `failed`: the denominator of every rate. */
  total: number;
  medianLookups: number | undefined;
  medianSeconds: number | undefined;
  /** The fixed reasons of `failed` outcomes with how often each occurred, e.g. `exit 1 x2`. */
  failures: string[];
  /** The same for `gave-up`: `no tool call`, `no final action` (stopped after lookups), `lookup limit`; with lookups, each adds `(N lookups, M unanswerable: tools)`. */
  gaveUp: string[];
  /** `gave-up` outcomes whose lookups the stub answered none of: a limit of the harness, not model behaviour. */
  harnessLimited: number;
  /** Tools of the lookups the stub could not answer in `recovered` outcomes, e.g. `Bash x2`; empty when none. */
  recoveredUnanswered: string;
  /** Outcomes that reached a final action, and the median time to it. */
  finals: number;
  medianFinalSeconds: number | undefined;
}

/** Compacted against control over the points where both arms reached a final action. */
export interface Paired {
  points: number;
  medianExtraSeconds: number | undefined;
  medianExtraLookups: number | undefined;
}

export interface Summary {
  run: RunInfo;
  arms: ArmSummary[];
  paired: Paired;
  points: PointRecord[];
}

const CLASSES: readonly OutcomeClass[] = ['same', 'recovered', 'wrong', 'gave-up', 'unreachable', 'failed'];
const DEVIATIONS: readonly OutcomeClass[] = ['wrong', 'gave-up', 'unreachable'];
const FINAL: readonly OutcomeClass[] = ['same', 'recovered', 'wrong', 'unreachable'];
const reachedFinal = (outcome: Outcome | undefined): outcome is Outcome => outcome !== undefined && FINAL.includes(outcome.class);

function median(values: readonly number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/** Tool names with how often each occurred, e.g. `Bash x2, Grep`, in order of first appearance. */
function toolCounts(tools: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const tool of tools) counts.set(tool, (counts.get(tool) ?? 0) + 1);
  return [...counts].map(([tool, n]) => (n > 1 ? `${tool} x${n}` : tool)).join(', ');
}

/** What the model looked up before it stopped: a count and the tools the stub could not answer. */
function lookupNote(outcome: Outcome): string {
  const unanswered = outcome.unanswered ?? [];
  return ` (${outcome.lookups} lookups, ${unanswered.length} unanswerable${unanswered.length > 0 ? `: ${toolCounts(unanswered)}` : ''})`;
}

/** `reason xN` for each distinct reason of the given class, in order of first appearance. */
function reasons(outcomes: readonly Outcome[], cls: OutcomeClass): string[] {
  const counts = new Map<string, number>();
  for (const outcome of outcomes) {
    if (outcome.class !== cls) continue;
    const label = `${outcome.reason ?? 'unknown'}${cls === 'gave-up' && outcome.lookups > 0 ? lookupNote(outcome) : ''}`;
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  return [...counts].map(([label, n]) => `${label} x${n}`);
}

export function summarize(points: readonly PointRecord[], run: RunInfo): Summary {
  const arms: Arm[] = run.summaryRan ? ['control', 'compacted', 'summary'] : ['control', 'compacted'];
  const pairs = points.flatMap((point) => {
    const { control, compacted } = point.outcomes;
    return reachedFinal(control) && reachedFinal(compacted) ? [{ seconds: compacted.seconds - control.seconds, lookups: compacted.lookups - control.lookups }] : [];
  });
  return {
    run,
    points: [...points],
    paired: {
      points: pairs.length,
      medianExtraSeconds: median(pairs.map((pair) => pair.seconds)),
      medianExtraLookups: median(pairs.map((pair) => pair.lookups)),
    },
    arms: arms.map((arm) => {
      const outcomes = points.flatMap((point) => point.outcomes[arm] ?? []);
      const counts = Object.fromEntries(
        CLASSES.map((c) => [c, outcomes.filter((outcome) => outcome.class === c).length]),
      ) as Record<OutcomeClass, number>;
      const recovered = outcomes.filter((outcome) => outcome.class === 'recovered');
      const finals = outcomes.filter(reachedFinal);
      return {
        arm,
        counts,
        total: outcomes.length - counts.failed,
        medianLookups: median(recovered.map((outcome) => outcome.lookups)),
        medianSeconds: median(recovered.map((outcome) => outcome.seconds)),
        failures: reasons(outcomes, 'failed'),
        gaveUp: reasons(outcomes, 'gave-up'),
        harnessLimited: outcomes.filter((outcome) => outcome.class === 'gave-up' && outcome.harnessLimited === true).length,
        recoveredUnanswered: toolCounts(recovered.flatMap((outcome) => outcome.unanswered ?? [])),
        finals: finals.length,
        medianFinalSeconds: median(finals.map((outcome) => outcome.seconds)),
      };
    }),
  };
}

const unique = <T>(items: readonly T[]): T[] => [...new Set(items)];

function armLines(summary: ArmSummary): string[] {
  const label = summary.arm === 'summary' ? 'summary (approximate: does not re-attach recently read files)' : summary.arm;
  const lines = [`Arm: ${label} (${summary.total} judged, ${summary.counts.failed} failed to run)`];
  for (const c of CLASSES.filter((name) => name !== 'failed')) {
    const count = summary.counts[c];
    const rate = summary.total === 0 ? 'n/a' : `${Math.round((count / summary.total) * 100)}%`;
    const why = c === 'gave-up' && summary.gaveUp.length > 0 ? ` - ${summary.gaveUp.join(', ')}` : '';
    lines.push(`  ${c.padEnd(12)}${count}/${summary.total} (${rate})${why}`);
  }
  if (summary.harnessLimited > 0) {
    lines.push(`  harness-limited: ${summary.harnessLimited} of the gave-up (the stub answered none of their lookups)`);
  }
  if (summary.medianFinalSeconds !== undefined) {
    lines.push(`  time to final action: median ${summary.medianFinalSeconds.toFixed(1)} s over ${summary.finals} outcome(s)`);
  }
  if (summary.medianLookups !== undefined && summary.medianSeconds !== undefined) {
    lines.push(
      `  recovered: median ${summary.medianLookups} extra lookups, ${summary.medianSeconds.toFixed(1)} s to the final action` +
        `${summary.recoveredUnanswered === '' ? '' : `; unanswerable lookups: ${summary.recoveredUnanswered}`}`,
    );
  }
  if (summary.failures.length > 0) lines.push(`  failed to run: ${summary.failures.join(', ')}`);
  return lines;
}

const signed = (value: number, digits = 1): string => `${value > 0 ? '+' : ''}${value.toFixed(digits)}`;

function comparisonLines(arms: readonly ArmSummary[], paired: Paired, points: readonly PointRecord[]): string[] {
  const control = arms.find((arm) => arm.arm === 'control');
  const compacted = arms.find((arm) => arm.arm === 'compacted');
  if (control === undefined || compacted === undefined) return [];
  const lines = ['Control comparison (compacted against the uncompacted history)'];
  // A point the control arm also got wrong is not decided by the history, so it says nothing about compaction.
  const controlWrong = points.filter((point) => point.outcomes.control?.class === 'wrong');
  const bothWrong = controlWrong.filter((point) => point.outcomes.compacted?.class === 'wrong').length;
  for (const c of DEVIATIONS) {
    // Harness-limited gave-ups say nothing about compaction, so they are left out of the comparison.
    const [compactedCount, controlCount] =
      c === 'gave-up'
        ? [compacted.counts[c] - compacted.harnessLimited, control.counts[c] - control.harnessLimited]
        : c === 'wrong'
          ? [compacted.counts[c] - bothWrong, control.counts[c] - controlWrong.length]
          : [compacted.counts[c], control.counts[c]];
    const verdict = compactedCount > controlCount ? 'more than the control arm shows' : 'within what the control arm shows';
    const excluded =
      c === 'gave-up' && compacted.harnessLimited + control.harnessLimited > 0
        ? ` (harness-limited left out: compacted ${compacted.harnessLimited}, control ${control.harnessLimited})`
        : c === 'wrong' && controlWrong.length > 0
          ? ` (left out: ${controlWrong.length} point(s) wrong in the control arm too, so not decided by the history; compacted wrong there: ${bothWrong})`
          : '';
    lines.push(`  ${c}: compacted ${compactedCount}, control ${controlCount}${excluded} - ${verdict}`);
  }
  lines.push(
    paired.medianExtraSeconds === undefined || paired.medianExtraLookups === undefined
      ? '  extra effort: no point reached a final action in both arms, so there is nothing to compare'
      : `  extra effort of compacted over control: median ${signed(paired.medianExtraSeconds)} s and ` +
          `${signed(paired.medianExtraLookups, 0)} lookups, over ${paired.points} point(s) where both arms reached a final action`,
  );
  return lines;
}

/** A pure function of its input: no clock, no environment, and no transcript text. */
export function formatReport(summary: Summary): string {
  const { run } = summary;
  const lines = [
    'Behaviour test: what the assistant does at points where compaction lost a value',
    `Model: ${run.model}`,
    `Corpus: ${run.sessions} sessions, ${run.available} lost points available` +
      `${run.skipped > 0 ? ` (${run.skipped} more skipped: one of several parallel calls of its kind)` : ''}, ${run.tried} tried`,
    ...(run.selection === undefined ? [] : [`Selection: ${run.selection}`]),
    `Tokens: ${run.tokens} used, cap ${run.tokenCap}${run.stopped === undefined ? '' : ` - stopped early (${run.tried} point(s) done): ${run.stopped}`}`,
    '',
  ];
  for (const arm of summary.arms) lines.push(...armLines(arm), '');
  if (!run.summaryRan) lines.push('Summary arm not run (pass --summary to add an approximate one)', '');
  lines.push(...comparisonLines(summary.arms, summary.paired, summary.points), '', 'Points');
  for (const point of summary.points) {
    const results = summary.arms.map((arm) => `${arm.arm}=${point.outcomes[arm.arm]?.class ?? '-'}`).join(' ');
    lines.push(
      `  ${point.session} #${point.messageIndex} ${point.tool} [${unique(point.kinds).join('+')}] ` +
        `[${unique(point.rules).join('+')}] ${results}`,
    );
  }
  lines.push(
    '',
    'Caveats: a stand-in system prompt and five stub tools replace Claude Code\'s own, and the history',
    'reaches the model as text rather than as real tool turns, so rates are not the real app\'s;',
    'lookups are answered only for exact repeats (Read matches on path), so a recovery by another',
    'command or pattern counts as not available; with this few points, read the counts, not the',
    'percentages.',
  );
  return lines.join('\n');
}
