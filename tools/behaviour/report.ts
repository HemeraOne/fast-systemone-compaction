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
}

export interface Summary {
  run: RunInfo;
  arms: ArmSummary[];
  points: PointRecord[];
}

const CLASSES: readonly OutcomeClass[] = ['same', 'recovered', 'wrong', 'gave-up', 'unreachable', 'failed'];
const DEVIATIONS: readonly OutcomeClass[] = ['wrong', 'gave-up', 'unreachable'];

function median(values: readonly number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

/** `reason xN` for each distinct reason, in order of first appearance. */
function failureReasons(outcomes: readonly Outcome[]): string[] {
  const counts = new Map<string, number>();
  for (const outcome of outcomes) {
    if (outcome.class === 'failed') counts.set(outcome.reason ?? 'unknown', (counts.get(outcome.reason ?? 'unknown') ?? 0) + 1);
  }
  return [...counts].map(([reason, n]) => `${reason} x${n}`);
}

export function summarize(points: readonly PointRecord[], run: RunInfo): Summary {
  const arms: Arm[] = run.summaryRan ? ['control', 'compacted', 'summary'] : ['control', 'compacted'];
  return {
    run,
    points: [...points],
    arms: arms.map((arm) => {
      const outcomes = points.flatMap((point) => point.outcomes[arm] ?? []);
      const counts = Object.fromEntries(
        CLASSES.map((c) => [c, outcomes.filter((outcome) => outcome.class === c).length]),
      ) as Record<OutcomeClass, number>;
      const recovered = outcomes.filter((outcome) => outcome.class === 'recovered');
      return {
        arm,
        counts,
        total: outcomes.length - counts.failed,
        medianLookups: median(recovered.map((outcome) => outcome.lookups)),
        medianSeconds: median(recovered.map((outcome) => outcome.seconds)),
        failures: failureReasons(outcomes),
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
    lines.push(`  ${c.padEnd(12)}${count}/${summary.total} (${rate})`);
  }
  if (summary.medianLookups !== undefined && summary.medianSeconds !== undefined) {
    lines.push(
      `  recovered: median ${summary.medianLookups} extra lookups, ${summary.medianSeconds.toFixed(1)} s`,
    );
  }
  if (summary.failures.length > 0) lines.push(`  failed to run: ${summary.failures.join(', ')}`);
  return lines;
}

function comparisonLines(arms: readonly ArmSummary[]): string[] {
  const control = arms.find((arm) => arm.arm === 'control');
  const compacted = arms.find((arm) => arm.arm === 'compacted');
  if (control === undefined || compacted === undefined) return [];
  const lines = ['Control comparison (compacted against the uncompacted history)'];
  for (const c of DEVIATIONS) {
    const verdict =
      compacted.counts[c] > control.counts[c]
        ? 'more than the control arm shows'
        : 'within what the control arm shows';
    lines.push(`  ${c}: compacted ${compacted.counts[c]}, control ${control.counts[c]} - ${verdict}`);
  }
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
    `Tokens: ${run.tokens} used, cap ${run.tokenCap}${run.stopped === undefined ? '' : ` - stopped early (${run.tried} point(s) done): ${run.stopped}`}`,
    '',
  ];
  for (const arm of summary.arms) lines.push(...armLines(arm), '');
  if (!run.summaryRan) lines.push('Summary arm not run (pass --summary to add an approximate one)', '');
  lines.push(...comparisonLines(summary.arms), '', 'Points');
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
