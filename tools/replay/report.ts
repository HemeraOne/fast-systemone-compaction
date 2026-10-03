import type { Loss, Rule, SessionResult, ValueKind } from './replay.js';

export interface Summary {
  sessionsRead: number;
  sessionsSkipped: number;
  malformedLines: number;
  pointsChecked: number;
  excludedTooShort: number;
  excludedLowReduction: number;
  neededValues: number;
  lostValues: number;
  pointsWithLoss: number;
  byRule: Record<Rule, number>;
  neededByKind: Record<ValueKind, number>;
  lostByKind: Record<ValueKind, number>;
  losses: Loss[];
}

const RULE_LABELS: Record<Rule, string> = { rule1: 'rule 1', rule2: 'rule 2', both: 'both' };

function byLocator(a: Loss, b: Loss): number {
  const [x, y] = [a.locator, b.locator];
  return (
    x.session.localeCompare(y.session) ||
    x.messageIndex - y.messageIndex ||
    x.tool.localeCompare(y.tool) ||
    a.kind.localeCompare(b.kind)
  );
}

export function summarize(results: readonly SessionResult[]): Summary {
  const summary: Summary = {
    sessionsRead: 0,
    sessionsSkipped: 0,
    malformedLines: 0,
    pointsChecked: 0,
    excludedTooShort: 0,
    excludedLowReduction: 0,
    neededValues: 0,
    lostValues: 0,
    pointsWithLoss: 0,
    byRule: { rule1: 0, rule2: 0, both: 0 },
    neededByKind: { path: 0, command: 0, editTarget: 0 },
    lostByKind: { path: 0, command: 0, editTarget: 0 },
    losses: [],
  };
  for (const result of results) {
    summary.malformedLines += result.malformedLines;
    if (result.skipped) {
      summary.sessionsSkipped++;
      continue;
    }
    summary.sessionsRead++;
    for (const point of result.points) {
      if (point.status === 'excluded') {
        if (point.excludedReason === 'history-too-short') summary.excludedTooShort++;
        else summary.excludedLowReduction++;
        continue;
      }
      summary.pointsChecked++;
      summary.neededValues += point.needed;
      for (const kind of Object.keys(point.neededByKind) as ValueKind[]) {
        summary.neededByKind[kind] += point.neededByKind[kind];
      }
      if (point.losses.length > 0) summary.pointsWithLoss++;
      for (const loss of point.losses) {
        summary.lostValues++;
        summary.byRule[loss.rule]++;
        summary.lostByKind[loss.kind]++;
        summary.losses.push(loss);
      }
    }
  }
  summary.losses.sort(byLocator);
  return summary;
}

function rate(lost: number, needed: number): string {
  return needed === 0 ? 'n/a' : `${((100 * lost) / needed).toFixed(1)}%`;
}

export function formatReport(summary: Summary): string {
  const excluded = summary.excludedTooShort + summary.excludedLowReduction;
  const rule = (label: string, lost: number): string =>
    `  ${label.padEnd(8)}${String(lost).padStart(7)}${rate(lost, summary.neededValues).padStart(8)}`;
  const kind = (label: ValueKind): string =>
    `  ${label.padEnd(12)}${String(summary.neededByKind[label]).padStart(7)}${String(summary.lostByKind[label]).padStart(6)}${rate(summary.lostByKind[label], summary.neededByKind[label]).padStart(8)}`;
  const lossLine = (loss: Loss): string =>
    `  ${loss.locator.session} #${loss.locator.messageIndex} ${loss.locator.tool} ${loss.kind} ${loss.length} chars ${RULE_LABELS[loss.rule]}`;
  return [
    'Safety replay of rule-based compaction (default options)',
    `Sessions read: ${summary.sessionsRead}   skipped: ${summary.sessionsSkipped}   malformed lines: ${summary.malformedLines}`,
    `Replay points checked: ${summary.pointsChecked}   excluded: ${excluded} (history too short: ${summary.excludedTooShort}, below minimum reduction: ${summary.excludedLowReduction})`,
    `Needed values: ${summary.neededValues}   lost: ${summary.lostValues}   loss rate: ${rate(summary.lostValues, summary.neededValues)}   points with a loss: ${summary.pointsWithLoss}`,
    '',
    `${'By rule'.padEnd(10)}${'lost'.padStart(7)}${'rate'.padStart(8)}`,
    rule('rule 1', summary.byRule.rule1),
    rule('rule 2', summary.byRule.rule2),
    rule('both', summary.byRule.both),
    '',
    `${'By kind'.padEnd(14)}${'needed'.padStart(7)}${'lost'.padStart(6)}${'rate'.padStart(8)}`,
    kind('path'),
    kind('command'),
    kind('editTarget'),
    '',
    `Losses (${summary.losses.length})`,
    ...summary.losses.map(lossLine),
    '',
    'A lost value is an upper bound on harm: the assistant used text that compaction removed,',
    'not proof that it would have acted differently. A surviving value does not prove the',
    'assistant would still have understood its context.',
    '',
  ].join('\n');
}
