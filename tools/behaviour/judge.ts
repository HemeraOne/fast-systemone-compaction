import { readFileSync } from 'node:fs';
import { runText } from './session.js';
import type { Action, JudgeVerdict, SessionDeps } from './session.js';

/** Starting values (spec 006, SC-003), not statistical claims. */
export const JUDGE_BAR = 0.9;
export const JUDGE_MIN_PAIRS = 20;

/** The verdict definitions the judge is asked to apply; documented verbatim in the README (FR-011). */
export const JUDGE_PROMPT =
  'Two tool calls follow. The first is what an assistant really did next in a coding session (the recorded step). ' +
  'The second is what an assistant did when given a shortened history. Decide whether the second call would do the ' +
  'same job as the first. Answer "equivalent" if it would reach the same result (for example a command spelled ' +
  'differently, a path written another way, or an edit that replaces a different span of the same text with the same ' +
  'effect). Answer "different" if it would act on another file, run another command, or change something else. ' +
  'Answer with exactly one word: equivalent or different.';

/** The first word of the reply, when it is a verdict word; anything else is `undecided`. */
export function parseVerdict(text: string): JudgeVerdict {
  const word = text.trim().split(/\s+/)[0]?.toLowerCase().replace(/[^a-z]/g, '');
  return word === 'equivalent' || word === 'different' ? word : 'undecided';
}

export function judgePrompt(recorded: Action, action: Action): string {
  return `${JUDGE_PROMPT}\n\nRecorded step:\n${JSON.stringify(recorded)}\n\nSecond call:\n${JSON.stringify(action)}`;
}

/** The judge's own spend. It closes at the cap, and when a call reports no usage the cap cannot be enforced. */
export class JudgeBudget {
  used = 0;
  private blind = false;

  constructor(readonly cap: number) {}

  get open(): boolean {
    return !this.blind && this.used < this.cap;
  }

  add(tokens: number, unmetered: boolean): void {
    this.used += tokens;
    if (unmetered) this.blind = true;
  }
}

/**
 * One verdict per pair, from a tool-less child. A failed, empty or malformed reply is
 * `undecided`; `undefined` means the budget was closed and the pair was not judged.
 */
export async function judgeIfOpen(
  budget: JudgeBudget,
  recorded: Action,
  action: Action,
  deps: SessionDeps,
): Promise<JudgeVerdict | undefined> {
  if (!budget.open) return undefined;
  const reply = await runText(judgePrompt(recorded, action), deps);
  budget.add(reply.tokens, reply.unmetered);
  return reply.reason === undefined ? parseVerdict(reply.text) : 'undecided';
}

export interface CalibrationPair {
  recorded: Action;
  action: Action;
  label: 'equivalent' | 'different';
}

export interface CalibrationResult {
  correct: number;
  total: number;
  rate: number;
  bar: number;
  minimum: number;
  passed: boolean;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const isAction = (value: unknown): value is Action =>
  isRecord(value) && typeof value['tool'] === 'string' && isRecord(value['input']);

/** Fixed messages only: a calibration file may hold private text, so none of it is echoed. */
export const CALIBRATION_UNREADABLE = 'The calibration file could not be read';
export const CALIBRATION_MALFORMED =
  'The calibration file must be a JSON array of { recorded, action, label } with label equivalent or different';

export function parseCalibrationSet(json: string): CalibrationPair[] {
  let data: unknown;
  try {
    data = JSON.parse(json);
  } catch {
    throw new Error(CALIBRATION_MALFORMED);
  }
  if (!Array.isArray(data) || data.length === 0) throw new Error(CALIBRATION_MALFORMED);
  return data.map((item) => {
    if (!isRecord(item) || !isAction(item['recorded']) || !isAction(item['action'])) throw new Error(CALIBRATION_MALFORMED);
    const label = item['label'];
    if (label !== 'equivalent' && label !== 'different') throw new Error(CALIBRATION_MALFORMED);
    return { recorded: item['recorded'], action: item['action'], label };
  });
}

export function loadCalibrationSet(path: string): CalibrationPair[] {
  let json: string;
  try {
    json = readFileSync(path, 'utf8');
  } catch {
    throw new Error(CALIBRATION_UNREADABLE);
  }
  return parseCalibrationSet(json);
}

/**
 * Grades each labelled pair once. A pair left unjudged by the cap, like an `undecided` reply,
 * counts as a miss. Passing needs at least `JUDGE_MIN_PAIRS` pairs and `JUDGE_BAR` agreement.
 */
export async function calibrate(pairs: readonly CalibrationPair[], budget: JudgeBudget, deps: SessionDeps): Promise<CalibrationResult> {
  let correct = 0;
  for (const pair of pairs) {
    if ((await judgeIfOpen(budget, pair.recorded, pair.action, deps)) === pair.label) correct++;
  }
  const total = pairs.length;
  const rate = total === 0 ? 0 : correct / total;
  return { correct, total, rate, bar: JUDGE_BAR, minimum: JUDGE_MIN_PAIRS, passed: total >= JUDGE_MIN_PAIRS && rate >= JUDGE_BAR };
}
