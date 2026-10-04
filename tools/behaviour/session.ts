import { familyOf, matchesRecorded } from './history.js';
import type { ApiBlock, ApiMessage, PointContext } from './history.js';
import { callModel, TOOLS } from './model.js';

export type Arm = 'control' | 'compacted' | 'summary';

export type OutcomeClass = 'same' | 'recovered' | 'wrong' | 'gave-up' | 'unreachable' | 'failed';

export interface Outcome {
  class: OutcomeClass;
  /** Lookups issued before the terminal action, answerable or not. */
  lookups: number;
  seconds: number;
  /** Input plus output tokens of this outcome's model calls. */
  tokens: number;
}

export interface SessionDeps {
  fetch: typeof fetch;
  apiKey: string;
  model: string;
  now: () => number;
}

export const MAX_LOOKUPS = 5;

const NOT_AVAILABLE = 'not available in this test';

/**
 * Lets the model take its next step from `history` and classifies it against the recorded
 * step. The first call in the recorded step's tool family is the terminal action; every other
 * call is a lookup answered from recorded results. Nothing is written anywhere: a proposed
 * edit is only compared.
 */
export async function runPoint(
  context: PointContext,
  history: readonly ApiMessage[],
  deps: SessionDeps,
): Promise<Outcome> {
  const started = deps.now();
  const messages = [...history];
  let lookups = 0;
  let tokens = 0;
  const finish = (outcomeClass: OutcomeClass): Outcome => ({
    class: outcomeClass,
    lookups,
    seconds: (deps.now() - started) / 1000,
    tokens,
  });

  for (;;) {
    const reply = await callModel({ apiKey: deps.apiKey, model: deps.model, messages, tools: TOOLS }, deps.fetch);
    if (!reply.ok) return finish('failed');
    tokens += reply.inputTokens + reply.outputTokens;

    const calls = reply.content.filter((block) => block['type'] === 'tool_use');
    if (calls.length === 0) return finish('gave-up');

    const answers: ApiBlock[] = [];
    for (const block of calls) {
      const input = block['input'];
      const call = {
        tool: String(block['name']),
        input: typeof input === 'object' && input !== null ? (input as Record<string, unknown>) : {},
      };
      if (familyOf(call.tool) === familyOf(context.step.tool)) {
        if (matchesRecorded(call, context.step, context.prefix)) return finish(lookups === 0 ? 'same' : 'recovered');
        return finish(context.reachable ? 'wrong' : 'unreachable');
      }
      lookups++;
      if (lookups > MAX_LOOKUPS) return finish('gave-up');
      const served = context.lookup.serve(call);
      answers.push({
        type: 'tool_result',
        tool_use_id: String(block['id']),
        content: served ?? NOT_AVAILABLE,
        is_error: served === undefined,
      });
    }
    messages.push({ role: 'assistant', content: reply.content }, { role: 'user', content: answers });
  }
}
