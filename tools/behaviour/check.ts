import { parseTranscript } from '../replay/transcript.js';
import { pointContext } from './history.js';
import { runPoint } from './session.js';
import type { SessionDeps } from './session.js';
import type { LostPoint } from './select.js';

/** What a check prints on failure and success; the caller turns it into an exit code. */
export interface CheckResult {
  code: number;
  output: string;
}

/** A synthetic four-message session: a Read the stub can serve, then an Edit as the recorded step. */
const SESSION = [
  { type: 'user', message: { role: 'user', content: 'fix check.txt' } },
  { type: 'assistant', message: { id: 'a1', role: 'assistant', content: [{ type: 'tool_use', id: 'r1', name: 'Read', input: { file_path: 'check.txt' } }] } },
  { type: 'user', message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'r1', content: 'hello' }] } },
  { type: 'assistant', message: { id: 'a2', role: 'assistant', content: [{ type: 'tool_use', id: 'e1', name: 'Edit', input: { file_path: 'check.txt', old_string: 'hello', new_string: 'bye' } }] } },
]
  .map((entry) => JSON.stringify(entry))
  .join('\n');

const PROMPT =
  'This is a connectivity check, not a coding task. Call the Read tool once with file_path "check.txt", ' +
  'then reply with the single word: done';

/**
 * Runs one tiny synthetic prompt through the same child, isolation and stub as a real run and
 * reports whether the model could use a stub tool. It sends no session text and costs a few
 * thousand tokens, so a wrong CLI assumption shows up here instead of in a 250k-token point.
 */
export async function runCheck(session: SessionDeps): Promise<CheckResult> {
  const path = session.scratch.file('check.jsonl', SESSION);
  const { messages } = parseTranscript(SESSION);
  const point: LostPoint = { session: 'check.jsonl', path, messageIndex: 3, toolUseId: 'e1', tool: 'Edit', losses: [], messages };
  const outcome = await runPoint(point, pointContext(point), PROMPT, session);

  const problem =
    outcome.class === 'failed'
      ? `the child did not run to a usable result (${outcome.reason ?? 'unknown'})`
      : outcome.lookups === 0 && outcome.class === 'gave-up'
        ? 'the child ran, but the model made no stub tool call'
        : undefined;
  const lines = [
    'Environment check: one synthetic prompt, no session text sent',
    `Model: ${session.model}`,
    `Result: ${problem === undefined ? 'OK, the model called a stub tool and the stub answered' : `PROBLEM, ${problem}`}`,
    `Stub tool calls seen: ${outcome.lookups}`,
    `Tokens used: ${outcome.tokens}${outcome.unmetered ? ' (usage not reported)' : ''}`,
  ];
  return { code: problem === undefined ? 0 : 1, output: lines.join('\n') };
}
