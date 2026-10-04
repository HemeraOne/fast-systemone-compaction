import type { ApiBlock, ApiMessage } from './history.js';

const ENDPOINT = 'https://api.anthropic.com/v1/messages';
const MAX_OUTPUT_TOKENS = 2048;

/** A short stand-in: the transcripts do not contain Claude Code's own system prompt. */
export const SYSTEM_PROMPT =
  'You are a coding assistant working in a software project, helping the user with their request. ' +
  'Continue the work with the tools you have. Call one tool at a time and act on what the earlier ' +
  'conversation already established.';

const stub = (name: string, description: string, properties: Record<string, string>, required: string[]) => ({
  name,
  description,
  input_schema: {
    type: 'object',
    properties: Object.fromEntries(Object.entries(properties).map(([key, text]) => [key, { type: 'string', description: text }])),
    required,
  },
});

/** The five stub tools; their results come from recorded history, never from real files. */
export const TOOLS = [
  stub('Read', 'Read a file.', { file_path: 'Path of the file' }, ['file_path']),
  stub('Grep', 'Search file contents.', { pattern: 'Pattern to search for', path: 'Where to search' }, ['pattern']),
  stub('Glob', 'Find files by name pattern.', { pattern: 'Glob pattern' }, ['pattern']),
  stub(
    'Edit',
    'Replace text in a file.',
    { file_path: 'Path of the file', old_string: 'Text to replace', new_string: 'Replacement text' },
    ['file_path', 'old_string', 'new_string'],
  ),
  stub('Bash', 'Run a shell command.', { command: 'The command' }, ['command']),
];

export interface ModelRequest {
  apiKey: string;
  model: string;
  messages: ApiMessage[];
  /** Left out for the summary call. */
  tools?: typeof TOOLS;
}

export type ModelReply =
  | { ok: true; content: ApiBlock[]; inputTokens: number; outputTokens: number }
  | { ok: false; reason: string };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * One Messages API call. Never throws and never puts the key, the request, or the reply body
 * into the failure reason: any problem becomes `{ ok: false }` and the point is `failed`.
 */
export async function callModel(request: ModelRequest, fetchImpl: typeof fetch): Promise<ModelReply> {
  let response: Response;
  try {
    response = await fetchImpl(ENDPOINT, {
      method: 'POST',
      headers: {
        'x-api-key': request.apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: request.model,
        max_tokens: MAX_OUTPUT_TOKENS,
        system: SYSTEM_PROMPT,
        messages: request.messages,
        ...(request.tools ? { tools: request.tools } : {}),
      }),
    });
  } catch {
    return { ok: false, reason: 'network error' };
  }
  if (!response.ok) return { ok: false, reason: `http ${response.status}` };

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { ok: false, reason: 'malformed reply' };
  }
  if (!isRecord(body) || !Array.isArray(body['content']) || !isRecord(body['usage'])) {
    return { ok: false, reason: 'malformed reply' };
  }
  if (body['stop_reason'] === 'refusal') return { ok: false, reason: 'refusal' };
  const { input_tokens: input, output_tokens: output } = body['usage'];
  if (typeof input !== 'number' || typeof output !== 'number') return { ok: false, reason: 'malformed reply' };
  return { ok: true, content: body['content'].filter(isRecord), inputTokens: input, outputTokens: output };
}
