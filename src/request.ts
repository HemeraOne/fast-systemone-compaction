import type { JevAnswer, JevQuestions, JevResponse, JevState } from './types.js';

export const SYSTEM_ONE_URL = 'https://api.typesafe.ai/v1/systemone';
export const DEFAULT_MODEL = 'jev-latest';

export interface JevRequest {
  url: string;
  method: 'POST';
  headers: Record<string, string>;
  body: string;
}

/** The HTTP request for one Jev call, for any fetch-like transport. */
export function buildJevRequest(
  params: {
    apiKey: string;
    model?: string;
    baseUrl?: string;
  },
  state: JevState,
  questions: JevQuestions,
): JevRequest {
  return {
    url: params.baseUrl ?? SYSTEM_ONE_URL,
    method: 'POST',
    headers: {
      authorization: `Bearer ${params.apiKey}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: params.model ?? DEFAULT_MODEL,
      state,
      questions,
    }),
  };
}

/** Validates a raw `baseUrl` option: unset/empty means "use the default", anything
 * else must be an absolute http(s) URL or it is rejected (never silently defaulted). */
export function resolveBaseUrl(raw: string | undefined): { baseUrl?: string; invalidBaseUrl?: string } {
  const trimmed = raw?.trim();
  if (!trimmed) return {};
  try {
    const url = new URL(trimmed);
    if (url.protocol === 'http:' || url.protocol === 'https:') return { baseUrl: trimmed };
  } catch {
    // falls through to invalid
  }
  return { invalidBaseUrl: trimmed };
}

/** `<host> <model>` for outcome logging; falls back to the raw input if it isn't a URL. */
export function backendMarker(urlOrRaw: string, model: string): string {
  let host = urlOrRaw;
  try {
    const url = new URL(urlOrRaw);
    if ((url.protocol === 'http:' || url.protocol === 'https:') && url.host) host = url.host;
  } catch {
    // not a URL (e.g. a rejected invalid baseUrl) — use the raw value as-is
  }
  return `${host} ${model}`;
}

/** Validates a Jev response body; throws on anything but an `answers` object. */
export function parseJevResponse(
  status: number,
  ok: boolean,
  text: string,
): JevResponse {
  if (!ok) {
    throw new Error(`Jev request failed (${status}): ${text.slice(0, 200)}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('Jev returned malformed JSON');
  }
  // Cloudflare Workers AI REST wraps the System One body in { result, success, errors, messages }.
  if (
    parsed !== null &&
    typeof parsed === 'object' &&
    !('answers' in parsed) &&
    'result' in parsed &&
    parsed.result !== null &&
    typeof parsed.result === 'object'
  ) {
    parsed = parsed.result;
  }
  if (
    parsed === null ||
    typeof parsed !== 'object' ||
    !('answers' in parsed) ||
    parsed.answers === null ||
    typeof parsed.answers !== 'object'
  ) {
    throw new Error('Jev response is missing answers');
  }
  return parsed as JevResponse;
}

/** The `noul` probability of one answer; throws when it is not there. */
export function noulAnswer(
  answers: Record<string, JevAnswer>,
  name: string,
): number {
  const answer = answers[name];
  if (
    !answer ||
    !('noul' in answer) ||
    typeof answer.noul !== 'number' ||
    !Number.isFinite(answer.noul)
  ) {
    throw new Error(`Invalid Jev answer for ${name}`);
  }
  return answer.noul;
}
