import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

/** Longest one child may run before it is killed and its point counts as failed to run. */
export const CHILD_TIMEOUT_MS = 300_000;

/** A short stand-in: the transcripts do not contain Claude Code's own system prompt. */
export const SYSTEM_PROMPT =
  'You are a coding assistant working in a software project, helping the user with their request. ' +
  'Continue the work with the tools you have. Call one tool at a time and act on what the earlier ' +
  'conversation already established.';

const SERVER = 'stub';
const PREFIX = `mcp__${SERVER}__`;
const STUB_TOOL_NAMES = ['Read', 'Grep', 'Glob', 'Edit', 'Bash'];

export interface ArgsOptions {
  model: string;
  /** The MCP config that starts the stub; left out for the summary call, which has no tools. */
  mcpConfig?: string;
}

/**
 * The one place that lists the CLI flags (research R9). `--bare` is deliberately absent: it
 * never reads the subscription login. The prompt goes on stdin, not in the arguments.
 */
export function buildArgs(options: ArgsOptions): string[] {
  const args = [
    '-p',
    '--output-format',
    'stream-json',
    '--verbose',
    // Not `--safe-mode`: its help text says it disables MCP servers, and the first runs listed no
    // stub. The settings source `project` skips the user's settings, hooks and plugins; the
    // working directory is empty, so no project settings or CLAUDE.md exist either.
    '--setting-sources',
    'project',
    '--disable-slash-commands',
    '--no-session-persistence',
    '--model',
    options.model,
    '--system-prompt',
    SYSTEM_PROMPT,
    '--tools',
    '',
    '--strict-mcp-config',
    '--permission-prompts',
    'none',
  ];
  if (options.mcpConfig !== undefined) {
    args.push(
      '--mcp-config',
      options.mcpConfig,
      '--allowedTools',
      STUB_TOOL_NAMES.map((name) => `${PREFIX}${name}`).join(','),
    );
  }
  return args;
}

export interface StreamLine {
  text: string;
  /** Clock reading when the line arrived. */
  at: number;
}

export interface StreamCall {
  tool: string;
  input: Record<string, unknown>;
  at: number;
}

/** What the init event said about the stub; a fixed vocabulary, so it is safe to print. */
export type StubStatus = 'connected' | 'pending' | 'failed' | 'needs-auth' | 'disabled' | 'not listed' | 'unknown';

const KNOWN_STATUSES: readonly string[] = ['connected', 'pending', 'failed', 'needs-auth', 'disabled'];

/** A definitive "no": `pending` may still connect before the first turn, so it is not one. */
export function stubFailed(status: StubStatus | undefined): boolean {
  return status !== undefined && status !== 'connected' && status !== 'pending';
}

export interface ParsedStream {
  calls: StreamCall[];
  /** The stub's status in the init event; `undefined` when the event listed no servers array. */
  stubStatus: StubStatus | undefined;
  /** Whether the init event offered any stub tool; `undefined` when it listed no tools. */
  toolsOffered: boolean | undefined;
  /** The child compacted its own conversation, so the history it saw is not the one we sent. */
  compacted: boolean;
  /** Input, output and cache tokens of the final result; `undefined` when not reported. */
  tokens: number | undefined;
  sawResult: boolean;
  isError: boolean;
  /** The final result text; only the summary arm uses it. */
  resultText: string;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const count = (value: unknown): number => (typeof value === 'number' && Number.isFinite(value) ? value : 0);

/** Reads the child's `stream-json` lines; lines that are not JSON are skipped. */
export function parseStream(lines: readonly StreamLine[]): ParsedStream {
  const parsed: ParsedStream = {
    calls: [],
    stubStatus: undefined,
    toolsOffered: undefined,
    compacted: false,
    tokens: undefined,
    sawResult: false,
    isError: false,
    resultText: '',
  };
  const seen = new Set<string>();
  for (const line of lines) {
    let event: unknown;
    try {
      event = JSON.parse(line.text);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;

    if (event['type'] === 'system' && event['subtype'] === 'init') {
      if (Array.isArray(event['mcp_servers'])) {
        const server = event['mcp_servers'].find((entry) => isRecord(entry) && entry['name'] === SERVER);
        const status = isRecord(server) ? server['status'] : undefined;
        parsed.stubStatus =
          server === undefined ? 'not listed' : typeof status === 'string' && KNOWN_STATUSES.includes(status) ? (status as StubStatus) : 'unknown';
      }
      if (Array.isArray(event['tools'])) {
        parsed.toolsOffered = event['tools'].some((name) => typeof name === 'string' && name.startsWith(PREFIX));
      }
    } else if (event['type'] === 'system' && event['subtype'] === 'compact_boundary') {
      parsed.compacted = true;
    } else if (event['type'] === 'assistant' && isRecord(event['message']) && Array.isArray(event['message']['content'])) {
      for (const block of event['message']['content']) {
        if (!isRecord(block) || block['type'] !== 'tool_use' || typeof block['name'] !== 'string') continue;
        if (typeof block['id'] === 'string') {
          if (seen.has(block['id'])) continue;
          seen.add(block['id']);
        }
        const name = block['name'];
        parsed.calls.push({
          tool: name.startsWith(PREFIX) ? name.slice(PREFIX.length) : name,
          input: isRecord(block['input']) ? block['input'] : {},
          at: line.at,
        });
      }
    } else if (event['type'] === 'result') {
      parsed.sawResult = true;
      parsed.isError = event['is_error'] === true;
      parsed.resultText = typeof event['result'] === 'string' ? event['result'] : '';
      const usage = event['usage'];
      if (isRecord(usage) && typeof usage['input_tokens'] === 'number' && typeof usage['output_tokens'] === 'number') {
        parsed.tokens =
          count(usage['input_tokens']) +
          count(usage['output_tokens']) +
          count(usage['cache_creation_input_tokens']) +
          count(usage['cache_read_input_tokens']);
      }
    }
  }
  return parsed;
}

export interface ChildRequest {
  args: string[];
  /** Goes to the child's stdin. */
  prompt: string;
  cwd: string;
  timeoutMs: number;
}

/** A failure carries a fixed, non-sensitive reason: never stderr, never output text. */
export type ChildResult = { ok: true; lines: StreamLine[]; code: number | null } | { ok: false; reason: string };

export type ChildRunner = (request: ChildRequest) => Promise<ChildResult>;

const WINDOWS = process.platform === 'win32';

/** Quotes an argument for the Windows shell that starts `claude.cmd`. */
export function quoteArg(arg: string): string {
  return arg === '' || /[\s"]/.test(arg) ? `"${arg.replace(/"/g, '\\"')}"` : arg;
}

/** Kills the child and, on Windows, everything it started (`claude.cmd` runs through a shell). */
function stop(child: ReturnType<typeof spawn>): void {
  if (WINDOWS && child.pid !== undefined) spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
  else child.kill();
}

/**
 * Runs the real `claude` CLI as a child process. A child whose init event says the stub is
 * definitively unavailable is killed at once: it would otherwise spend a whole turn on the
 * flattened history for a result that cannot be used.
 */
export function spawnChild(now: () => number, command = 'claude'): ChildRunner {
  return (request) =>
    new Promise((resolve) => {
      let settled = false;
      const lines: StreamLine[] = [];
      let buffer = '';
      let timer: NodeJS.Timeout | undefined;
      let halted: ChildResult | undefined;
      const finish = (result: ChildResult): void => {
        if (settled) return;
        settled = true;
        if (timer !== undefined) clearTimeout(timer);
        resolve(result);
      };
      const take = (final: boolean): void => {
        const parts = buffer.split('\n');
        buffer = final ? '' : (parts.pop() ?? '');
        for (const text of parts) if (text.trim() !== '') lines.push({ text, at: now() });
        if (halted === undefined && stubFailed(parseStream(lines).stubStatus)) {
          halt({ ok: true, lines, code: null });
        }
      };
      /**
       * Kills the child, then answers once it has really closed, so the stub is gone before the
       * scratch directory is removed; the fallback keeps a child that will not die from
       * blocking the run.
       */
      const halt = (result: ChildResult): void => {
        if (halted !== undefined) return;
        halted = result;
        if (timer !== undefined) clearTimeout(timer);
        timer = setTimeout(() => finish(result), 5000);
        stop(child);
      };

      let child: ReturnType<typeof spawn>;
      try {
        child = spawn(WINDOWS ? quoteArg(command) : command, WINDOWS ? request.args.map(quoteArg) : request.args, {
          cwd: request.cwd,
          shell: WINDOWS,
          stdio: ['pipe', 'pipe', 'ignore'],
          windowsHide: true,
        });
      } catch {
        finish({ ok: false, reason: 'cannot start claude' });
        return;
      }
      timer = setTimeout(() => halt({ ok: false, reason: 'timeout' }), request.timeoutMs);
      child.on('error', () => finish({ ok: false, reason: 'cannot start claude' }));
      child.stdout?.setEncoding('utf8');
      child.stdout?.on('data', (chunk: string) => {
        buffer += chunk;
        take(false);
      });
      child.on('close', (code) => {
        take(true);
        finish(halted ?? { ok: true, lines, code });
      });
      child.stdin?.on('error', () => undefined);
      child.stdin?.end(request.prompt);
    });
}

/** The stub's point and the files the tool may write: only the scratch directory (research R4). */
export interface StubPoint {
  path: string;
  messageIndex: number;
  toolUseId: string;
}

export interface Scratch {
  /** An empty working directory for the child. */
  cwd: string;
  /** Writes the MCP config that starts the stub for this point and returns its path. */
  mcpConfigFor(point: StubPoint): string;
  /** Writes a synthetic file into the scratch directory and returns its path. */
  file(name: string, content: string): string;
  cleanup(): void;
}

/** One scratch directory per run under the system temp directory, removed by `cleanup`. */
export function createScratch(): Scratch {
  const dir = mkdtempSync(join(tmpdir(), 'behaviour-'));
  const cwd = join(dir, 'work');
  mkdirSync(cwd);
  const stub = fileURLToPath(new URL('./stub.ts', import.meta.url));
  let written = 0;
  return {
    cwd,
    mcpConfigFor(point) {
      const path = join(dir, `mcp-${++written}.json`);
      const config = {
        mcpServers: {
          [SERVER]: {
            command: process.execPath,
            args: [
              '--import',
              import.meta.resolve('tsx'),
              stub,
              '--session',
              point.path,
              '--index',
              String(point.messageIndex),
              '--tool-use-id',
              point.toolUseId,
            ],
          },
        },
      };
      writeFileSync(path, JSON.stringify(config));
      return path;
    },
    file(name, content) {
      const path = join(dir, name);
      writeFileSync(path, content);
      return path;
    },
    cleanup() {
      // On Windows the stub may still hold its working directory for a moment after the child
      // exits, so retry. A leftover directory holds only the MCP config (no transcript text),
      // so a failure here must never cost the report.
      try {
        rmSync(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
      } catch {
        // left for the operating system's temp cleanup
      }
    },
  };
}
