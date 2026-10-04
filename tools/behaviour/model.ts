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
    '--safe-mode',
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

export interface ParsedStream {
  calls: StreamCall[];
  /** Whether the init event listed the stub as connected; `undefined` when it listed no servers. */
  stubConnected: boolean | undefined;
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
    stubConnected: undefined,
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
        parsed.stubConnected = event['mcp_servers'].some(
          (server) => isRecord(server) && server['name'] === SERVER && server['status'] === 'connected',
        );
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

/** Runs the real `claude` CLI as a child process. */
export function spawnChild(now: () => number): ChildRunner {
  return (request) =>
    new Promise((resolve) => {
      let settled = false;
      const lines: StreamLine[] = [];
      let buffer = '';
      let timer: NodeJS.Timeout | undefined;
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
      };

      let child: ReturnType<typeof spawn>;
      try {
        child = spawn('claude', WINDOWS ? request.args.map(quoteArg) : request.args, {
          cwd: request.cwd,
          shell: WINDOWS,
          stdio: ['pipe', 'pipe', 'ignore'],
          windowsHide: true,
        });
      } catch {
        finish({ ok: false, reason: 'cannot start claude' });
        return;
      }
      timer = setTimeout(() => {
        if (WINDOWS && child.pid !== undefined) spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
        else child.kill();
        finish({ ok: false, reason: 'timeout' });
      }, request.timeoutMs);
      child.on('error', () => finish({ ok: false, reason: 'cannot start claude' }));
      child.stdout?.setEncoding('utf8');
      child.stdout?.on('data', (chunk: string) => {
        buffer += chunk;
        take(false);
      });
      child.on('close', (code) => {
        take(true);
        finish({ ok: true, lines, code });
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
