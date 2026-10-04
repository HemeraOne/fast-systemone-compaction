import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import type { Message } from '../../src/index.js';
import { parseTranscript } from '../replay/transcript.js';
import { familyOf, pointContext } from './history.js';

/** The most lookups a point may issue before the test ends the conversation. */
export const MAX_LOOKUPS = 5;

export const NOT_AVAILABLE = 'not available in this test';
export const STOP = 'The test ends here. Reply with the single word: done';

const tool = (name: string, description: string, properties: Record<string, string>, required: string[]) => ({
  name,
  description,
  inputSchema: {
    type: 'object',
    properties: Object.fromEntries(Object.entries(properties).map(([key, text]) => [key, { type: 'string', description: text }])),
    required,
  },
});

/** The five stub tools; their results come from recorded history, never from real files. */
export const STUB_TOOLS = [
  tool('Read', 'Read a file.', { file_path: 'Path of the file' }, ['file_path']),
  tool('Grep', 'Search file contents.', { pattern: 'Pattern to search for', path: 'Where to search' }, ['pattern']),
  tool('Glob', 'Find files by name pattern.', { pattern: 'Glob pattern' }, ['pattern']),
  tool(
    'Edit',
    'Replace text in a file.',
    { file_path: 'Path of the file', old_string: 'Text to replace', new_string: 'Replacement text' },
    ['file_path', 'old_string', 'new_string'],
  ),
  tool('Bash', 'Run a shell command.', { command: 'The command' }, ['command']),
];

export interface RpcRequest {
  id?: number | string | null;
  method?: string;
  params?: Record<string, unknown>;
}

export interface RpcResponse {
  jsonrpc: '2.0';
  id: number | string | null;
  result?: unknown;
  error?: { code: number; message: string };
}

const text = (value: string, isError = false) => ({ content: [{ type: 'text', text: value }], isError });

/**
 * A stub MCP server for one point: lookups are answered from the uncompacted history before
 * the recorded step, the recorded step's own tool family (the terminal action) and every
 * lookup past the limit are answered with the stop reply. Nothing is read or written.
 */
export function createStub(point: { messages: readonly Message[]; messageIndex: number; toolUseId: string }) {
  const context = pointContext(point);
  let lookups = 0;

  const call = (name: string, input: Record<string, unknown>) => {
    if (familyOf(name) === familyOf(context.step.tool)) return text(STOP);
    lookups++;
    if (lookups > MAX_LOOKUPS) return text(STOP);
    const served = context.lookup.serve({ tool: name, input });
    return served === undefined ? text(NOT_AVAILABLE, true) : text(served);
  };

  return {
    /** The reply to one JSON-RPC message, or `undefined` for a notification. */
    handle(request: RpcRequest): RpcResponse | undefined {
      if (request.id === undefined || request.id === null) return undefined;
      const reply = (result: unknown): RpcResponse => ({ jsonrpc: '2.0', id: request.id!, result });
      switch (request.method) {
        case 'initialize': {
          const version = request.params?.['protocolVersion'];
          return reply({
            protocolVersion: typeof version === 'string' ? version : '2024-11-05',
            capabilities: { tools: {} },
            serverInfo: { name: 'stub', version: '0.0.0' },
          });
        }
        case 'ping':
          return reply({});
        case 'tools/list':
          return reply({ tools: STUB_TOOLS });
        case 'tools/call': {
          const name = request.params?.['name'];
          const args = request.params?.['arguments'];
          const input = typeof args === 'object' && args !== null ? (args as Record<string, unknown>) : {};
          return reply(typeof name === 'string' ? call(name, input) : text(NOT_AVAILABLE, true));
        }
        default:
          return { jsonrpc: '2.0', id: request.id, error: { code: -32601, message: 'method not found' } };
      }
    },
  };
}

function flag(args: readonly string[], name: string): string | undefined {
  const at = args.indexOf(name);
  return at >= 0 ? args[at + 1] : undefined;
}

/** Serves one point over stdio. Prints protocol messages only, never transcript text. */
function main(args: readonly string[]): void {
  const session = flag(args, '--session');
  const index = Number(flag(args, '--index'));
  const toolUseId = flag(args, '--tool-use-id');
  if (session === undefined || toolUseId === undefined || !Number.isInteger(index)) {
    process.stderr.write('stub: missing arguments\n');
    process.exit(1);
  }
  const { messages } = parseTranscript(readFileSync(session, 'utf8'));
  const stub = createStub({ messages, messageIndex: index, toolUseId });
  createInterface({ input: process.stdin }).on('line', (line) => {
    let request: RpcRequest;
    try {
      request = JSON.parse(line) as RpcRequest;
    } catch {
      return;
    }
    const response = stub.handle(request);
    if (response !== undefined) process.stdout.write(`${JSON.stringify(response)}\n`);
  });
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main(process.argv.slice(2));
}
