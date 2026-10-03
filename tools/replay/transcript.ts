import type { Message, ToolResult, ToolUse } from '../../src/index.js';

type Block = Record<string, unknown>;

function isRecord(value: unknown): value is Block {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function joinText(blocks: readonly Block[]): string {
  return blocks.map((block) => (block['type'] === 'text' && typeof block['text'] === 'string' ? block['text'] : '')).join('');
}

function resultText(content: unknown): string {
  if (typeof content === 'string') return content;
  return Array.isArray(content) ? joinText(content.filter(isRecord)) : '';
}

/**
 * Turns the text of a Claude Code session file (one JSON entry per line) into the
 * library's `Message` shape. Sidechain (subagent) entries and non-message entries are
 * ignored; lines that are not valid JSON are skipped and counted.
 */
export function parseTranscript(jsonlText: string): { messages: Message[]; malformedLines: number } {
  const messages: Message[] = [];
  let malformedLines = 0;
  let lastAssistantId: string | undefined;

  for (const raw of jsonlText.split('\n')) {
    if (raw.trim() === '') continue;
    let entry: unknown;
    try {
      entry = JSON.parse(raw);
    } catch {
      malformedLines++;
      continue;
    }
    if (!isRecord(entry) || entry['isSidechain'] === true) continue;
    const type = entry['type'];
    const message = entry['message'];
    if ((type !== 'user' && type !== 'assistant') || !isRecord(message)) continue;

    const content = message['content'];
    const blocks = Array.isArray(content) ? content.filter(isRecord) : [];

    if (type === 'assistant') {
      const text = joinText(blocks);
      const toolUses: ToolUse[] = blocks
        .filter((block) => block['type'] === 'tool_use')
        .map((block) => ({
          tool_use_id: String(block['id']),
          tool: String(block['name']),
          input: isRecord(block['input']) ? block['input'] : {},
        }));
      const id = typeof message['id'] === 'string' ? message['id'] : undefined;
      const previous = messages[messages.length - 1];
      if (id !== undefined && id === lastAssistantId && previous?.role === 'assistant') {
        previous.text += text;
        previous.toolUses.push(...toolUses);
      } else {
        messages.push({ role: 'assistant', text, toolUses });
      }
      lastAssistantId = id;
    } else {
      lastAssistantId = undefined;
      const toolResults: ToolResult[] = blocks
        .filter((block) => block['type'] === 'tool_result')
        .map((block) => ({
          tool_use_id: String(block['tool_use_id']),
          text: resultText(block['content']),
          isError: block['is_error'] === true,
        }));
      const text = typeof content === 'string' ? content : joinText(blocks);
      messages.push(
        toolResults.length > 0
          ? { role: 'user', text, toolUses: [], toolResults }
          : { role: 'user', text, toolUses: [] },
      );
    }
  }
  return { messages, malformedLines };
}
