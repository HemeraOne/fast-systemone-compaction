# fast-systemone-compaction

Claude Code plugin that replaces the compaction summary with verbatim pruning:
a System One (Jev) model scores every tool call and result in one fast request,
or fixed rules shorten large old results and drop superseded file reads with no
model at all. Stale items are dropped or truncated, everything kept stays
verbatim. Also usable as an npm library.

## What and why

Most context compaction asks an LLM to summarize old turns. A summary is
lossy: a file path, exact error, constraint, or command can disappear even when
it matters later. This library never rewrites anything. It only deletes tool
calls and tool results Jev says are no longer needed, and it asks Jev while
showing it the whole conversation. User and assistant text stays verbatim and
in order.

The repository is both an npm package (`src/`) and a Claude Code plugin
(`hooks/`, `.claude-plugin/`) that uses the package to replace Claude Code's
built-in compaction summary with the original messages.

## How it works

1. Every `tool_use` is paired with its `tool_result` by `tool_use_id`. Calls in
   the first message or in the newest `preserveRecentMessages` messages are
   pinned and never touched.
2. The **state** sent to Jev is the whole conversation so far, oldest first,
   with every tool result replaced by a short note (`ok, 4213 chars (omitted)`).
   Tool inputs are included, texts are included, nothing is summarized.
3. The state is fitted into `maxStateTokens` (25k by default) in stages, each
   applied only if the previous one was not enough: tool inputs truncated to
   1000, then 200, then 60 characters; long texts abridged to head + tail,
   oldest non-pinned messages first; old non-pinned messages collapsed to a
   `[… N chars omitted …]` note; old tool calls reduced to one line each
   (`t12 Read file_path=src/a.ts → ok 480ch`); old call-less messages left
   out; runs of old call-only messages folded into one entry. If it still
   does not fit, compaction throws. Tokens are estimated without a tokenizer (a
   word per six letters, half a token per digit, ~one per other symbol),
   calibrated to land a little above the counts Jev reports.
4. For every non-pinned call Jev gets two `noul` questions: should the **call**
   stay (knowing it was made, with its input, still matters), and should the
   **result** stay verbatim (its contents are still needed and re-running the
   tool would not do).
5. Questions are split into as many requests as needed so state plus questions
   stays under `maxRequestTokens` (30k by default, under Jev's 32k request
   limit). The same full state is resent with every request; requests run
   concurrently and their answers are merged.
6. Decisions per call, against `keepThreshold`:
   - `keepResult ≥ threshold` → keep call and result;
   - else `keepCall ≥ threshold` → keep the call, truncate the result to
     `truncateHeadChars` characters (half from its start, half from its end)
     with a one-line note between them;
   - else → remove the call together with its result.
7. The message list is rebuilt: a message that loses all its content is
   removed, untouched messages are returned as the same objects, and no result
   is ever left without its call.

Jev failures, malformed answers, a missing key, or a history that cannot be
fitted throw; the caller (or the Claude Code hook) decides what to fall back to.

## Install and usage

```sh
npm install fast-systemone-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-systemone-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Never edit src/generated.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: '…file…' }] },
  // …
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session transcript
can be passed in as is.

To bring your own transport, implement `JevAsker` (one `ask(state, questions)`
method) and call `compact(messages, asker, options)`; `buildJevRequest` and
`parseJevResponse` give you the HTTP request body and response validation.
The building blocks (`collectToolCalls`, `fitState`, `batchCalls`,
`decideCall`, `applyDecisions`) are exported too.

`apiKey` defaults to `process.env.TYPESAFE_API_KEY`. Never commit the key or
put it in a source file.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `apiKey` | `TYPESAFE_API_KEY` | TypeSafe API key (`compactMessages`/`JevClient`) |
| `model` | `jev-latest` | Jev model name |
| `baseUrl` | `https://api.typesafe.ai/v1/systemone` | System One endpoint |
| `fetch` | native `fetch` | Injectable fetch implementation for tests |
| `goal` | last 3 user prompts | Ongoing task description included in the state |
| `keepThreshold` | `0.5` | Minimum keep probability for a call or result to stay |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `maxStateTokens` | `25000` | Estimated token ceiling for the state |
| `maxRequestTokens` | `30000` | Estimated ceiling for state plus one batch of questions |
| `truncateHeadChars` | `300` | Characters of a dropped tool result retained, split between its start and end around the note |

`result.stats` reports message and character counts before and after, the
per-reason decision counts, the state size in estimated tokens, which fitting
stage was needed, and the number of requests.

## Limitations

- Only tool calls and results are candidates; text messages are never removed
  or shortened in the output (they are only abridged in the state Jev sees).
- Token sizes are estimates from character counts, not a tokenizer.
- Calibration is at the request level; a probability is not a proof that a
  result is safe to delete. The assistant can always re-run the tool.
- The full state is repeated with every request, so a history near the state
  ceiling costs one request per handful of questions.

## Claude Code plugin

The repository root is a Claude Code function-hook plugin: `hooks/fast-jev.ts`
is a thin adapter that feeds `session.compact` transcripts through `src/` and
falls back to Claude Code's built-in summary on errors or insufficient
reduction. See [`hooks/README.md`](hooks/README.md) for configuration and the
Claude Code 2.1.274 type reference.

### Install in Claude Code

Function hooks are an early-access Claude Code feature (2.1.274+), so the
opt-in flag must be set wherever Claude Code runs, e.g. in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1", "TYPESAFE_API_KEY": "<your key>" } }
```

Then add this repository as a plugin marketplace and install the plugin,
either from the shell or as slash commands inside a session:

```sh
claude plugin marketplace add considerITman/fast-systemone-compaction
claude plugin install fast-systemone-compaction@fast-systemone-compaction
```

The install prompts for the plugin options (API key, thresholds, `truncateHeadChars`,
…); leave them at their defaults to use `TYPESAFE_API_KEY` from the environment.
Restart Claude Code or run `/reload-plugins`. From then on `/compact` (and
auto-compaction) goes through Jev: the toast reads
`fast-systemone-compaction: kept N/M messages, no summary (…)` when the pruned history
replaced the built-in summary, or `fallback to built-in summary (…)` when Jev
could not remove enough (short sessions, or when it fails).

To run from a checkout without installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`
from the repository root. No publishing step is required; the marketplace is
just the repo's `.claude-plugin/marketplace.json`.

### Rule-based mode (no model, no network)

Set the `compactionMode` plugin option to `rules` to compact without any
backend: large old tool results (over 2,000 characters) are cut to a head and a
tail around a note (except reads of a file that a later edit or write follows,
which are kept whole), and file reads that a later full read or write of the
same file supersedes are removed (a partial read only when a later full read
exists).
No key or endpoint is needed and no request is made. Unset keeps the System
One mode; an unknown value falls back to the built-in summary. The rule set
is validated for size and for lost information (see Safety replay below), not
for effect on later answers, and is the baseline a model-based mode must beat.
The library function is `compactByRules(messages, options)`. Details, including the deferred
failed-command rule, are in [`hooks/README.md`](hooks/README.md#rule-based-mode-no-model-no-network).

### Self-hosted / local Laya

Set the `baseUrl` plugin option to point compaction requests at any
Jev-compatible endpoint instead of TypeSafe, e.g. a local Laya server at
`http://127.0.0.1:8000/v1/systemone` with `model=typed-decisions`,
`apiKey=local`, and small budgets (`maxStateTokens=650`,
`maxRequestTokens=950` — Laya's first model has a 1,024-token context).
Leaving `baseUrl` unset keeps the existing TypeSafe behaviour. The Laya setup
is experimental: with its current small model most sessions fall back to the
built-in summary. See
[`hooks/README.md`](hooks/README.md#self-hosted--local-laya) for the start
command, the full request/response contract, and what to expect from a small
local model.

## Development

```sh
npm install
npm run typecheck        # library + hook + tools
npm test
npm run build
npm run validate:plugin  # claude plugin validate
TYPESAFE_API_KEY="$(cat ~/.typesafe_key)" npm run demo
```

The unit tests use a fake Jev and never contact TypeSafe. The demo is the live
network check.

### Safety replay

`npm run replay` checks the rule-based mode against your own past Claude Code
sessions (default `~/.claude/projects`, or `-- --root <dir>`; `-- --before
<date>` keeps only session files last modified before that date, 00:00 UTC, to
approximate an earlier corpus). At points in each
session it compacts the history before a tool call with `compactByRules` and
counts how often a value that call used (a file path, a command, the text an
edit targets) occurred earlier but is gone after compaction, per rule. It runs
offline with no model or key, prints counts, rates, and locators but never
transcript text, and gives the same report on every run. A lost value is an
upper bound on harm, not proof of it. The code lives in `tools/replay/` and is
not part of the plugin or the published package.

### Behaviour test

`npm run behaviour -- --model <alias-or-id> --max-points <n> --token-cap <n>`
asks what the assistant actually does at the points where the replay found a
lost value. It runs the `claude` CLI in print mode on your own login (no API
key), so it uses plan usage and shares the sampled session history with the
model provider; it runs only when all three flags are given, and stops starting
new points once the tokens the CLI reports reach the cap. For each point it runs
the compacted history and the uncompacted one (the control) in an isolated
child (user settings, hooks and plugins skipped, no built-in tools, no saved session), lets the model take
its next step with five stub tools served by a small MCP server from the
recorded history, and compares that step with the recorded one: `same` (right
at once), `recovered` (right after extra lookups), `wrong` (a different action),
`gave-up` (no tool call, stopping after lookups, or too many lookups, each shown
as its own count), `unreachable` (the lost value could not be recovered by any
stub, and the action was not right), `failed` (the child did not run; left out
of the rates, with a fixed reason in the report). The report lists counts, extra
lookups, the time to a final action, the control's own deviation and the extra
effort of the compacted arm over the control (median extra time and lookups over
the points where both arms reached a final action), never transcript text; the only files written are in a temporary
scratch directory that is removed at the end. `-- --summary` adds an approximate
third arm built from a model-written summary; it does not re-attach recently
read files as the built-in summary does. Caveats: the history reaches the model
as text rather than as real tool turns, a stand-in system prompt and tool set
replace Claude Code's own, lookups match only exact repeats (a `Read` matches on
path), a point whose recorded step is one of several parallel calls of its kind is
skipped and counted, the point count is small, and the CLI details the tool relies on are
checked only against the real CLI: `-- --check --model <id>` runs one tiny
synthetic prompt (a few thousand tokens, no session text) through the same child
and stub and says whether the model could use a stub tool (on a problem it also
prints what the child itself reported, which is safe because the prompt is
synthetic), so run it before a real point, which costs about 250k tokens per arm. The code lives in `tools/behaviour/`, not in the plugin or the
published package.

## Animated demo (macOS)

`demo/JevDemo` is a small native SwiftUI app that plays a scripted, dramatized
version of the compaction flow inside a Claude Code-style terminal: the tool
calls of a canned transcript are scored, results and calls Jev lets go turn red
and collapse away, and the rest stays verbatim. It never calls the API; it
exists to be screen recorded.

```sh
demo/JevDemo/build.sh   # builds demo/JevDemo/build/JevDemo.app and launches it
```

Press space in the app to replay from the start.
