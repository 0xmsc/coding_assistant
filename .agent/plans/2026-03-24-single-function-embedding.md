# Single-Function Embedding Plan

## Objective

- Replace the session-first embedding design with a simpler core architecture centered on one public async function: `run_agent(...)`.
- Make caller-owned `history` the primary state boundary.
- Remove mixed startup modes such as `history` plus `initial_user_message` from the core API.
- Remove session-era abstractions that are no longer justified by a history-driven core.
- Keep transcript construction outside the core loop instead of passing instructions separately.

## Approach

- Introduce a new core module whose primary primitive is:
  - `run_agent(history=..., model=..., tools=..., ...)`
- Make `run_agent(...)` the real agent loop implementation, not a wrapper around `ManagedSession`.
- Keep the loop direct and transcript-driven:
  - read history
  - call the model
  - stream optional deltas/events to callbacks
  - append assistant messages
  - execute tool calls
  - append tool results
  - continue until the agent yields for user input or fails
- Express transcript rewriting as pure helpers such as `compact_history(history, summary)` rather than mutable session methods.
- Implement sub-agents by recursively calling `run_agent(...)`.
- Keep persistence outside the core function. `run_agent(...)` returns updated history; callers decide whether and how to save it.
- Keep streaming minimal. The first version should expose at most a narrow text callback such as `on_delta`, not a general event bus.
- Keep cancellation simple. Normal task cancellation should propagate instead of being modeled as a returned state.

## Target Architecture

- Core:
  - `run_agent(...)`
  - `AgentRunResult`
  - pure helpers such as `compact_history(...)`
- Types:
  - messages
  - tool protocol
- Adapters:
  - CLI and defaults call `run_agent(...)`
  - any remaining session-style APIs become thin adapters over the function or are removed if they no longer justify themselves

## Post-Refactor Outline

```text
src/coding_assistant/
  __init__.py              # exports run_agent, AgentRunResult, compact_history
  agent.py                 # core run_agent loop and built-in tool dispatch
  agent_types.py           # AgentRunResult and status enum/literals
  history.py               # compact_history and transcript helpers
  defaults.py              # build default tools/instructions/history store; no session objects
  adapters/cli.py          # CLI loop that repeatedly calls run_agent(history=...)
  llm/
    openai.py
    types.py               # message/tool protocols only
  tools/
    builtin.py             # launch_agent and redirect_tool_call
    mcp.py
```

- Files expected to be removed entirely:
  - `src/coding_assistant/managed_session.py`
  - `src/coding_assistant/runtime/assistant_session.py`
  - `src/coding_assistant/runtime/events.py`
  - `src/coding_assistant/runtime/__init__.py`
  - `src/coding_assistant/tool_results.py`
- The `runtime/` package should disappear unless a concrete non-session runtime helper remains.

## Scope

- In scope:
  - new core `run_agent(...)` implementation
  - top-level export and README examples built around `history` in and `history` out
  - compaction moved to a pure history transform
  - `launch_agent` reworked to call `run_agent(...)` recursively
  - migration of current internal callers to the new core path
- Out of scope for the first milestone:
  - redesigning tool schemas
  - changing provider integrations
  - broad CLI UX changes unrelated to the new core

## Features To Remove Now

- Remove `ManagedSession` and `AssistantSession` as the primary embedding surfaces.
- Remove session lifecycle methods and state-machine APIs:
  - `start`
  - `send_user_message`
  - `submit_tool_result`
  - `submit_tool_error`
  - `next_event`
  - `events`
  - `cancel`
- Remove the event queue model and `SessionEvent` classes.
- Remove `replace_history` and any other generic transcript mutation entry point.
- Remove mixed startup modes from the core API, especially `initial_user_message`.
- Remove history persistence from the core API surface. `HistoryStore` stays as an external concern.
- Remove `ToolResult`, `TextResult`, `CompactConversationResult`, and `normalize_tool_result` if tools can simply return strings.
- Remove built-in tool toggles such as `include_launch_agent` and `include_redirect_tool_call` unless a concrete caller still needs that configurability after the refactor.
- Remove the websocket command/event adapter built around the deleted session protocol.
- Remove top-level exports of `ManagedSession` and `AssistantSession`.

## Deferred Re-Adds If Needed

- A low-level host-driven tool loop for embedders that truly need manual tool execution.
- A session object with pause/resume semantics.
- A typed streaming event API beyond simple text callbacks.
- A dedicated cancellation API or returned cancellation status.
- A websocket adapter if a concrete remote embedding protocol is needed after the function-first core lands.
- A convenience helper for building initial histories from system/user inputs.
- Rich tool result objects beyond plain string output.
- Selective enabling/disabling of built-in management tools if a real caller needs it after the simpler core exists.

## API Shape

- Input:
  - `history`
  - `model`
  - `tools`
  - optional `expert_model`
  - optional `tool_policy`
  - optional `on_delta`
- Output:
  - `history`
  - `status` such as `awaiting_user` or `failed`
  - optional `error`

- Explicitly not in the core API:
  - `instructions`
  - `initial_user_message`
  - `history_store`
  - event iterators
  - manual tool result submission
  - explicit cancellation commands or cancellation result objects

## Steps

- [x] Add a new core module for `run_agent(...)`, its result type, and pure history helpers.
- [x] Implement the managed agent loop directly in `run_agent(...)` using caller-provided history as the only transcript input.
- [x] Simplify the tool contract to plain string outputs if no remaining structured tool result use case survives the refactor.
- [x] Move conversation compaction to a pure helper and remove generic transcript replacement from the runtime path.
- [x] Rework `launch_agent` to invoke `run_agent(...)` recursively instead of spinning up a nested managed session.
- [x] Migrate the CLI to repeatedly call `run_agent(...)` and prompt when the returned status is `awaiting_user`.
- [x] Replace session-returning defaults/helpers with input/dependency builders for `run_agent(...)`.
- [x] Remove session modules, runtime event modules, and obsolete exports with no compatibility shims.
- [x] Update top-level exports and README examples to recommend `run_agent(...)` as the only primary embedding API.

## Verification

- [x] Add focused tests for `run_agent(...)` covering:
  - normal assistant reply
  - tool execution loop
  - input handoff with updated history
  - failure propagation
  - recursive sub-agent execution
  - compaction behavior via pure history transforms
- [x] Add focused tests proving removed abstractions are no longer required:
  - no session imports in public examples
  - no structured tool result requirement for normal tool execution
- [x] `just test`
- [x] `just lint`

## Risks

- Doing a direct architectural inversion is more disruptive than adding a wrapper, so migration order matters.
- The CLI currently benefits from streaming; the replacement callback must stay narrow and not recreate a hidden event framework.
- Removing `ToolResult` may touch multiple tests and helper layers, but it is justified if plain strings cover all surviving tool behavior.
- If some host genuinely needs pause/resume semantics beyond transcript ownership, that should be justified after the simpler core exists, not assumed up front.
