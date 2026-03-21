# Mode- and Transport-Independent Core Plan

This plan supersedes the mode-specific parts of [2026-03-21-lightweight-runtime-surface.md](/home/marcel/Programming/coding_assistant/.agent/plans/2026-03-21-lightweight-runtime-surface.md).

## Objective

Make `coding_assistant.runtime` a pure transcript/state-machine library with no knowledge of product run modes (`chat`, `agent`) or transports (CLI, websocket, JSON commands).

Success criteria:

- A host can run `AssistantSession` with explicit messages, tool specs, a completer, and runtime options only.
- `runtime/` contains no `chat`/`agent` mode branches, start-message templates, task framing, or model-selection policy.
- `runtime/` contains no transport mapping or transport-specific payload shapes.
- Product semantics such as chat mode, agent mode, `finish_task`, sub-agent framing, approvals, and default model choice live outside the core runtime.
- Runtime tests read as pure library tests; wrapper and CLI tests cover product behavior separately.
- Public names match their owning layer and responsibility after the split.

## Scope

In scope:

- Remove mode-specific prompt construction and model selection from the core runtime.
- Remove `SessionMode` and other mode-level API from `AssistantSession`.
- Define a transcript-driven session API that operates on explicit messages instead of synthesized chat/agent starts.
- Move managed tool semantics such as `finish_task` out of the core runtime boundary.
- Keep transports as adapters over typed Python events and commands.

Out of scope:

- Implementing a websocket server/transport.
- Redesigning MCP shell/python background task behavior.
- Large user-facing CLI behavior changes beyond what is required by the boundary cleanup.

## Approach

- Treat the core runtime as a generic loop over:
  - transcript/history
  - model completion
  - tool-call requests
  - host-submitted tool results
  - cancellation and terminal states
- Move all product framing into a managed wrapper layer:
  - chat vs agent semantics
  - initial system/task/user message construction
  - model choice such as default vs expert model
  - managed tools such as `finish_task`, compaction, and sub-agent launch
- Keep transport adapters thin. They should translate between transport payloads and the wrapper/core API, not define runtime behavior.
- Prefer tests to use the simplest public API that matches the layer under test rather than reaching into deeper internals unnecessarily.

## Target Core Boundary

Core runtime should own:

- `AssistantSession`
- typed runtime events
- transcript/history state
- model stepping
- tool-call parsing and `tool_call_requested`
- host submission of tool results/errors
- cancellation, failure, and finished state transitions
- optional persistence hooks

Core runtime should not own:

- `chat` vs `agent`
- task templates or instruction templates
- `expert_model` selection
- managed tool identities such as `launch_agent` or `finish_task`
- CLI prompting/rendering
- websocket/JSON command mapping
- MCP startup or tool provisioning

## Target API Shape

Low-level core:

```python
session = AssistantSession(
    tools=tool_specs,
    options=SessionOptions(...),
    completer=completer,
    history_store=history_store,
)

await session.start(
    history=[
        SystemMessage(content=instructions),
        UserMessage(content="..."),
    ],
    model="...",
)
```

Managed wrapper:

- builds the initial transcript for chat or agent flows
- decides which model to use
- injects managed tools
- handles `tool_call_requested`
- maps managed tool calls to runtime control actions

## Naming Targets

The final names should reflect the new architecture rather than the current transitional one.

- Keep `AssistantSession` as the low-level core session type.
- Keep `ToolSpec` and `ToolCallRequestedEvent`.
- Remove `SessionMode` from the codebase.
- Rename `WaitingForUserEvent` to a transport-neutral name such as `InputRequestedEvent` or `HostInputRequestedEvent`.
- Remove or replace `FinishedEvent` in the core if completion becomes a wrapper-managed concept rather than a runtime-owned one.
- Rename `AgentRunner` to something that reflects a managed wrapper over the core rather than an agent-only abstraction.
- Move or rename `runtime/builtin_tools.py` once those tools are no longer runtime-owned.
- Keep wrapper-side config and bundle names aligned with the wrapper role instead of calling everything a generic "session" or "runner" without distinction.

## Milestones / PR Sequence

### Milestone 1 / PR 1: Define a Transcript-Driven Core Contract

- [x] Replace `start(mode=..., task=..., instructions=...)` with a mode-free API based on explicit history/transcript input.
- [x] Remove `SessionMode` from the core runtime public surface.
- [x] Remove `expert_model` from `SessionOptions`; keep only runtime-level tuning in the core.
- [x] Keep finish/compaction as temporary runtime-owned control points for now so the mode-free core can land cleanly before managed-tool migration.

Verification:

- A low-level test can start a session from explicit `SystemMessage`/`UserMessage` history with no mode or task arguments.
- No public runtime API requires the caller to mention `chat` or `agent`.

### Milestone 2 / PR 2: Remove Mode Semantics from `runtime/`

- [x] Move chat/agent prompt templates out of [`engine.py`](/home/marcel/Programming/coding_assistant/src/coding_assistant/runtime/engine.py).
- [x] Remove mode-based model selection from [`session.py`](/home/marcel/Programming/coding_assistant/src/coding_assistant/runtime/session.py).
- [x] Remove mode-based builtin-tool selection from the core runtime.
- [x] Keep `engine.py` limited to generic completion, tool-call parsing, and progress emission helpers.

Verification:

- `rg -n "chat|agent|SessionMode" src/coding_assistant/runtime` only finds neutral comments/tests or returns nothing relevant.
- The runtime package can be explained without mentioning chat mode or agent mode.

### Milestone 3 / PR 3: Move Managed Tool Semantics to the Wrapper

- [ ] Make `finish_task`, sub-agent launch, and other product-shaped tools wrapper-managed rather than runtime-managed.
- [ ] Let the wrapper translate those managed tool calls into runtime control actions or terminal outcomes.
- [ ] Keep the core tool protocol generic: request, allow/deny/execute outside, submit result back.

Verification:

- The core runtime no longer reserves product tool names.
- Wrapper tests cover finish/sub-agent flows without those concepts leaking back into the core.

### Milestone 4 / PR 4: Rebuild the Wrapper Around the New Core

- [ ] Make the wrapper own chat-mode startup behavior.
- [ ] Make the wrapper own agent-mode task framing and model choice.
- [ ] Keep sub-agent orchestration in the wrapper layer only.
- [ ] Preserve current CLI-visible behavior through wrapper logic rather than runtime branching.

Verification:

- The CLI still supports both chat and agent workflows.
- The wrapper, not the core runtime, is the only place that knows what those workflows mean.

### Milestone 5 / PR 5: Tighten Adapter Boundaries and Tests

- [ ] Keep CLI and websocket mapping code outside `runtime/`.
- [ ] Rewrite runtime tests around explicit transcripts and generic tool-call events.
- [ ] Rewrite wrapper tests around chat/agent semantics and managed tools.
- [ ] Remove legacy tests or helpers that still assume mode-aware runtime behavior.
- [ ] Apply the agreed naming cleanup so core, wrapper, and adapter names read consistently.
- [ ] Make tests use the simplest applicable public API by default:
  - runtime tests should prefer `coding_assistant.runtime` public types
  - managed-wrapper tests should prefer the wrapper public API
  - package-level embedding tests should prefer `from coding_assistant import ...`
  - only keep direct internal-module tests where the module itself is the unit under test

Verification:

- `just test` passes.
- `just lint` passes.
- The runtime test suite has no product-mode fixtures or CLI assumptions.
- Public class and event names can be explained without relying on deprecated architecture terms.
- Tests that can reasonably target the simple embedding API do so, instead of importing deeper modules without need.

## Risks

- The biggest design risk is managed completion semantics: once `finish_task` leaves the core, the runtime needs a clean generic way to transition to `FinishedEvent` without reintroducing product naming.
- A second risk is keeping too much convenience in `AssistantSession.start(...)`; if it still synthesizes product prompts, the old coupling will remain.
- A third risk is allowing wrapper concerns to leak back through reserved tool names or mode-shaped helper functions inside `runtime/`.

## Recommended Next Milestone

- [ ] Continue with Milestone 3: move `finish_task` and `compact_conversation` out of the core runtime and let the wrapper translate those tool calls into runtime control actions.
