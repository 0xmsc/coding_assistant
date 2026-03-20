# Runtime Refactor Plan

## Objective

Refactor the project from a CLI-first architecture into a transport-agnostic async session runtime that lives inside this package, can be embedded from Python code, can drive a CLI, and can later be exposed over websockets. The goal is a clean runtime boundary inside one repo/package, not a separate repo or distribution split. The first implementation should be deliberately small: one public session object, one runtime-owned agent loop, one host interaction boundary via normal user messages, internal non-interactive tools only, incremental assistant output via delta messages, and terminal completion/failure/cancellation events. This is a clean-break refactor: no permanent compatibility layers, no attempt to preserve the current callback-based control model, and broad test rewrites are expected.

## Scope

In scope:

- Replace the current `Session`/`UI`/callback control model with an async runtime API.
- Make user interaction and cancellation explicit runtime commands instead of callbacks.
- Keep read-only progress streaming separate from control flow.
- Rebuild the CLI on top of the new runtime instead of letting the runtime depend on terminal UI abstractions.
- Introduce a host-friendly API for embedding in Python programs.
- Prepare the runtime boundary required for websocket adapters and future REST wrapping.
- Rewrite tests to target the new runtime contract rather than the current implementation details.

Out of scope for the first implementation pass:

- Backward compatibility with the current `Session`, `UI`, `ProgressCallbacks`, or `ToolCallbacks` APIs.
- Adapters that preserve old CLI internals.
- Partial compatibility shims for existing tests.
- Splitting the project into separate repositories or separately published packages.
- Approvals or host-executed tool calls in the first implementation pass.
- Reasoning events in the first implementation pass.
- REST transport in the first implementation pass.
- Worker-process isolation in the first implementation pass.

## Target Design

### Core Principles

- The runtime owns agent execution.
- The host owns external decisions.
- Read-only observation is separate from runtime control.
- Tools are runtime-owned and non-interactive in the first implementation.
- The public API should use the minimum number of core concepts.

### Runtime Shape

- `AssistantSession`: the main public runtime object.
- `SessionOptions`: configuration for a session.
- `SessionEvent`: outbound event type for observation and adapters.

### Package Boundary

- Keep one repository and one Python package for this refactor.
- Put the reusable runtime under `coding_assistant.runtime`.
- Make the CLI import and drive the runtime rather than letting the runtime depend on CLI abstractions.
- Treat any future package split as a follow-up decision, not part of this refactor.

### Control Model

The runtime should expose commands such as:

- `start(...)`
- `send_user_message(...)`
- `cancel()`

The runtime owns the full agent loop internally:

- model step
- internal tool execution
- message/history updates
- next model step

The host only re-enters control when:

- it starts a run
- it sends a normal user message after the runtime asks for one
- it cancels a run
- it consumes terminal events

### Observation Model

Read-only progress should be exposed through one async event stream in the core API. Adapters may wrap that stream in callbacks or transport-specific messaging, but the runtime itself should only have one observation model. In V1, observation should stay minimal but must support incremental assistant output. The initial event model should therefore include assistant content deltas, completed assistant messages, and terminal state changes.

Observation must never influence runtime control flow.

### Tool Ownership

For the first implementation, all tools should be runtime-owned and non-interactive:

- tools execute entirely inside the runtime
- tools do not ask the host for input
- tools do not return control to the host

If a tool hits ambiguity or missing information, it should fail or return a result that leads the agent to ask the user through the normal runtime-owned conversation flow.

This means there is no host-facing tool-result submission API in V1.

### Sub-Agent Model

Sub-agents should be implemented as runtime-owned child sessions rather than as separate host-managed public objects.

- The host interacts only with the top-level `AssistantSession` in V1.
- When the parent agent invokes `launch_agent`, the runtime creates a child session that uses the same runtime contract internally.
- Child sessions have their own history, state, and cancellation scope.
- Child sessions return structured results to the parent runtime, which then continues the parent conversation.
- Child sessions do not directly surface `waiting_for_user` to the host in V1.
- If a child lacks information, it should return that fact to the parent; the parent may then ask the user through the normal top-level host boundary.
- Cancellation cascades from parent session to all child sessions.
- In V1, child-session progress does not need a separate public event stream; the public contract only needs the parent-visible result. If deeper observability is needed later, add session metadata to events rather than exposing a second control model.

### Minimal V1 Event Set

The first implementation should keep the outbound message set deliberately small:

- `assistant_delta`
- `assistant_message`
- `waiting_for_user`
- `finished`
- `failed`
- `cancelled`

Only `waiting_for_user` requires host action. All other events are observational or terminal.

Additional progress events such as reasoning deltas or tool lifecycle events can be added later if needed without changing the basic control model.

### Sandbox Boundary

Sandboxing must not remain a process-global side effect of session startup. In V1, the only requirement is that sandboxing and process isolation are not designed into the core runtime API. They can be reintroduced later at an adapter or worker boundary if needed.

## Proposed Module Restructure

- `src/coding_assistant/runtime/`
- `src/coding_assistant/runtime/session.py`
- `src/coding_assistant/runtime/events.py`
- `src/coding_assistant/runtime/engine.py`
- `src/coding_assistant/runtime/persistence.py`
- `src/coding_assistant/adapters/cli.py`
- `src/coding_assistant/adapters/websocket.py`

The current CLI-facing modules should be reduced or removed as part of the migration rather than wrapped.

## Milestones / PR Sequence

### Milestone 1 / PR 1: Define the Runtime Contract

- [x] Define the minimal public runtime types: `AssistantSession`, `SessionOptions`, and `SessionEvent`.
- [x] Define the minimal V1 event set: `assistant_delta`, `assistant_message`, `waiting_for_user`, `finished`, `failed`, `cancelled`.
- [x] Define the only V1 host boundary: `waiting_for_user` followed by `send_user_message(...)`.
- [x] Define V1 tool semantics: runtime-owned, non-interactive, no host-executed tool path.
- [x] Define V1 sub-agent semantics: runtime-owned child sessions, top-level host boundary only.
- [x] Define cancellation semantics for V1.

Verification:

- New API types are stable enough to write tests against before adapter work starts.
- Delta message ordering and end-of-message semantics are unambiguous in the API and in tests.
- No remaining core dependency on terminal UI abstractions.
- No approval API and no delegated tool-result API exist in V1.

### Milestone 2 / PR 2: Extract Runtime Services

- [x] Move instruction assembly behind a runtime service boundary.
- [x] Move persistence/history loading and saving behind a runtime service boundary.
- [x] Move MCP/tool loading behind runtime-owned service objects or factories.
- [x] Keep temporary coexistence with the old path only as a migration aid, not as a new long-term API.

Verification:

- Service boundaries can be used by the new runtime without importing CLI-specific code.
- Temporary coexistence does not introduce new public compatibility APIs.

### Milestone 3 / PR 3: Implement the Core Runtime Engine

- [x] Implement the new runtime engine and run state machine.
- [x] Move chat/agent orchestration behind one runtime-controlled execution engine with mode/policy differences instead of separate host-facing loops.
- [x] Rework tool execution so tools are executed internally without host callbacks or delegated tool handoff.
- [x] Remove `UI` and multi-method progress callback contracts from the new core runtime in favor of explicit commands plus one async event stream.

Verification:

- Unit tests cover completion, user-wait, failure, and cancellation.
- Unit tests cover delta emission and assembly into completed assistant messages.
- The new runtime can complete a basic end-to-end run without any CLI objects.

### Milestone 4 / PR 4: Rebuild Sub-Agent Execution on the Runtime Contract

- [x] Rework sub-agent launching so `launch_agent` creates runtime-owned child sessions rather than recursively re-entering the old UI/callback loop.
- [x] Ensure child sessions have isolated history/state and return structured results to the parent session.
- [x] Prevent child sessions from creating direct host interaction requirements in V1.
- [x] Make parent cancellation cascade to all active child sessions.

Verification:

- Tests cover parent/child completion, child failure propagation, and cancellation cascading.
- No child session requires its own public host control surface in V1.

### Milestone 5 / PR 5: Rebuild the CLI as an Adapter

- [x] Replace the current CLI orchestration path with an adapter that drives `AssistantSession`.
- [x] Move slash commands and terminal-only affordances into the CLI adapter layer.
- [x] Replace dense output rendering with event-driven CLI rendering.
- [x] Replace direct prompt callbacks with normal host message handling in the CLI adapter.
- [x] Stop using the old `Session`/`UI`/callback model for CLI execution.

Verification:

- CLI can run chat and agent modes through the new runtime only.
- CLI handles user replies, cancellation, and final output through the new boundary types.

### Milestone 6 / PR 6: Expose the Embeddable Python API

- [x] Expose a clean import surface for embedding from Python code.
- [x] Document the one-shot and interactive embedding patterns.
- [x] Provide a minimal in-process example that drives the runtime without any CLI objects.

Verification:

- Example embedding works without importing terminal UI code.
- API feels natural for a host application to control explicitly.

### Milestone 7 / PR 7: Add the Websocket Adapter

- [ ] Implement a websocket adapter that maps runtime events to outbound messages and host commands to runtime inputs.
- [ ] Define the JSON protocol for session creation, run start, user replies, and cancellation.

Verification:

- A browser/web client can drive an interactive run over websockets.
- Concurrent sessions do not share mutable session state.

### Milestone 8 / PR 8: Remove Legacy Runtime Paths and Finish Test Migration

- [x] Remove tests that assert the old `Session`, `UI`, `ProgressCallbacks`, or `ToolCallbacks` contracts.
- [x] Rewrite chat/agent tests to validate observable runtime behavior instead of internal callback ordering.
- [x] Rewrite CLI tests as adapter tests that exercise the new runtime boundary.
- [x] Rewrite cancellation tests against explicit session commands and terminal events.
- [x] Add tests for delta sequencing, completed message boundaries, and adapter rendering behavior.
- [x] Rewrite tool execution tests to cover internal runtime-owned tools only in V1.
- [ ] Update integration tests to use the new public embedding API.
- [x] Remove the old `Session` implementation and other dead runtime paths rather than adapting them.

Verification:

- Test names and assertions describe the new architecture rather than the old implementation.
- `just test` is green.
- `just lint` is green.

## Test Migration Plan

- [x] Add runtime-focused tests first, before CLI cutover, so the new contract is proven independently.
- [x] Replace old callback-centric assertions with session-command and session-event assertions.
- [x] Add explicit tests for parent/child session behavior, including cancellation cascading.
- [x] Keep CLI tests focused on adapter behavior rather than core runtime internals.
- [x] Ensure final cleanup leaves no test coverage coupled to legacy runtime APIs.

Verification:

- Runtime tests stand on their own without prompt-toolkit or dense output rendering.
- Adapter tests cover CLI behavior without reasserting runtime internals.
- `just test` is green.
- `just lint` is green.

## Risk Areas

- The runtime/host boundary can become muddled if normal user messages and runtime pause states are not clearly separated.
- Sandbox/isolation work can leak back into the core runtime if not explicitly pushed to an adapter/worker boundary.
- Sub-agent orchestration can accidentally reintroduce callback-style control flow if it is not rebuilt on the same session/run contract.
- Allowing child sessions to wait on the host directly would complicate the public API and cancellation semantics too early.
- The CLI rewrite can stall progress if terminal-specific behavior is allowed to shape the runtime API.

## Implementation Order

- [x] Land the contract and service boundaries before cutting over adapters.
- [x] Allow temporary side-by-side old/new internals during migration, but remove them before the final milestone closes.
- [x] Land sub-agent runtime integration before the CLI cutover so the adapter does not depend on legacy recursion paths.
- [x] Rebuild the CLI on top of the new engine.
- [x] Expose the Python embedding surface.
- [ ] Add websocket transport.
- [x] Remove legacy runtime paths and finish the test rewrite.

## Success Criteria

- A host program can embed the assistant without importing terminal UI abstractions.
- The CLI is only an adapter over the runtime.
- The project remains one repo/package; the runtime boundary, not package splitting, is the architectural goal.
- Normal user messages are the only host-driven interaction boundary in V1.
- Read-only progress is observable through one async event stream.
- The runtime can stream assistant output incrementally through delta messages.
- Sub-agents are runtime-owned child sessions rather than host-managed public objects.
- Only the top-level session is a public host control surface in V1.
- The runtime can support websocket driving without architectural workarounds.
- No compatibility layers remain for the old session/callback model.
