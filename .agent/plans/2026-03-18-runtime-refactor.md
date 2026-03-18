# Runtime Refactor Plan

## Objective

Refactor the project from a CLI-first architecture into a transport-agnostic async session runtime that can be embedded in Python code, driven by a CLI, and exposed over websockets. The first implementation should be deliberately small: one session object, one runtime-owned agent loop, one host interaction boundary via normal user messages, internal non-interactive tools only, incremental assistant output via delta messages, and terminal completion/failure/cancellation events. This is a clean-break refactor: no compatibility layers, no attempt to preserve the current callback-based control model, and broad test rewrites are expected.

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

## Milestones

### Milestone 1: Define and Land the New Runtime Contract

- [ ] Define the minimal public runtime types: `AssistantSession`, `SessionOptions`, and `SessionEvent`.
- [ ] Define the minimal V1 event set: `assistant_delta`, `assistant_message`, `waiting_for_user`, `finished`, `failed`, `cancelled`.
- [ ] Define the only V1 host boundary: `waiting_for_user` followed by `send_user_message(...)`.
- [ ] Define V1 tool semantics: runtime-owned, non-interactive, no host-executed tool path.
- [ ] Define cancellation semantics for V1.
- [ ] Remove `UI` and control callbacks from the core runtime design.
- [ ] Remove multi-method progress callback contracts from the core runtime design in favor of one async event stream.

Verification:

- New API types are stable enough to write tests against before adapter work starts.
- Delta message ordering and end-of-message semantics are unambiguous in the API and in tests.
- No remaining core dependency on terminal UI abstractions.
- No approval API and no delegated tool-result API exist in V1.
- No soft-interrupt API exists in V1.

### Milestone 2: Rebuild the Core Engine Around the New Contract

- [ ] Implement the new runtime engine and run state machine.
- [ ] Move chat/agent orchestration behind one runtime-controlled execution engine with mode/policy differences instead of separate host-facing loops.
- [ ] Rework tool execution so tools are executed internally without host callbacks or delegated tool handoff.
- [ ] Rework instruction assembly and history persistence as runtime services rather than CLI-owned behavior.
- [ ] Rework sub-agent launching so it uses the same runtime contract rather than direct recursive UI/callback coupling.
- [ ] Remove the old `Session` implementation rather than adapting it.

Verification:

- Unit tests cover completion, user-wait, failure, and cancellation.
- Unit tests cover delta emission and assembly into completed assistant messages.
- History persists and resumes correctly under the new model.

### Milestone 3: Rebuild the CLI as an Adapter

- [ ] Replace the current CLI orchestration path with an adapter that drives `AssistantSession`.
- [ ] Move slash commands and terminal-only affordances into the CLI adapter layer.
- [ ] Replace dense output rendering with event-driven CLI rendering.
- [ ] Replace direct prompt callbacks with normal host message handling in the CLI adapter.
- [ ] Remove old CLI-specific runtime assumptions.

Verification:

- CLI can run chat and agent modes through the new runtime only.
- CLI handles user replies, cancellation, and final output through the new boundary types.

### Milestone 4: Add Embeddable Python API

- [ ] Expose a clean import surface for embedding from Python code.
- [ ] Document the one-shot and interactive embedding patterns.
- [ ] Provide a minimal in-process example that drives the runtime without any CLI objects.

Verification:

- Example embedding works without importing terminal UI code.
- API feels natural for a host application to control explicitly.

### Milestone 5: Add Web Transport Boundary

- [ ] Implement a websocket adapter that maps runtime events to outbound messages and host commands to runtime inputs.
- [ ] Define the JSON protocol for session creation, run start, user replies, and cancellation.

Verification:

- A browser/web client can drive an interactive run over websockets.
- Concurrent sessions do not share mutable session state.

## Test Migration Plan

- [ ] Remove tests that assert the old `Session`, `UI`, `ProgressCallbacks`, or `ToolCallbacks` contracts.
- [ ] Replace them with tests centered on session events and session commands.
- [ ] Rewrite chat/agent tests to validate observable runtime behavior instead of internal callback ordering.
- [ ] Rewrite CLI tests as adapter tests that exercise the new runtime boundary.
- [ ] Rewrite cancellation tests against explicit session commands and terminal events.
- [ ] Add tests for delta sequencing, completed message boundaries, and adapter rendering behavior.
- [ ] Rewrite tool execution tests to cover internal runtime-owned tools only in V1.
- [ ] Update integration tests to use the new public embedding API.

Verification:

- Test names and assertions describe the new architecture rather than the old implementation.
- `just test` is green.
- `just lint` is green.

## Risk Areas

- The runtime/host boundary can become muddled if normal user messages and runtime pause states are not clearly separated.
- Sandbox/isolation work can leak back into the core runtime if not explicitly pushed to an adapter/worker boundary.
- Sub-agent orchestration can accidentally reintroduce callback-style control flow if it is not rebuilt on the same session/run contract.
- The CLI rewrite can stall progress if terminal-specific behavior is allowed to shape the runtime API.

## Implementation Order

- [ ] Finalize the runtime contract before touching adapter code.
- [ ] Rebuild the engine and tests around the new contract.
- [ ] Rebuild the CLI on top of the new engine.
- [ ] Expose the Python embedding surface.
- [ ] Add websocket transport.

## Success Criteria

- A host program can embed the assistant without importing terminal UI abstractions.
- The CLI is only an adapter over the runtime.
- Normal user messages are the only host-driven interaction boundary in V1.
- Read-only progress is observable through one async event stream.
- The runtime can stream assistant output incrementally through delta messages.
- The runtime can support websocket driving without architectural workarounds.
- No compatibility layers remain for the old session/callback model.
