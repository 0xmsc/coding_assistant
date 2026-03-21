# Lightweight Runtime Surface Plan

## Objective

Make the embeddable runtime significantly smaller by turning `AssistantSession` into a pure agent state machine that owns transcript and model progression, but does not provision or execute external tools itself.

Success criteria:

- A host can create and run an `AssistantSession` without importing MCP, CLI, sandbox, or skills code.
- `SessionOptions` contains only runtime behavior/configuration, not environment discovery or product wiring.
- MCP startup, tool wrapping, instruction assembly, and file-history defaults live outside the core session.
- The core runtime sees only instructions, tool metadata, completer, and optional persistence.
- Tool policy and execution decisions happen outside the core runtime via explicit tool-call events and result submission.
- Sub-agent behavior and the event model remain coherent under the new host-mediated tool boundary.

## Scope

In scope:

- Introduce an explicit host-mediated tool-call protocol for the core runtime.
- Move instruction assembly and tool provisioning out of `AssistantSession.__aenter__`.
- Move default MCP/bootstrap logic into a separate integration layer or wrapper used by the CLI.
- Update tests and embedding docs to target the lighter runtime surface.

Out of scope:

- Websocket work.
- Redesigning MCP shell/python background tasks.
- Large changes to user-facing CLI behavior.

## Approach

- Split the current runtime into two layers:
  - a low-level core session that emits `tool_call_requested` and accepts tool results back from the host
  - an optional wrapper/default app layer that resolves tools, applies policy, executes tool calls, and feeds results back into the core session
- Treat MCP, skills, user-instruction merging, approvals, sandboxing, and default history storage as wrapper concerns.
- Keep the CLI behavior the same, but make it run on top of the wrapper instead of talking directly to provisioning code inside the session.

## Target Shape

- `AssistantSession`
  - owns the model loop, history, cancellation, event queue, builtin runtime tools, and sub-agent execution state.
- `SessionOptions`
  - keeps only runtime tuning such as model selection and compaction threshold.
- `ToolSpec` (name TBD)
  - describes tool name, description, and parameters only.
- Core runtime protocol
  - emits `tool_call_requested`
  - accepts `submit_tool_result(...)` / `submit_tool_error(...)`
- Wrapper / default app layer
  - starts MCP servers
  - exposes tool specs to the session
  - decides whether tool calls are allowed
  - executes approved tool calls
  - assembles instructions
  - creates file-backed history storage
  - may expose a higher-level convenience runner for "fully managed" use
  - is used by the CLI and any future "default app" integrations

## Milestones / PR Sequence

### Milestone 1 / PR 1: Define the New Runtime Boundary

- [x] Define the host-mediated tool protocol for the core runtime.
- [x] Shrink `SessionOptions` to runtime-only fields.
- [x] Introduce a tool metadata type such as `ToolSpec`.
- [x] Define what remains core: runtime loop, builtin runtime tools, sub-agent execution, persistence hook, tool-call state machine.
- [x] Define what moves out: MCP config, skills directories, user instructions, package-root discovery, default history-store construction, approvals, and external tool execution.

Verification:

- The public constructor shape is clear enough to write tests against before moving adapters.
- The session event/command contract is unambiguous for tool calls and tool results.

### Milestone 2 / PR 2: Make `AssistantSession` Use the New Tool Protocol

- [x] Change `AssistantSession` to accept explicit instructions, tool specs, completer, and optional persistence directly.
- [x] Add `tool_call_requested` session events and result-submission commands.
- [x] Remove MCP startup and instruction assembly from `AssistantSession.__aenter__`.
- [x] Rework the run loop so external tool calls suspend the session until the host replies.
- [x] Preserve cancellation and non-tool event semantics.

Verification:

- A unit test can run a session with only explicit instructions, tool specs, and a fake completer.
- Tool-request / tool-result round trips are fully covered in runtime tests.

### Milestone 3 / PR 3: Rebuild Sub-Agents on the New Boundary

- [x] Decide how child sessions issue tool requests under the new contract.
- [x] Keep the public host boundary top-level only, unless there is a strong reason to expose nested tool requests.
- [x] Make parent-owned wrappers/executors satisfy child tool requests as needed.
- [x] Preserve current cancellation behavior across parent/child sessions.

Verification:

- Sub-agent tests still pass under the new tool-request boundary.
- The contract for nested tool calls is explicit and test-covered.

### Milestone 4 / PR 4: Build the Default Wrapper / Integration Layer

- [x] Add a non-runtime wrapper/builder module that turns current app inputs into a managed runner.
- [x] Move default MCP server config creation there.
- [x] Move instruction assembly there.
- [x] Move default file-history-store creation there.
- [x] Implement allow/deny/execute behavior for tool requests there.
- [x] Keep MCP/task/background-process behavior unchanged, but ensure it is reached only through the wrapper/tool stack.

Verification:

- The CLI can still boot the same default tool environment.
- The runtime package no longer imports MCP bootstrap helpers during normal session construction.

### Milestone 5 / PR 5: Rewire the CLI and Embedding Surface

- [x] Update the CLI adapter to run on top of the wrapper/default integration layer.
- [x] Update root exports and examples to show both the low-level session API and the higher-level managed wrapper.
- [x] Decide whether to expose the default wrapper publicly or keep it as an app-only helper.

Verification:

- CLI chat and agent modes still work unchanged from the user's perspective.
- README embedding examples no longer imply that a host must rely on MCP/bootstrap internals.

### Milestone 6 / PR 6: Cleanup and Test Reshape

- [x] Remove no-longer-needed session bootstrap helpers and config fields.
- [x] Rewrite or split tests so runtime tests use explicit tool specs and tool-result submission, while CLI tests use the default wrapper path.
- [x] Ensure no runtime tests depend on MCP startup side effects unless they are explicitly integration tests.

Verification:

- `just test` passes.
- `just lint` passes.
- Runtime tests read as pure library tests, while wrapper/CLI tests cover the product wiring separately.

## Risks

- The main risk is nested tool-call complexity for sub-agents; the top-level host boundary should stay explicit.
- The second risk is preserving the old constructor shape via compatibility layers; this would weaken the boundary instead of clarifying it.
- The third risk is letting wrapper concerns drift back into `runtime/`, which would keep the package import graph product-shaped.
- The fourth risk is making the low-level API too noisy; a convenience wrapper should absorb that without contaminating the core runtime.

## Recommended Next Milestone

- [x] Start with Milestone 1 only: define the low-level tool-call protocol, tool metadata type, and revised constructor boundary before moving any implementation.
