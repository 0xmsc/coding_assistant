# Actor Pattern Migration Plan

## What this codebase does
- Python CLI that orchestrates coding tasks via chat mode or autonomous agent mode.
- Manages sessions, history, and instructions; connects to MCP servers that expose tools (shell, python, filesystem, todo, tasks, skills).
- Runs async loops for chat/agent execution, tool calls, and UI prompting, with sandboxing and logging.

## How an actor pattern would fit
- The system already separates concerns (Session, agent/chat loops, tool execution, MCP server management, UI), but coordinates them directly via async calls and shared state.
- Actor-style message passing could isolate state and concurrency boundaries (tool execution, UI prompts, task management, MCP server lifecycle) and make cancellation/supervision clearer.
- It is most useful where multiple concurrent tasks run and need controlled coordination (tool calls, background tasks, interruptions).

## Recommendation
- Migrate to an actor-based architecture in strict cutovers: each step replaces the old code path entirely (no hybrid runtime).
- Target components with concurrency and lifecycle complexity first (tool execution, MCP server management, UI interactions), then move the agent loop into an actor.
- Each cutover must also port and update all relevant tests, removing legacy-path coverage at the same time.
- Execute the work in the proposed chunks; before each chunk, expand it into detailed sub-todos as needed, then implement, write tests, and commit.

## Target actor system (end state)
- Create long-lived system actors at startup (e.g., AgentActor, ToolCallActor, UserActor) and let them communicate via messages.
- Avoid per-call actor creation inside loops; actor instances should own state and run for the session lifetime.
- Ensure the execution flow is driven by actor messages (no direct orchestration logic outside actors).

## Proposed incremental transformation (no hybrid runtime, tests ported per step)
- Each checklist item is a chunk; add sub-todos beneath it before starting that chunk.
- [x] **Introduce a lightweight actor runtime**: asyncio task + mailbox (Queue), typed messages, start/stop lifecycle, and a minimal supervision strategy.
  - [x] Add actor runtime module with start/stop/send/ask and exception handling.
  - [x] Add unit tests for send/ask/stop behavior.
- [x] **Cut over Tool Execution**:
  - Implement ToolExecutor actor and route all tool calls through it.
  - Remove the old direct tool execution path immediately after the cutover.
  - Port tool execution tests to the actor path and delete legacy-path tests.
  - [x] Define ToolExecutor message types (execute call, batch calls).
  - [x] Implement ToolExecutor actor wiring in framework execution.
  - [x] Replace handle_tool_calls with actor messaging and remove direct path.
  - [x] Update tool execution tests to use the actor path; remove legacy tests.
  - [x] Add any missing unit tests for ToolExecutor error propagation/cancellation.
- [x] **Cut over MCP Server Management**:
  - Implement MCPServerManager actor for startup/shutdown and tool registry updates.
  - Remove direct server lifecycle management from Session once the actor is live.
  - Port MCP lifecycle tests to the actor path and delete legacy-path tests.
  - [x] Define MCPServerManager messages and result bundle shape.
  - [x] Implement MCPServerManager actor (start/initialize/shutdown) using existing MCP helpers.
  - [x] Update Session to use MCPServerManager and remove direct MCP lifecycle handling.
  - [x] Add MCPServerManager tests; update Session tests to mock the manager.
  - [x] Remove/adjust any legacy MCP lifecycle tests or mocks.
- [x] **Cut over UI interactions**:
  - Implement UI actor to serialize prompts/asks.
  - Replace direct UI calls with actor messages and delete the old call sites.
  - Port UI prompt/ask tests to the actor path and delete legacy-path tests.
  - [x] Implement ActorUI wrapper and scope helper.
  - [x] Route chat loop/tool handling through ActorUI.
  - [x] Add ActorUI serialization tests and update UI mocks if needed.
- [x] **Cut over History persistence**:
  - Implement History actor for save/compact operations.
  - Remove direct history writes from chat/agent loops.
  - Port history tests to the actor path and delete legacy-path tests.
  - [x] Define HistoryManager messages and scope helper.
  - [x] Implement HistoryManager actor to save orchestrator history.
  - [x] Update Session to use HistoryManager for chat/agent history saves.
  - [x] Add HistoryManager tests and update Session tests to mock the manager.
- [x] **Cut over Agent loop**:
  - Implement Agent actor that owns run_agent_loop state and transitions.
  - Replace the existing run_agent_loop orchestration with actor message flow.
  - Port agent loop tests to the actor path and delete legacy-path tests.
  - [x] Define AgentLoop actor message and wiring.
  - [x] Move agent loop execution into actor handler.
  - [x] Ensure agent loop tests run through the actor path.
- [x] **System-level actor orchestration**:
  - [x] Create long-lived system actors in Session (AgentActor, ToolCallActor, UserActor).
  - [x] Move per-call actor creation (chat loop, single step, UI) into these long-lived actors.
  - [x] Rewire message passing so actors coordinate without direct function orchestration.
  - [x] Remove short-lived actor scaffolding after system actors take ownership.
- [x] **Add tests and metrics**: actor unit tests (message handling), integration tests for chat/agent flows, and tracing for actor message latency.
  - [x] Update existing tests to exercise actor interfaces (system actors, ToolCallActor, AgentActor).
  - [x] **ChatLoop actor cutover** (separate task)
    - [x] Replace run_chat_loop with a ChatLoop actor (no direct loop logic outside actors).
    - [x] Add integration tests for actor-based chat flows.
  - [x] **do_single_step actor cutover** (separate task)
    - [x] Move do_single_step/completer invocation behind an actor boundary.
    - [x] Add integration tests for actor-based agent flows.
  - [x] Add tracing/metrics for actor message latency and lifecycle events.
  - [x] Remove any remaining legacy scaffolding found during audit.

## Next steps (actor-driven core)
- [x] Move agent loop logic from `framework/agent.py` into `AgentActor` so the loop is fully message-driven and state-owned by the actor.
- [x] Move tool call execution + tool-result history append into `ToolCallActor`, removing direct orchestration from `framework/execution.py`.
- [x] Remove `SystemActors` and wire actors directly (expose run_chat/run_agent via actor messages).
- [x] Slim `Session` to wiring/config + actor lifecycle only, delegating run calls to actors directly.
- [x] Update tests to drive agent/tool behavior via actor messages; keep `Session` tests focused on wiring and integration.
- [x] Refactor actors to send-only messaging with response channels (remove `Actor.ask` usage).
- [x] Remove `ChatLoopActor` and run chat flow directly in `AgentActor` using `UserActor`/`ToolCallActor`.
- [x] Remove `SystemActors` and wire actors directly in Session/tools/tests.
- [x] Make `AgentActor` own history and expose history snapshots via `get_history` / `get_agent_history`.

## When not to proceed
- If you cannot accept a step-by-step cutover that removes each legacy path as you go, or if feature delivery is the priority, delay this migration; the actor pattern adds overhead and risk during transition.
