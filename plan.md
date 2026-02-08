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
- [ ] **Add tests and metrics**: actor unit tests (message handling), integration tests for chat/agent flows, and tracing for actor message latency.

## When not to proceed
- If you cannot accept a step-by-step cutover that removes each legacy path as you go, or if feature delivery is the priority, delay this migration; the actor pattern adds overhead and risk during transition.
