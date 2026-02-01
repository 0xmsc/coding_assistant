# Implementation Plan: Evolution to Actor-Based Architecture

This plan outlines the strategic transformation of the `coding-assistant` from a procedural, call-stack-based architecture to a reactive, **Actor-based** system. This shift will enable superior concurrency, native multi-tenancy, and resilient headless operation.

## 1. Core Concepts

### A. The Actor
Every logical component becomes an independent `Actor` with:
- **Mailbox**: An `asyncio.Queue` for incoming messages.
- **State Machine**: Logic to transition between states (e.g., `IDLE` -> `THINKING` -> `WAITING_FOR_INPUT`).
- **Context**: Access to the `ActorSystem` to send messages to others.

### B. The Envelope & Correlation
To solve the "Loss of Stack Trace" problem, every message is wrapped in an `Envelope`:
```python
@dataclass(frozen=True)
class Envelope:
    sender: str              # Address of the sender
    recipient: str           # Address of the recipient
    correlation_id: str      # Unique ID for the specific "conversation" or task
    trace_id: str            # Global ID for the entire lifecycle (for logging)
    payload: Message         # The actual data (Pydantic model)
    timestamp: datetime
```

## 2. Architecture Overview

- **ActorSystem (The Postman)**: The central registry and router.
- **OrchestratorActor**: The brain. Manages LLM requests and task logic.
- **ToolActor(s)**: Executes shell, filesystem, and MCP operations.
- **UIActor (Terminal/WebSocket)**: Translates system messages into human interaction.
- **ObserverActor(s)**: Passive listeners for Logging, Tracing, and Cost Tracking.

## 3. Phase-by-Phase Transformation

### Phase 1: The Infrastructure (Foundation)
1. **Model Definition**: Expand the existing Pydantic models in `api/models.py` to cover all internal actor communications.
2. **Actor Base Class**: Create a robust `BaseActor` handling the boilerplate of `mailbox.get()` and error management.
3. **The Dispatcher**: Implement the `ActorSystem` with:
    - `tell(envelope)`: Async fire-and-forget.
    - `ask(message)`: A helper that creates a transient `Future` and waits for a response with a matching `correlation_id`.

### Phase 2: State Machine orchestrator
1. **Refactor `AgentLoop`**: The current linear loop in `agent.py` must be rewritten as a **Finite State Machine (FSM)**.
2. **State Persistence**: Implement a `StateStore` so an Actor's state can be serialized. This allows the agent to survive "hibernation" while waiting for user input.

### Phase 3: The UI Actors (CLI & API)
1. **TerminalActor**: Wrap `prompt_toolkit` logic. It becomes a client that "subscribes" to the Orchestrator's status updates.
2. **WebSocketActor**: Replace the current `WebSocketUI` bridge. It now maps WebSocket frames directly into the `ActorSystem` as messages.

### Phase 4: Observability & Tracing (Crucial)
1. **TraceBroadcaster**: A built-in feature of the Dispatcher that emits every `Envelope` to a special `TraceObserver`.
2. **Unified Tracer**: Implement a tracing tool that uses `trace_id` and `parent_id` to reconstruct the "Virtual Stack Trace" across multiple actors.

## 4. Addressing Disadvantages

| Challenge | Strategy |
| :--- | :--- |
| **Logic Fragmentation** | Use a formal FSM library (like `transitions` or a custom DSL) to visualize and document actor states. |
| **Debugging** | Implement **Breadcrumbs**: every `Envelope` carries a history of previous `trace_ids` it was spawned from. |
| **Performance** | Use `ProcessPoolExecutor` for `ToolActors` that perform heavy CPU tasks, keeping the `OrchestratorActor` (I/O bound) responsive. |

## 5. Proposed Structural Overhaul: The "Actor" Blueprint

### A. Directory Structure Changes
The current logic in `framework/` would be decoupled into a message-driven package.

```text
src/coding_assistant/
├── actors/                 # Actor implementations
│   ├── base.py             # BaseActor and Mailbox logic
│   ├── system.py           # The "Postman" (Dispatcher/Registry)
│   ├── orchestrator.py     # REPLACES agent.py (The FSM Brain)
│   ├── tool_worker.py      # REPLACES execution.py (Stateless workers)
│   ├── ui_gateway.py       # REPLACES ui.py (Bridge to Terminal/Web)
│   └── observer.py         # Passive telemetry/logging actors
├── messaging/              # Communication protocol
│   ├── envelopes.py        # Envelope, CorrelationID, TraceID
│   └── schema.py           # Pydantic models for all message types
└── domain/                 # Refactored: Logic without side-effects
    ├── history.py          # Pure state management for history
    └── tools.py            # Individual tool logic (Stateless)
```

### B. Component Mapping: Current vs. Actor
| Current Component | Actor Equivalent | Change in Responsibility |
| :--- | :--- | :--- |
| `run_agent_loop` | `OrchestratorActor` | Moves from a linear `while` loop to an **Event Handler**. It reacts to `TaskStarted`, `LLMResponseReceived`, and `ToolResultReceived`. |
| `handle_tool_calls` | `ToolWorkerActor` | Instead of an awaited function, it becomes a **pool of actors**. This allows running a long shell command without blocking the Orchestrator. |
| `PromptToolkitUI` | `UIGatewayActor` | No longer "called" by the agent. It **listens** for `DisplayMessage` events and **emits** `UserInputReceived` events. |
| `callbacks.py` | `ObserverActor` | Instead of passing callback objects deep into the stack, a passive actor simply **subscribes** to all messages in the system. |

### C. Execution Flow: The "Virtual Stack"
In the current code, the stack trace handles the flow. In the Actor system, the **Correlation ID** in the `Envelope` handles it.

**Example: A Tool Execution Flow**
1.  **User** types a command → `UIGateway` sends `UserCommand` to `Orchestrator`.
2.  **Orchestrator** sends `PromptModel` to `LLMActor`.
3.  **LLMActor** returns `ToolCallRequested`.
4.  **Orchestrator** transitions to `WAITING_FOR_TOOLS` state and sends `ExecuteTool` to `ToolWorker`.
5.  **ToolWorker** finishes and sends `ToolResult` back to `Orchestrator`.
6.  **Orchestrator** sees the `CorrelationID` matches its pending task and resumes logic.

### D. Why this is "Harder" but "Stronger"
*   **The Difficulty:** You cannot use `try/except` around a tool call. If the `ToolWorker` crashes, the `Orchestrator` just never gets a message back. You must implement "Supervision" (e.g., the `ActorSystem` detects the crash and sends a `WorkerCrashed` message).
*   **The Strength:** You can restart the `UIGateway` (e.g., switch from Terminal to Web) without affecting the `Orchestrator`. The `Orchestrator` doesn't even know the UI restarted; it just waits for the next `UserInput` message with the correct ID.

## 6. Why do this? (The "North Star")

By moving to this model, `coding-assistant` becomes more than a CLI—it becomes a **Distributed Agent Platform**.
- **Collaborative Coding**: Multiple users can connect their `TerminalActors` to a single `OrchestratorActor` and watch/influence the task in real-time.
- **Resilient Long-Running Tasks**: A task can run for hours; if the server restarts, the actor reloads its state from the `correlation_id` and continues.
- **Native Monitoring**: Dashboarding and safety audits become "plug-ins" (Observers) that don't need to be integrated into the core engine.

## 7. Phased Implementation Strategy

To ensure stability and maintainability, the transition to the Actor-based architecture is broken down into incremental steps.

### Step 1: Communication Protocol (Done)
**Objective**: Define the "Language" of the actors without changing any existing logic.
- **Action**: Create `src/coding_assistant/messaging/` containing `envelopes.py` and `messages.py`.
- **Logic**: Use existing types from `src/coding_assistant/framework/types.py` as payloads for the new `Envelope` system.
- **Testing Strategy**: 
    - **Unit Tests**: Create `src/coding_assistant/messaging/tests/test_protocol.py`.
    - **Verify**: Pydantic validation of Envelopes, JSON serialization/deserialization for future persistability, and generation of unique `correlation_id`/`trace_id`.

### Step 2: Infrastructure & "Shadow" Logging (Done)
**Objective**: Introduce the `ActorSystem` and `BaseActor` as a passive utility.
- **Action**: Implement the `ActorSystem` (dispatcher) and an `ObserverActor`.
- **Integration**: Hook the `ObserverActor` into the existing `AgentLoop` (via callbacks) to log events without affecting execution.
- **Testing Strategy**:
    - **Infrastructure Tests**: Test the `ActorSystem` dispatch logic—ensure messages sent to an address reach the correct mailbox.
    - **Integration Test**: Run a mock `AgentLoop` session and verify the `ObserverActor` captured the expected sequence of events (e.g., `START` -> `LLM_PROMPT` -> `TOOL_CALL`).

### Step 3: Tool-Worker Isolation
**Objective**: Offload tool execution to a dedicated actor.
- **Action**: Create `ToolWorkerActor`. Refactor `handle_tool_calls` in `execution.py` to wrap requests into an `Envelope` and use `ask()` to get results from the actor.
- **Testing Strategy**:
    - **Regression Tests**: Run `src/coding_assistant/framework/tests/test_tool_execution.py` and `test_mcp_wrapped_tool.py`.
    - **Actor-Specific Test**: Verify that sending a `ExecuteTool` message results in a `ToolResult` message with the same `correlation_id`.

### Step 4: UI Gateway (Decoupling the User Interface)
**Objective**: Make the UI reactive rather than imperative.
- **Action**: Implement `UIGatewayActor`. Move from direct UI calls to sending `DisplayMessage` envelopes.
- **Testing Strategy**:
    - **Mock UI Testing**: Use a `MockUIGatewayActor` in existing `test_ui.py` to verify that the agent's "output" events are correctly translated to actor messages.
    - **Round-trip Test**: Verify that a "Human Input Required" state correctly waits for a `UserInputReceived` message to proceed.

### Step 5: The Orchestrator (Final FSM Shift)
**Objective**: Replace the linear `while` loop with a formal State Machine.
- **Action**: Rewrite `AgentLoop` logic into `OrchestratorActor`.
- **Testing Strategy**:
    - **The "Great Alignment"**: Run the full suite of existing integration tests (e.g., `test_agent_loop.py`). The behavior should be indistinguishable from the previous linear implementation.
    - **FSM Trace Test**: Verify that the Orchestrator transitions through the correct states (`IDLE` -> `THINKING` -> `BUSY` -> `IDLE`) for a standard completion.

## 8. Clarifying Questions (Refined)

1. **Model Location**: Consolidation of Pydantic models vs. importing. (Proposed: Import for compatibility initially).
2. **State Store**: Backend for `StateStore`. (Proposed: In-memory for MVP, SQLite later).
3. **Supervision**: Error policy for ToolActors. (Proposed: Commit to supervision/restart rather than fallback).
