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

## 6. Immediate Next Steps (Clarifying Questions)

1. **Serialization**: Should we enforce that ALL messages must be JSON-serializable? (Required for future-proofing multi-process actors).
2. **Concurrency**: How many parallel tasks (Orchestrators) should a single `ActorSystem` manage before spinning up new processes?
3. **Supervision**: What is the "Restart Policy" if a `ToolActor` crashes? (e.g., Restart the tool, or fail the entire task?)
