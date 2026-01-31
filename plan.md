# Implementation Plan: API Interface for Headless Agent

This plan outlines the steps to implement a JSON-based API and WebSocket interface, allowing the agent to run in headless environments (like Docker) and be integrated into web applications or other software.

## 1. Goal
Provide a long-running Agent Service accessible via HTTP and WebSockets. This replaces the terminal-bound UI with a structured JSON protocol, enabling multi-session support and remote integration.

## 2. Architecture: Direct-to-Bridge
Instead of running the agent as a separate subprocess and piping JSON over Stdio, the `FastAPI` server will manage `Session` objects in-process. 

### Key Components
- **`Bridge` Classes**: Implement specialized versions of `UI` and `ProgressCallbacks` that translate internal agent calls into WebSocket messages.
- **`SessionManager`**: Manages the lifecycle of multiple `Session` objects, mapping them to unique `session_id`s and temporary working directories.
- **`FastAPI` Server**: Provides REST endpoints for session management and a WebSocket endpoint for real-time interaction.

## 3. Protocol Design (JSON)

Every message is a JSON object.

### Agent -> Client (Events & Requests)
- **Status**: `{"type": "status", "level": "info|success|warning|error", "message": "string"}`
- **Content Chunk**: `{"type": "chunk", "content": "string"}` (for streaming LLM output)
- **Tool Start**: `{"type": "tool_start", "id": "call_id", "name": "string", "arguments": {}}`
- **Tool Result**: `{"type": "tool_message", "id": "call_id", "name": "string", "content": "string"}`
- **User Request**: Requires a response. Includes a `request_id`.
    - `{"type": "ask", "request_id": "uuid", "prompt": "string", "default": "string"}`
    - `{"type": "confirm", "request_id": "uuid", "prompt": "string"}`

### Client -> Agent (Responses & Commands)
- **Task Start**: `{"type": "start", "task": "string"}`
- **User Answer**: `{"type": "answer", "request_id": "uuid", "text": "string"}`
- **User Confirmation**: `{"type": "confirmation", "request_id": "uuid", "value": true|false}`
- **Interrupt**: `{"type": "interrupt"}` (Stops current execution)

## 4. Implementation Steps

### Phase 1: Foundation
1.  **Dependencies**: Add `fastapi`, `uvicorn`, and `pydantic` to `pyproject.toml`.
2.  **Schema**: Create `src/coding_assistant/api/models.py` to define the JSON message shapes using Pydantic.
3.  **The Bridge**: Create `src/coding_assistant/api/bridge.py`:
    - `WebSocketUI`: Implements `UI`. `ask()` and `confirm()` send a message and await a response from an internal `asyncio.Queue`.
    - `WebSocketCallbacks`: Implements `ProgressCallbacks`. Translates calls like `on_status_message` into WebSocket JSON.

### Phase 2: Service Layer
1.  **`SessionManager`**: 
    - Registry of `session_id -> Session`.
    - Handles creation of scoped working directories.
    - Manages `asyncio.Task` for the agent loop per session.
2.  **FastAPI Core**:
    - `POST /sessions`: Initialize a session.
    - `GET /sessions`: List active sessions.
    - `WS /ws/{session_id}`: The real-time bridge.

### Phase 3: CLI & Integration
1.  **Entry Point**: Update `src/coding_assistant/main.py` with an `--api` flag and `--port` / `--host` options.
2.  **Logging**: Ensure all logs go to `stderr` in API mode to avoid interference.
3.  **Cleanup**: Implement logic to terminate sessions and clean up filesystem artifacts on WebSocket disconnect or session timeout.

## 5. Verification & Testing

1.  **Unit Tests**: Test the `WebSocketUI` queue logic independently of the network.
2.  **Integration Tests**:
    - Build a small "mock client" script using `websockets` library.
    - Run the agent in `--api` mode.
    - Verify the mock client can start a task, receive status updates, and answer a confirmation request.
3.  **Concurrency Test**: Verify that two separate WebSocket clients can run tasks in two separate sessions without cross-talk.
