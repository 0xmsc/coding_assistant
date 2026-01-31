# Implementation Plan: Stdio/JSON Interface for Headless Agent

This plan outlines the steps to implement a JSON-based communication interface over Stdio, allowing the agent to run in headless environments (like Docker) or be integrated into other software without its CLI frontend.

## 1. Goal
Decouple the agent from the `prompt_toolkit` and `rich` UI implementations by providing a line-delimited JSON protocol over Stdin/Stdout.

## 2. Protocol Design

Every message is a single JSON line. To support multiplexing in a web environment, every message **may** include a `session_id`.

### Message Structure (Envelope)
`{"session_id": "optional-uuid", "type": "...", ...}`

### Agent -> Client (Events)
- **Status**: `{"type": "status", "level": "info|success|warning|error", "message": "string"}`
- **Content Chunk**: `{"type": "chunk", "content": "string"}` (for streaming LLM output)
- **Tool Start**: `{"type": "tool_start", "id": "call_id", "name": "string", "arguments": {}}`
- **Tool Result**: `{"type": "tool_message", "id": "call_id", "name": "string", "content": "string"}`

### Agent -> Client (Requests)
Requests require a response from the client. They include a `request_id` to correlate the response.
- **Ask User**: `{"type": "ask", "request_id": "uuid", "prompt": "string", "default": "string"}`
- **Confirm**: `{"type": "confirm", "request_id": "uuid", "prompt": "string"}`

### Client -> Agent (Responses)
- **Answer**: `{"type": "answer", "request_id": "uuid", "text": "string"}`
- **Confirmation**: `{"type": "confirmation", "request_id": "uuid", "value": true|false}`
- **Interrupt**: `{"type": "interrupt"}` (Stops current execution/generation)

## 3. Architecture Changes

### A. New UI Class: `JsonUI`
- **Location**: `src/coding_assistant/ui.py`
- **Logic**:
    - Implement `ask`, `confirm`, and `prompt`.
    - Interface with the `SessionStore` to send/receive JSON via the correct transport.

### B. New Component: `SessionStore` & `SessionManager`
To support multiple sessions in a single long-running process:
1.  **`SessionStore`**: Maintains a registry of active `Session` objects.
2.  **`SessionManager`**: 
    - Handles incoming WebSocket connections.
    - Generates a unique `session_id` and a scoped `working_directory` for each new connection.
    - Maps incoming network packets to the correct `JsonUI` and `JsonProgressCallbacks` instances.

### C. New Transport Layer: `APIServer` (WebSocket + HTTP)
- **Location**: `src/coding_assistant/api/`
- **Logic**:
    - Uses `FastAPI` + `Uvicorn` to provide both Real-time (WebSocket) and REST (HTTP) endpoints.
    - **REST Endpoints**:
        - `POST /sessions`: Creates a new session. Can optionally accept initial instructions or model overrides. Returns a `session_id`.
        - `GET /sessions`: Returns a list of all active `session_id`s and their metadata.
        - `GET /sessions/{session_id}/history`: Returns the complete message history for a specific session.
    - **WebSocket Endpoint**: `/ws/{session_id}` - For real-time chat and tool interaction.
    - On WebSocket connection: Connects to the existing session or starts a persistent agent loop if the session was just created.

### D. Entry Point Update
- **Location**: `src/coding_assistant/main.py`
- **Changes**:
    - Add `--api --port <port>` flag.
    - When provided, it starts the `WebSocketServer` instead of a single CLI session.
    - Uses `sys.stderr` for all internal logging to keep the process controllable.

## 4. Web Application Integration (The "Two-Docker" Goal)

This architecture specifically fulfills the requirement for a long-running Agent container:
1.  **Deployment**: One Docker container runs the `coding-assistant --api`.
2.  **Lifecycle**: The container stays alive. Your Backend container connects/disconnects at will.
3.  **Concurrency**: Each new WebSocket connection triggers a new `Session` instance inside the Agent container.
4.  **Cleanup**: When a WebSocket closes, the `SessionManager` cleans up the associated `Session` and temporary files.

## 5. Dockerization
Create a `Dockerfile` that:
1. Uses `python:3.12-slim` or `astral-sh/uv`.
2. Installs `git` and other necessary system tools.
3. Sets `PYTHONUNBUFFERED=1` to ensure JSON lines are emitted immediately.
4. Defaults to `ENTRYPOINT ["uv", "run", "python", "-m", "coding_assistant.main", "--stdio", "--model", "...", "--task", "..."]`.

## 5. Verification & Testing

### A. Integration Testing (Headless/LLM-Mocked)
Since we cannot use a real LLM for integration tests, the test suite must use a **Mock Completer**:
- **Mock Completer**: A Python function that implements the `Completer` protocol but returns pre-defined `AssistantMessages` based on the conversation history.
- **Protocol Test (`src/coding_assistant/tests/test_stdio_protocol.py`)**:
    - Use `pytest` to run the agent with a special test entry point or config that injects the Mock Completer.
    - **Happy Path**: Verify that status updates and tool starts are emitted as JSON lines on `stdout`.
    - **Interaction Path**: Mock an LLM response that triggers a tool requiring user confirmation. Verify that the agent emits a `confirm` request and pauses. Write a `confirmation` JSON to `stdin` and verify the agent continues.
    - **Cleanup**: Verify the agent exits cleanly after the task is finished.
    - **Log Isolation**: Verify that `stderr` contains standard logs and `stdout` contains strictly JSON lines.
