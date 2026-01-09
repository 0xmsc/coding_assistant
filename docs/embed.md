# Embedding Coding Assistant into a Web Application

This document outlines the current state of abstraction between the frontend (terminal) and backend (agent execution) and provides a plan for embedding the agent into a web application.

## Current Architecture Analysis

The codebase is already designed with a strong separation of concerns using abstract interfaces for user interaction and event reporting.

### Abstraction Layers

1.  **Input Abstraction (`UI` class in `src/coding_assistant/ui.py`)**:
    *   Defines abstract methods `ask`, `confirm`, and `prompt`.
    *   The terminal uses `PromptToolkitUI` to implement these.
    *   Embedding simply requires implementing a `WebUI` that bridges these calls to a frontend via WebSockets or API polling.

2.  **Event Abstraction (`ProgressCallbacks` in `src/coding_assistant/framework/callbacks.py`)**:
    *   Defines hooks for `on_user_message`, `on_assistant_message`, `on_tool_start`, `on_tool_message`, and streaming chunks (`on_content_chunk`, `on_reasoning_chunk`).
    *   This is perfectly suited for streaming updates to a web frontend.

3.  **Tool Control (`ToolCallbacks` in `src/coding_assistant/framework/callbacks.py`)**:
    *   Provides a way to intercept tool execution (e.g., for user confirmation).

### Strengths for Embedding
- **Async natively supported**: The entire execution loop is `asyncio`-based.
- **Dependency Injection**: The agent and chat loops take `UI` and `callbacks` as arguments.
- **Streaming by design**: The callback system supports fine-grained streaming of LLM output.

### Current Gaps & Needed Changes
- **Hardcoded CLI logic in `main.py`**: Much of the setup logic (MCP server initialization, tool gathering, config parsing) is currently coupled with the CLI entry point.
- **Direct prints**: Some UI-related output (like token usage or command help) bypasses the callback system and prints directly to the console using `rich` or standard `print()`.
- **UI hardcoded in `main.py` functions**: The `run_chat_session` and `run_root_agent` functions in `main.py` currently instantiate `PromptToolkitUI` directly.

---

## Embedding Plan

To successfully embed Coding Assistant into a web app, follow these steps:

### Phase 1: Refactoring for Library Use
- [ ] **Extract Session Management**: Move the setup logic from `main.py` into a reusable `Session` or `AgentEngine` class. This class should handle:
    - MCP server lifecycle.
    - Tool discovery and wrapping.
    - Configuration loading.
- [ ] **Decouple Output**: Refactor `run_chat_loop` and other framework functions to avoid direct `print()` calls. Redirect all intentional user-facing output through a new callback method (e.g., `on_system_message`).
- [ ] **Parameterize UI**: Modify `run_chat_loop` and `run_root_agent` to accept a `UI` instance as an optional parameter, defaulting to `NullUI` or `DefaultAnswerUI` if none is provided.

### Phase 2: Web Implementation
- [ ] **Create `WebUI`**: Implement a `UI` subclass that uses an async queue or event emitter to communicate with a web backend (e.g., FastAPI).
- [ ] **Create `WebProgressCallbacks`**: Implement a `ProgressCallbacks` subclass that serializes events to JSON for transmission over WebSockets/SSE.
- [ ] **State Persistence**: Implement a backend-agnostic way to save and load history (e.g., a `HistoryStore` interface) to replace direct filesystem access.

### Phase 3: Web Frontend Integration
- [ ] **Frontend Bridge**: Develop a web-based terminal or chat UI that can:
    - Render Markdown and streaming chunks.
    - Handle tool execution status visualization.
    - Support interactive prompts (asking/confirming) triggered by the backend.

## Recommended Tech Stack
- **Backend**: FastAPI (Python) for its excellent async support.
- **Communication**: WebSockets for bi-directional real-time interaction (prompts/answers) and SSE for one-way streaming of events.
- **Frontend**: React or Vue with a Markdown rendering library (e.g., `react-markdown`).
