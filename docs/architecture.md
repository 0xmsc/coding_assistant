# Architecture

This project has one core agent runtime with several ways to drive it. The
terminal CLI, local remote endpoint, manager service, and worker service should
share the same core types instead of inventing separate event or history models.

## Packages

`coding_assistant.core` owns the agent runtime.

- `AgentSession` owns the in-memory transcript, prompt queue, steering prompts,
  cancellation, and the active agent run loop.
- `run_agent_event_stream` drives model streaming and tool-call boundaries.
- `session_updates` converts internal agent events into UI/remote updates and
  committed-message records.
- `history` and `llm.types` define the message and completion data structures
  shared by local and remote callers.

`coding_assistant.tools` owns local tool implementations. Tool calls are
executed by `AgentSession` and the core agent loop, not by the manager or the
JSON-RPC transport.

`coding_assistant.cli` owns the interactive terminal application. It builds an
agent, creates an `AgentSession` directly, renders session updates in the
terminal, and may expose a local single-session remote endpoint for other
clients. The terminal UI itself is not a JSON-RPC client.

`coding_assistant.remote` owns JSON-RPC framing and protocol serialization.

- `acp.py` contains the custom ACP-inspired JSON-RPC helpers and common payload
  validation.
- `control.py` contains the shared remote-control implementation for
  JSON-RPC-driven `AgentSession` prompt, cancel, update, and commit behavior.
- `protocol.py` converts between core messages/session updates and JSON-RPC
  payloads.
- `client.py` contains `RemoteSessionClient`, the single client abstraction for
  talking to a live remote session endpoint.
- `server.py` contains the local single-session endpoint used around an
  existing `AgentSession`.

`coding_assistant.manager` owns durable web-session state and manager-facing
RPC.

- `SessionStore` persists sessions and committed messages in SQLite.
- `ManagerService` validates opaque scope metadata, loads canonical history,
  calls a worker runner, and commits completed turns.
- `manager.server` adapts `ManagerService` to JSON-RPC.
- `workspace.py` derives a managed workspace path from the session id.

`coding_assistant.worker` owns the worker-side remote runtime. A worker receives
manager-provided session state, creates one `AgentSession`, streams
`session/update` notifications, and sends `_session/commit` notifications for
the new messages produced by the active turn. Workers do not persist canonical
history.

## Modes

### CLI

```text
terminal UI
  -> cli
  -> AgentSession
  -> tools / model
  -> shared session updates
  -> terminal renderer
```

The CLI is the simplest runtime path. It creates and owns one local
`AgentSession`. History is in memory for that process unless a caller adds a
separate persistence layer.

### Local Remote Endpoint

```text
remote client
  -> remote/server.py
  -> existing AgentSession
  -> shared session updates
```

This endpoint wraps one already-created `AgentSession` with the shared
remote-control implementation. It emits the same provisional `session/update`
and completed-turn `_session/commit` notifications as managed workers, but
local callers may ignore commit notifications. It is useful for CLI-owned
remote access and tests. It is not the web manager.

### Managed Web Sessions

```text
application backend
  -> manager JSON-RPC service
  -> SessionStore
  -> worker runner
  -> worker JSON-RPC service
  -> AgentSession
```

The embedding application owns browser authentication. It strips any
browser-provided scope metadata and injects trusted `params._meta.scopeId` when
forwarding to the manager. The manager stores canonical history and enforces
scope. Worker processes execute turns and return committed messages.

## Session Ownership

The manager is the source of truth for managed web sessions.

- Session ids are durable identifiers.
- Workspace paths are derived from session ids.
- SQLite stores committed history and session metadata.
- Workers hold only one active in-memory `AgentSession`.
- A worker commit is appended by the manager after the worker completes a turn.

This avoids split durable history between manager and worker. The worker can be
discarded and recreated from manager-provided state.

## Protocol Boundary

The project uses a custom JSON-RPC protocol based on ACP shapes. It is not a
complete ACP implementation. See `docs/rpc.md` for method details.

JSON-RPC is only the process/container boundary. Inside the application, code
should use core dataclasses such as `BaseMessage` and `SessionUpdate` rather
than passing raw protocol dictionaries around.

Every remote-controlled `AgentSession` emits provisional `session/update`
notifications and a completed-turn `_session/commit` notification. Persistence
is still caller-specific: the manager persists commits, while CLI-owned local
remote callers may ignore them.

Private `_session/*` methods are manager/worker control methods. Browser-facing
or application-facing code should use the public session methods documented in
`docs/rpc.md`.

## Direction

Keep the architecture simple:

- Put agent execution behavior in `core`.
- Put terminal behavior in `app`.
- Put JSON-RPC framing and conversion in `remote`.
- Put durable managed-session state in `manager`.
- Put one-session worker execution in `worker`.
- Do not add application-specific concepts to `coding_assistant`; use opaque
  scope ids at the manager boundary.
