# Architecture

This project has one core agent runtime with several ways to drive it. The
terminal CLI, manager service, and worker service should
share the same core types instead of inventing separate event or history models.

## Packages

`coding_assistant.core` owns the agent runtime.

- `AgentSession` owns the in-memory transcript, prompt queue, steering prompts,
  cancellation, model streaming, tool execution, and the active agent run loop.
- Enqueuing a prompt returns a `ScheduledRun`. Its `RunOutcome` is the
  authoritative completion signal for that run; steering prompts become part
  of the active run rather than creating transport-level prompt identities.
- `AgentSessionEvent` describes in-process runtime events consumed directly by
  the CLI and remote controller. Events are observational and are not used to
  correlate RPC responses.
- `session_updates` defines only the committed and provisional updates that
  cross the manager/remote boundary.
- `history` and `llm.types` define the message and completion data structures
  shared by local and remote callers.

`coding_assistant.tools` owns local tool implementations. Tool calls are
executed by `AgentSession` and the core agent loop, not by the manager or the
JSON-RPC transport.

`coding_assistant.cli` owns the interactive terminal application. It builds an
agent, creates an `AgentSession` directly, and renders its runtime events in the
terminal. The terminal UI itself is not a JSON-RPC client.

`coding_assistant.remote` owns JSON-RPC framing and protocol serialization.

- `jsonrpc.py` contains the custom ACP-inspired JSON-RPC helpers and common payload
  validation.
- `protocol.py` converts between core messages/session updates and JSON-RPC
  payloads.
- `client.py` exposes `WorkerClient` on top of shared JSON-RPC connection machinery.
- `websocket_server.py` contains the async WebSocket server loop.

`coding_assistant.manager` owns durable web-session state and manager-facing
RPC.

- `SessionStore` persists sessions and complete messages in SQLite.
- `ManagerService` validates opaque scope metadata, loads canonical history,
  and reserves one active worker per session.
- `manager.server` adapts `ManagerService` to JSON-RPC.
- `workspace.py` derives managed session paths from the session id.

`coding_assistant.worker` owns the worker-side remote runtime. A worker receives
manager-provided session state, creates one `AgentSession`, streams
`session/update` notifications, and returns status and metadata from one
private `_worker/run` request. Workers do not persist canonical history.

## Modes

### CLI

```text
terminal UI
  -> cli
  -> AgentSession
  -> tools / model
  -> AgentSession events
  -> terminal renderer
```

The CLI is the simplest runtime path. It creates and owns one local
`AgentSession`. History is in memory for that process unless a caller adds a
separate persistence layer.

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
- SQLite stores complete message history and session metadata.
- Workers hold only one active in-memory `AgentSession`.
- The manager persists each complete worker message before publishing it and
  keeps the active assistant draft in memory.
- One manager uses the SQLite database, and one worker may be active for each
  session.

This avoids split durable history between manager and worker. The worker can be
discarded and recreated from manager-provided state.

## Protocol Boundary

The project uses a custom JSON-RPC protocol based on ACP shapes. It is not a
complete ACP implementation. See `docs/rpc.md` for method details.

JSON-RPC is only the process/container boundary. Inside the application, code
should use core dataclasses such as `BaseMessage`, `AgentSessionEvent`, and
`SessionUpdate` rather than passing raw protocol dictionaries around.

Every remotely observed `AgentSession` emits provisional `session/update`
notifications independently of whether this connection submitted a prompt.
Live `session/prompt` and private `_worker/run` responses carry run status.
The manager persists complete messages.

Assistant streaming uses one message id per model attempt. Deltas create a
provisional draft, and the complete assistant message finalizes that same id.
An internal retry starts a new id without adding retry/reset/status concepts to
the wire protocol.

Private `_worker/*` methods are manager/worker operations. They deliberately do
not emulate a live session. Browser-facing or application-facing code should
use the public session methods documented in `docs/rpc.md`.

## Direction

Keep the architecture simple:

- Put agent execution behavior in `core`.
- Put terminal behavior in `cli`.
- Put JSON-RPC framing and conversion in `remote`.
- Put durable managed-session state in `manager`.
- Put one-session worker execution in `worker`.
- Do not add application-specific concepts to `coding_assistant`; use opaque
  scope ids at the manager boundary.
