# RPC Protocol

`coding-assistant` uses a custom JSON-RPC 2.0 protocol based on the Agent
Client Protocol (ACP). It uses ACP method names and payload shapes where they
fit, but it is not a complete ACP implementation.

The interactive CLI does not use this protocol for its own terminal UI. It
creates and drives `AgentSession` directly, and currently exposes a local
single-session remote endpoint as an additional endpoint for other clients.

The web chat integration should not introduce a separate remote mode. The core
and CLI use `AgentSessionEvent` for in-process runtime events. Remote endpoints,
the web manager, workers/subagents, and persistence use `SessionUpdate` for the
smaller set of updates that cross a process boundary. JSON-RPC is their single
transport serialization.

The minimum Chat WebUI contract is:

- `initialize`
- `session/list`
- `session/new`
- `session/load`
- `session/rename`
- `session/prompt`
- `session/cancel`
- `session/update`

Everything else is optional until the UI has a concrete need for it.

## Topology

Browser clients should connect through an authenticated application backend,
not directly to `coding-assistant`.

```text
browser
  -> authenticated application backend WebSocket
  -> coding_assistant manager service
  -> temporary per-prompt worker container
```

The application backend owns browser authentication. The manager owns canonical
session state and worker lifecycle. Each active prompt runs in a temporary
worker container with the session's managed workspace mounted at `/workspace`
and session attachments mounted read-only at `/attachments`. Worker processes
are one-shot and exit after returning the completed `_worker/run` result.

The CLI path remains direct:

```text
terminal UI
  -> AgentSession
  -> AgentSession events
  -> CLI renderer
```

The CLI may continue to expose a local remote endpoint for one live session,
but that endpoint should use the same protocol helpers as manager-controlled
remote workers/subagents. The CLI terminal UI itself does not need to become a
JSON-RPC client.

## Compatibility Boundary

This protocol is ACP-inspired:

- It uses JSON-RPC 2.0 envelopes.
- It uses ACP method names where possible.
- It uses ACP-compatible content blocks and `session/update` payloads where
  practical.

This protocol also has `coding-assistant` extensions:

- Backend-injected `params._meta.scopeId` session scoping.
- Manager-owned session directories derived from `sessionId`.
- Manager-owned SQLite persistence.
- Session-scoped injected skills plus session-scoped and prompt-scoped private
  worker environment.
- Private manager/worker `_worker/*` methods.
- Per-active-prompt worker containers.

Do not treat this document as a promise of complete ACP compatibility.

## Transport

The current CLI can expose a local single-session WebSocket adapter and print
its endpoint:

```text
Remote endpoint: ws://127.0.0.1:43123
```

The web manager service listens on a stable host and port configured for the
deployment. Each WebSocket text frame contains one JSON-RPC message encoded as
UTF-8 JSON. Binary frames are not part of the protocol.

For web deployment, terminate browser WebSockets at the authenticated
application backend. The backend forwards JSON-RPC messages to the manager
service after injecting trusted scope metadata. Backend-to-manager WebSocket
connections must include the configured manager secret:

```text
Authorization: Bearer <manager-secret>
```

Browser clients and worker containers must not receive the manager secret. For
the Docker manager deployment, see [deployment.md](deployment.md) for
container environment variables, runtime requirements, and security notes.

## JSON-RPC Envelope

Requests:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {}
}
```

Successful responses:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {}
}
```

Error responses:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Request params must be an object."
  }
}
```

Notifications:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {}
}
```

Notifications have no `id` and must not receive responses.
Client-to-server notifications are only supported for `session/cancel`;
notification forms of request-only methods are ignored without side effects.

## Error Codes

| Code | Name | Meaning |
| --- | --- | --- |
| `-32600` | Invalid Request | The frame is not a valid JSON-RPC request for this server. |
| `-32601` | Method Not Found | The method is not supported. |
| `-32602` | Invalid Params | The method parameters are missing or malformed. |
| `-32000` | Server Error | The request is valid, but the session cannot perform it. |

## Session Scope

Sessions are scoped by a trusted application-provided scope key. The manager
does not know how the embedding application identifies callers. It only stores
and enforces an opaque `scopeId`.

The browser must never be trusted to provide or override scope. Before
forwarding scope-scoped methods to the manager, the application backend must:

1. Validate the caller through its own auth/session flow.
2. Strip any browser-provided `params._meta.scopeId`.
3. Inject `params._meta.scopeId`.

Example forwarded request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/list",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    }
  }
}
```

The manager requires `params._meta.scopeId` for all `session/*` methods:
`session/list`, `session/new`, `session/load`, `session/upload_file`,
`session/download_attachment`, `session/rename`, `session/set_model`,
`session/prompt`, `session/cancel`, and `session/delete`.

## Managed Workspaces

The web manager does not accept arbitrary `cwd` values for v1 sessions.

Each session gets a managed session directory derived from the session id:

```text
manager session root:     /data/sessions
manager workspace path:   /data/sessions/<sessionId>/workspace
manager attachments path: /data/sessions/<sessionId>/attachments
host session path:        $CODING_ASSISTANT_HOST_DATA_DIR/sessions/<sessionId>
worker workspace mount:   /workspace
worker attachments mount: /attachments (read-only)
```

The worker process starts with `/workspace` as its working directory. This is
the session's execution context, not a promise of ACP `cwd` compatibility. The
existing `cwd` examples from older versions of this document do not apply to
the web manager service.

If a session's derived workspace is missing on `session/load`, the manager
fails clearly instead of recreating an empty workspace or silently switching to
another path.

Workspace seeding/import is out of scope for v1.

## Persistence

The manager owns canonical session state in SQLite. Worker containers hold an
in-memory `AgentSession` only while active.

V1 uses these tables:

```text
sessions
  session_id text primary key
  scope_id text not null
  title text null
  created_at text not null
  updated_at text not null
  metadata_json text not null default '{}'
  worker_env_json text not null default '{}'

session_messages
  id integer primary key autoincrement
  session_id text not null
  role text not null
  payload_json text not null
  created_at text not null

session_attachments
  id integer primary key autoincrement
  attachment_id text not null
  session_id text not null
  message_id integer not null references session_messages(id) on delete cascade
  sequence integer not null
  name text not null
  mime_type text not null
  size integer not null
  path text not null
  sha256 text not null
  created_at text not null
```

`metadata_json` is public session metadata returned in `_meta`.
`worker_env_json` is private manager state and must not be returned by
`session/list`, `session/load`, or other public session metadata responses.
`session_messages` is the source of truth for LLM history and replay. Every
message row is replayed to clients in insertion order, including system
messages, active-run messages, and upload messages. Session attachments are
metadata rows in
`session_attachments` linked to the upload message row that introduced them.

Do not add a durable `session_runs` table for v1. Active runs live in manager
memory while a worker is running.

## Commit Semantics

The manager reserves the session for one active worker and persists every
complete message before publishing it. Assistant deltas remain in manager
memory until the worker emits the complete message.

A concurrent `session/load` replays durable messages followed by the current
in-memory draft and running status. A new model attempt replaces the previous
draft. Terminal run status removes any unfinished draft. The worker result
supplies status and metadata, not a transcript.

Deployments must run only one manager against a SQLite database. Within that
manager, the per-session reservation prevents concurrent workers or mutations
for the same session. These ownership constraints make transcript versions
unnecessary.

## Connection Lifecycle

Clients must call `initialize` before session methods. After initialization,
clients can list sessions, create a session, or load a session with history
replay.

The private one-shot worker endpoint is not a public session client and does
not use this initialization handshake.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": 1,
    "clientCapabilities": {},
    "clientInfo": {
      "name": "my-client",
      "title": "My Client",
      "version": "1.0.0"
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": 1,
    "agentCapabilities": {
      "loadSession": true,
      "promptCapabilities": {
        "image": true,
        "embeddedContext": true
      }
    },
    "agentInfo": {
      "name": "coding-assistant",
      "title": "Coding Assistant",
      "version": "0.0.0"
    },
    "authMethods": []
  }
}
```

## Methods

### initialize

Negotiates protocol version and returns manager capabilities.
The server supports protocol version 1 and rejects a client maximum below 1
without marking the connection initialized.

Parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `protocolVersion` | integer | yes | Highest protocol version supported by the client. |
| `clientCapabilities` | object | no | Client capabilities. Currently ignored. |
| `clientInfo` | object | no | Client metadata. Currently ignored. |

Response result:

| Field | Type | Description |
| --- | --- | --- |
| `protocolVersion` | integer | Negotiated protocol version. |
| `agentCapabilities.loadSession` | boolean | Advertise `true` when `session/load` is supported. |
| `agentCapabilities.promptCapabilities.image` | boolean | Image prompt blocks are accepted. |
| `agentCapabilities.promptCapabilities.embeddedContext` | boolean | Embedded resource prompt blocks are accepted. |
| `agentInfo` | object | Agent name, title, and version. |
| `authMethods` | array | Empty for v1 because the embedding application owns browser authentication. |

### model/list

Lists model IDs available from the configured OpenAI-compatible provider. This
method is authenticated by the manager connection but is not session-scoped.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "model/list",
  "params": {}
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "models": [
      {
        "id": "openai/gpt-5.1"
      }
    ]
  }
}
```

The manager caches provider results briefly. If provider discovery fails and no
cached list is available, `models` is empty. The manager does not choose a
default model; clients must persist a session model with `session/set_model`
before sending `session/prompt`.

### session/list

Lists persisted sessions in `params._meta.scopeId`.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/list",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    }
  }
}
```

Parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `_meta.scopeId` | string | yes | Trusted opaque scope injected by the application backend. |

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "sessions": [
      {
        "sessionId": "sess_abc123",
        "title": "Fix failing tests",
        "updatedAt": "2026-06-03T10:15:00Z",
        "_meta": {
          "model": "gpt-5.1-codex-mini"
        }
      }
    ],
    "nextCursor": null
  }
}
```

The current implementation returns every session in the scope and does not
paginate results. `nextCursor` is always `null`.

Session metadata must not expose arbitrary host workspace paths.

### session/new

Creates a new session in `params._meta.scopeId`. The manager may also accept
private session-scoped worker setup under `_meta`.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/new",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123",
      "skills": [
        {
          "name": "apps-api",
          "description": "Use apps REST APIs.",
          "files": {
            "SKILL.md": "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n",
            "references/calories.md": "calories reference"
          }
        }
      ]
    }
  }
}
```

`_meta.workerEnv` is optional private session state for durable values. The
manager stores it in SQLite outside public session metadata and passes it to
worker tool processes for prompts in that session. Keys must be uppercase
environment variable names and values must be strings. Send short-lived
credentials as prompt-scoped worker env on `session/prompt`; prompt-scoped
values override session-scoped values for that run only.

`_meta.skills` is an optional array of injected skill bundles. The manager
writes each bundle to `workspace/.agents/skills/<name>` in the session directory
before building the initial system message. Workers then discover those skills
through normal workspace skill loading. Each skill requires a valid `name`, a non-empty
`description`, and a `files` object containing `SKILL.md`. File paths must be
relative paths inside the skill directory.

These `_meta` fields are setup inputs only. They are not copied into public
session metadata and are not returned by `session/list` or `session/load`.

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "sessionId": "sess_new"
  }
}
```

The manager creates `/data/sessions/<sessionId>` inside the manager container,
writes any injected skills under `workspace/.agents/skills`, and initializes the
session transcript with system instructions for the managed workspace.

External `cwd` input is ignored or rejected until a future workspace import
feature is designed.

### session/load

Loads an existing session in `params._meta.scopeId` and replays its
transcript through `session/update` notifications before the request resolves.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "session/load",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123"
  }
}
```

During load, the server streams canonical message and attachment replay
updates:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "message_added",
      "message": {
        "id": "msg_123",
        "role": "user",
        "content": "Fix the failing tests.",
        "createdAt": "2026-06-23T19:30:00Z"
      }
    }
  }
}
```

When replay is complete:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "sessionId": "sess_abc123",
    "title": "Fix failing tests",
    "updatedAt": "2026-06-03T10:15:00Z",
    "_meta": {
      "model": "gpt-5.1-codex-mini"
    }
  }
}
```

The manager fails clearly when the session does not exist, belongs to another
scope, or has a missing derived workspace.

### session/rename

Renames an existing session in `params._meta.scopeId`. A `null` or blank title
clears the custom title.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "session/rename",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123",
    "title": "Fix failing tests"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "sessionId": "sess_abc123",
    "title": "Fix failing tests",
    "updatedAt": "2026-06-03T10:20:00Z",
    "_meta": {
      "model": "gpt-5.1-codex-mini"
    }
  }
}
```

The manager fails when the session does not exist or belongs to another scope.

### session/set_model

Sets the model used for future prompts in an existing session in
`params._meta.scopeId`. Changing the model updates session metadata but does
not rewrite transcript history.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "session/set_model",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123",
    "model": "openai/gpt-5.1"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "sessionId": "sess_abc123",
    "title": "Fix failing tests",
    "updatedAt": "2026-06-03T10:25:00Z",
    "_meta": {
      "model": "openai/gpt-5.1"
    }
  }
}
```

The manager rejects models that are not in `model/list` and rejects model
changes while the session has an active prompt.

### session/prompt

Submits one prompt to a session in `params._meta.scopeId`. The request
remains open until the run finishes, is cancelled, or fails. During the run,
the server sends `session/update` notifications.

The session must already have a model in metadata. Use `session/set_model`
before prompting new sessions or older sessions without a stored model.
`_meta.workerEnv` may provide private prompt-scoped environment variables. The
manager passes them to the worker for this prompt only, without persisting or
returning them. Prompt-scoped values override session-scoped values with the
same key for the run.

`_meta.skills` is not accepted on `session/prompt`; injected skills are session
setup and must be provided on `session/new`.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "session/prompt",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123",
      "workerEnv": {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "fresh-secret-token"
      }
    },
    "sessionId": "sess_abc123",
    "prompt": [
      {
        "type": "text",
        "text": "Inspect the repository."
      }
    ]
  }
}
```

Response on completion:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "stopReason": "end_turn"
  }
}
```

Response on cancellation:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "stopReason": "cancelled"
  }
}
```

Only one active prompt may run per session. Different sessions may run
concurrently. For each active prompt, the manager starts a temporary worker
container. The worker receives the session history, runs tools inside
`/workspace`, returns the completed run result, and exits normally.

Prompt blocks follow ACP-compatible content shapes where practical:

```json
{
  "type": "text",
  "text": "Inspect the repository."
}
```

```json
{
  "type": "image",
  "mimeType": "image/png",
  "data": "base64-encoded-image"
}
```

```json
{
  "type": "resource",
  "resource": {
    "uri": "file:///workspace/app.py",
    "mimeType": "text/x-python",
    "text": "print('hello')"
  }
}
```

### session/upload_file

Uploads one bounded file into the session attachments directory in
`params._meta.scopeId`. The manager validates scope, file name, MIME type, and
size, writes the bytes under the session attachments directory, commits an
upload message and linked attachment metadata, emits message and attachment
events through `session/update`, and returns attachment metadata. If `name` is
`null` or blank, the manager derives a generic filename with an extension from
the MIME type.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "session/upload_file",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123",
    "name": "meal.jpg",
    "mimeType": "image/jpeg",
    "data": "base64-encoded-file"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": {
    "attachment": {
      "id": "att_abc123",
      "name": "meal.jpg",
      "mimeType": "image/jpeg",
      "size": 12345,
      "path": "/attachments/att_abc123-meal.jpg",
      "sha256": "..."
    },
    "session": {
      "sessionId": "sess_abc123",
      "updatedAt": "2026-06-23T19:30:00Z",
      "_meta": {}
    }
  }
}
```

Workers do not receive attachment bytes automatically. For images, use the
worker `load_image(path)` tool with the returned `attachment.path` before
reasoning from the image. For PDFs, extract text in the workspace with
`pdftotext`, then inspect the extracted text. Text-like attachments can be read
with shell commands or Python scripts run through the shell from their
`/attachments/...` path.

### session/download_attachment

Downloads a stored session attachment. The manager authorizes the session by
`params._meta.scopeId`, finds the persisted attachment row, reads the file
from the manager-owned attachments directory, recomputes its SHA-256 hash, and
returns bytes only if the hash still matches the stored attachment metadata.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "method": "session/download_attachment",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123",
    "attachmentId": "att_abc123"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 9,
  "result": {
    "attachment": {
      "id": "att_abc123",
      "name": "meal.jpg",
      "mimeType": "image/jpeg",
      "size": 12345,
      "path": "/attachments/att_abc123-meal.jpg",
      "sha256": "..."
    },
    "encoding": "base64",
    "data": "base64-encoded-file"
  }
}
```

### session/cancel

Cancels the current run for a session in `params._meta.scopeId`. This
method may be sent as either a notification or a request.

Notification:

```json
{
  "jsonrpc": "2.0",
  "method": "session/cancel",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123"
  }
}
```

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "method": "session/cancel",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123"
  }
}
```

Response when sent as a request:

```json
{
  "jsonrpc": "2.0",
  "id": 8,
  "result": null
}
```

The original `session/prompt` request should later resolve with
`stopReason: cancelled` when cancellation reaches the active run.

### session/delete

Deletes a session in `params._meta.scopeId`, including persisted messages,
attachment metadata, uploaded attachment files, and the manager-owned workspace
directory. The manager rejects cross-scope deletes and rejects deletion while
the session has an active prompt; clients should cancel the prompt first.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "session/delete",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
    },
    "sessionId": "sess_abc123"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "result": null
}
```

## Notifications

### session/update

The server sends transcript history through JSON-RPC notifications.
`session/load`, `session/prompt`, and `session/upload_file`
all use this same update path; RPC responses acknowledge commands and return
metadata, but clients should not render transcript content from those
responses.

History replay starts with `history_reset`, sends persisted messages and linked
attachments as update events, and ends with `history_complete`:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "history_reset"
    }
  }
}
```

Message added:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "message_added",
      "message": {
        "id": "msg_123",
        "role": "assistant",
        "content": "I'll inspect the repository.",
        "createdAt": "2026-06-23T19:30:00Z"
      }
    }
  }
}
```

Attachment added:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "attachment_added",
      "attachment": {
        "id": "att_abc123",
        "name": "meal.jpg",
        "mimeType": "image/jpeg",
        "size": 12345,
        "path": "/attachments/att_abc123-meal.jpg",
        "sha256": "...",
        "createdAt": "2026-06-23T19:30:00Z"
      }
    }
  }
}
```

Assistant text delta. The first delta for an unknown `messageId` creates a
provisional assistant draft on the client:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "message_delta",
      "messageId": "msg_assistant_1",
      "appendText": "I'll inspect the repository."
    }
  }
}
```

Session metadata update:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "session_updated",
      "session": {
        "sessionId": "sess_abc123",
        "title": "Fix failing tests",
        "updatedAt": "2026-06-23T19:35:00Z",
        "_meta": {
          "model": "openai/gpt-5.1"
        }
      }
    }
  }
}
```

Run status update:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "run_updated",
      "run": {
        "runId": "run_abc123",
        "sessionId": "sess_abc123",
        "status": "completed",
        "startedAt": "2026-06-23T19:34:00Z",
        "updatedAt": "2026-06-23T19:35:00Z",
        "endedAt": "2026-06-23T19:35:00Z",
        "stopReason": "end_turn"
      }
    }
  }
}
```

Supported `sessionUpdate` values:

| Value | Description |
| --- | --- |
| `history_reset` | Clear local transcript before replay. |
| `message_added` | Add a complete message, or finalize a provisional draft with the same id. |
| `message_delta` | Append streamed text to a provisional assistant draft, creating it when its id is first seen. |
| `attachment_added` | Add an attachment metadata record linked to a message. |
| `session_updated` | Replace or merge session metadata such as title, model, and updated time. |
| `run_updated` | Report an active or finished prompt run and its status. |
| `history_complete` | Replay is complete. |

Each model attempt uses one message id. Its deltas arrive first, followed by a
complete `message_added` with the same id. The complete message replaces the
draft and includes any tool calls. If the model layer retries, the next attempt
uses a new message id; seeing that new provisional id supersedes the older
unfinished draft. There is no retry, reset, or status update in the wire
protocol.

Complete live messages are inserted before the manager publishes them.
Assistant deltas are buffered in memory for active-run replay. Tool calls and tool results are
represented as assistant/tool messages; clients that want tool cards or image
previews derive those UI elements from message content.
The manager sends `session_updated` after persisted session metadata changes,
including rename, model changes, uploads, and prompt commits.
It sends `run_updated` when a prompt starts, completes, is cancelled, or fails.
The run status is `running`, `completed`, `cancelled`, or `failed`; terminal
updates may also include `endedAt`, `stopReason`, or `error`.

## Internal Worker Methods

Manager-to-worker traffic uses the same JSON-RPC framing and message/update
serialization as external remote traffic, but it has a smaller lifecycle. A
live endpoint exposes an existing session through `session/new`,
`session/prompt`, and `session/cancel`. A one-shot worker accepts exactly one
private `_worker/run` request and then exits. The two paths do not share a
controller state machine.

### _worker/run

Creates a worker session from manager-owned state, executes one run, and
returns its completed result. The request remains pending while provisional
`session/update` notifications stream on the same connection.

Request params:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `sessionId` | string | yes | Session id. |
| `messages` | array | yes | Model-visible committed history from SQLite. |
| `prompt` | array | yes | ACP-shaped content blocks for the run's initial prompt. |

### `_worker/run` result

The worker's response to `_worker/run` reports only completion status and
metadata. Messages are delivered through `session/update` notifications before
this response.

Result fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `stopReason` | string | yes | `end_turn` or `cancelled`. |
| `_meta` | object | no | Implementation metadata such as title updates. |

Workers may include `_meta.title` to request a persisted session title update.
The manager stores the title and emits `session_updated` to
clients.

## Module Naming

Use honest module names:

- `remote/jsonrpc.py` for generic JSON-RPC helpers.
- `remote/protocol.py` for the custom `coding-assistant` remote protocol.

Do not put manager session paths, scope metadata, worker prompt-result payloads,
or private `_worker/*` methods into a module named as if it were pure ACP.

## Current Limitations

- The current CLI-owned remote endpoint is local-only and single-session.
- No permission request round trip before tools execute.
- The CLI-owned live endpoint has no model or configuration methods.
- No prompt queueing or steering through the remote protocol.
- Workspace seeding/import is not part of v1.

## Testing Expectations

The protocol must be tested without real model provider credentials by default.

Required coverage:

- JSON-RPC envelope validation and error responses.
- Scope isolation for list, new, load, prompt, and cancel.
- Application-backend stripping and injection of `params._meta.scopeId`.
- Session creation and derived workspace creation.
- Missing workspace failure on `session/load`.
- Transcript replay order.
- Prompt streaming through `session/update`.
- Tool call and tool call update shapes.
- Versioned prompt-result success and stale prompt-result rejection.
- Worker crash before run completion does not advance SQLite history.
- Concurrent prompts on different sessions.
- Rejection of a second active prompt for one session.
- Model listing, default fallback, per-session model changes, and per-prompt
  worker model selection.
- Worker container smoke test with `/workspace` as cwd.
- Fake OpenAI-compatible streaming provider for integration tests.

## Optional Extensions

Optional methods can be added when the UI needs them:

| Method | Purpose |
| --- | --- |
| `session/resume` | Reconnect to a session without replaying history. |
| `session/close` | Free live resources for an active session. |
| `session/request_permission` | Ask the browser to approve risky tool actions. |

Add private JSON-RPC methods with a leading underscore only for behavior this
custom protocol needs and ACP does not define.
