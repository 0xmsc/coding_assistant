# RPC Protocol

`coding-assistant` uses a custom JSON-RPC 2.0 protocol based on the Agent
Client Protocol (ACP). It uses ACP method names and payload shapes where they
fit, but it is not a complete ACP implementation.

The interactive CLI does not use this protocol for its own terminal UI. It
creates and drives `AgentSession` directly, and currently exposes a local
single-session remote endpoint as an additional endpoint for other clients.

The web chat integration should not introduce a separate remote mode or event
model. CLI rendering, the CLI-owned local remote endpoint, the web manager,
remote workers/subagents, and persistence should share the same internal
session update and committed-message models. JSON-RPC is the single remote
transport serialization for components that cross a process or container
boundary.

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
and session attachments mounted read-only at `/attachments`.

The CLI path remains direct:

```text
terminal UI
  -> AgentSession
  -> shared session update model
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
- Session-scoped private worker environment and workspace-backed injected skills.
- Private manager/worker `_session/*` methods.
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

The manager requires `params._meta.scopeId` for scoped methods and uses it
for `session/list`, `session/new`, `session/load`, `session/rename`,
`session/prompt`, and `session/cancel`.

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

V1 uses two tables:

```text
sessions
  session_id text primary key
  scope_id text not null
  title text null
  version integer not null default 0
  created_at text not null
  updated_at text not null
  metadata_json text not null default '{}'
  worker_env_json text not null default '{}'

session_messages
  id integer primary key autoincrement
  session_id text not null
  version integer not null
  role text not null
  payload_json text not null
  created_at text not null
```

`metadata_json` is public session metadata returned in `_meta`.
`worker_env_json` is private manager state and must not be returned by
`session/list`, `session/load`, or other public session metadata responses.

Do not add a durable `session_runs` table for v1. Active runs live in manager
memory while a worker is running.

## Commit Semantics

Worker stream output is provisional until the manager persists a completed
turn.

1. The manager starts a worker from history version `N`.
2. The worker creates an in-memory `AgentSession`.
3. The worker streams live `session/update` notifications.
4. The worker sends `_session/commit` with `baseVersion: N`.
5. The manager atomically verifies `sessions.version == N`.
6. The manager inserts committed messages and updates the session to version
   `N + 1`.

Stale commits are rejected. If a worker crashes before commit, SQLite history
is not advanced.

## Connection Lifecycle

Clients must call `initialize` before session methods. After initialization,
clients can list sessions, create a session, or load a session with history
replay.

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
      "sessionCapabilities": {
        "list": {}
      },
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
| `agentCapabilities.sessionCapabilities.list` | object | Present when `session/list` is supported. |
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
    },
    "cursor": "opaque-page-token"
  }
}
```

Parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `_meta.scopeId` | string | yes | Trusted opaque scope injected by the application backend. |
| `cursor` | string | no | Opaque pagination cursor from a previous response. |

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
          "version": 1,
          "model": "gpt-5.1-codex-mini",
          "messageCount": 12
        }
      }
    ],
    "nextCursor": "next-page-token"
  }
}
```

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
      "workerEnv": {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "secret-token"
      },
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

`_meta.workerEnv` is optional private session state. The manager stores it in
SQLite outside public session metadata and passes it to worker tool processes
for prompts in that session. Keys must be uppercase environment variable names
and values must be strings.

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

During load, the server streams item replay updates:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "item_added",
      "item": {
        "id": "item_abc123",
        "kind": "message",
        "sequence": 1,
        "payload": {
          "role": "user",
          "content": "Fix the failing tests."
        }
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
      "version": 1,
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
      "version": 1,
      "model": "gpt-5.1-codex-mini"
    }
  }
}
```

The manager fails when the session does not exist or belongs to another scope.

### session/set_model

Sets the model used for future prompts in an existing session in
`params._meta.scopeId`. Changing the model updates session metadata but does
not rewrite transcript history or increment the transcript version.

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
      "version": 1,
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
Session-scoped worker setup such as `_meta.workerEnv` and `_meta.skills` is
accepted only by `session/new`; prompt metadata is not used to change worker
environment variables or available skills.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "session/prompt",
  "params": {
    "_meta": {
      "scopeId": "tenant:abc123"
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
`/workspace`, and is removed after the prompt finishes.

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

Uploads one bounded file into the visible session attachments directory in
`params._meta.scopeId`. The manager validates scope, file name, MIME type, and
size, writes the bytes under the session attachments directory, commits
model-visible attachment context, emits a visible attachment item through
`session/update`, and returns attachment metadata. If `name` is `null` or
blank, the manager derives a generic filename with an extension from the MIME
type.

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
      "_meta": {
        "version": 2
      }
    }
  }
}
```

Workers do not receive attachment bytes automatically. For images, use the
worker `load_image(path)` tool with the returned `attachment.path` before
reasoning from the image. For PDFs, extract text in the workspace with
`pdftotext`, then inspect the extracted text. Text-like attachments can be read
with shell or Python from their `/attachments/...` path.

### session/download_attachment

Downloads a stored session attachment. The manager authorizes the session by
`params._meta.scopeId`, finds the persisted attachment item, reads the file
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

## Notifications

### session/update

The server sends all visible transcript history through JSON-RPC
notifications. `session/load`, `session/prompt`, and `session/upload_file`
all use this same update path; RPC responses acknowledge commands and return
metadata, but clients should not render transcript content from those
responses.

History replay starts with `history_reset`, sends one `item_added` per
persisted visible item, and ends with `history_complete`:

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

Visible item added:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "item_added",
      "item": {
        "id": "item_abc123",
        "kind": "attachment",
        "sequence": 3,
        "createdAt": "2026-06-23T19:30:00Z",
        "updatedAt": "2026-06-23T19:30:00Z",
        "payload": {
          "id": "att_abc123",
          "name": "meal.jpg",
          "mimeType": "image/jpeg",
          "size": 12345,
          "path": "/attachments/att_abc123-meal.jpg",
          "sha256": "..."
        }
      }
    }
  }
}
```

Assistant text delta for an existing message item:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "item_delta",
      "itemId": "item_assistant_1",
      "appendText": "I'll inspect the repository."
    }
  }
}
```

Tool call status patch:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "item_updated",
      "itemId": "item_tool_1",
      "patch": {
        "payload": {
          "status": "completed",
          "rawOutput": {
            "exit_code": 0
          },
          "content": "Command completed."
        }
      }
    }
  }
}
```

Supported `sessionUpdate` values:

| Value | Description |
| --- | --- |
| `history_reset` | Clear local visible transcript before replay. |
| `item_added` | Add a visible `message`, `tool_call`, or `attachment` item. |
| `item_delta` | Append text to an existing message item. |
| `item_updated` | Patch an existing item payload, primarily tool status/output. |
| `history_complete` | Replay is complete and includes the durable session version. |

Live streamed updates are provisional until the worker commits the completed
turn and the manager persists it.

## Internal Worker Methods

Manager-to-worker traffic uses the same JSON-RPC protocol family as external
remote traffic. This is not a second protocol mode: the CLI-owned local remote
endpoint and manager-controlled remote workers/subagents should share envelope
parsing, request/response/error handling, content blocks, `session/update`
serialization, and normalized update/commit models.

Private methods use the `_session/*` prefix only where the public session
methods do not define the manager/worker operation, such as starting a worker
from manager-owned state and completed-turn commit semantics.

### _session/start

Starts a worker session from manager-owned state.

Request params:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `sessionId` | string | yes | Session id. |
| `baseVersion` | integer | yes | History version used to start the worker session. |
| `messages` | array | yes | Model-visible committed history from SQLite. |
| `workspace` | string | yes | Worker workspace path, normally `/workspace`. |

### _session/commit

Sent by the worker when a prompt turn finishes and should become durable.

Notification params:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `sessionId` | string | yes | Session id. |
| `baseVersion` | integer | yes | History version used to start the run. |
| `messages` | array | yes | Newly committed model-visible messages. |
| `stopReason` | string | yes | `end_turn` or `cancelled`. |
| `usage` | object | no | Usage metadata. |
| `_meta` | object | no | Implementation metadata such as title updates. |

The manager rejects commits when `baseVersion` does not match the current
SQLite session version.

## Module Naming

Use honest module names:

- `remote/jsonrpc.py` for generic JSON-RPC helpers.
- `remote/protocol.py` for the custom `coding-assistant` remote protocol.

Do not put manager session paths, scope metadata, worker commits, or private
`_session/*` methods into a module named as if it were pure ACP.

## Current Limitations

- The current CLI-owned remote endpoint is local-only and single-session.
- No permission request round trip before tools execute.
- No model or configuration methods.
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
- Versioned commit success and stale commit rejection.
- Worker crash before commit does not advance SQLite history.
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
