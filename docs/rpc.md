# RPC Protocol

`coding-assistant` exposes an ACP-shaped JSON-RPC 2.0 protocol over WebSocket
for controlling coding-agent sessions remotely.

The current worker implementation is local-only and single-session. The web app
protocol described here is the target RPC contract: it supports multiple
sessions and uses native ACP methods where ACP already defines the operation.
Private `_session/*` methods are reserved for web-specific gaps.

## Transport

The local CLI worker starts a WebSocket server and prints its endpoint:

```text
Remote endpoint: ws://127.0.0.1:43123
```

Each WebSocket text frame contains one JSON-RPC message encoded as UTF-8 JSON.
Binary frames are not part of the protocol.

For a web deployment, terminate browser WebSockets at an authenticated backend.
The backend owns process lifetime, session storage, workspace access, and tool
permission policy.

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

## Error Codes

The server currently uses these JSON-RPC error codes:

| Code | Name | Meaning |
| --- | --- | --- |
| `-32600` | Invalid Request | The frame is not a valid JSON-RPC request for this server. |
| `-32601` | Method Not Found | The method is not supported. |
| `-32602` | Invalid Params | The method parameters are missing or malformed. |
| `-32000` | Server Error | The request is valid, but the session cannot perform it. |

## Connection Lifecycle

Clients must call `initialize` before using session methods. After
initialization, clients can list existing sessions, create a session, load a
session with history replay, or resume a session without history replay.

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
        "list": {},
        "resume": {},
        "close": {}
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

List existing sessions:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/list",
  "params": {
    "cwd": "/home/user/project"
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "sessions": [
      {
        "sessionId": "sess_abc123",
        "cwd": "/home/user/project",
        "title": "Fix failing tests",
        "updatedAt": "2026-06-03T10:15:00Z"
      }
    ]
  }
}
```

Create a new session:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/new",
  "params": {
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "sessionId": "sess_new"
  }
}
```

Load an existing session when the UI needs transcript replay:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "session/load",
  "params": {
    "sessionId": "sess_abc123",
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

Resume an existing session when the UI already has transcript state:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "session/resume",
  "params": {
    "sessionId": "sess_abc123",
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

## Methods

### initialize

Negotiates protocol version and returns agent capabilities.

Request:

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

Parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `protocolVersion` | integer | yes | Highest ACP protocol version supported by the client. |
| `clientCapabilities` | object | no | Client capabilities. Currently ignored. |
| `clientInfo` | object | no | Client metadata. Currently ignored. |

Response result:

| Field | Type | Description |
| --- | --- | --- |
| `protocolVersion` | integer | Negotiated protocol version. |
| `agentCapabilities.loadSession` | boolean | Advertise `true` when `session/load` is supported. |
| `agentCapabilities.sessionCapabilities.list` | object | Present when `session/list` is supported. |
| `agentCapabilities.sessionCapabilities.resume` | object | Present when `session/resume` is supported. |
| `agentCapabilities.sessionCapabilities.close` | object | Present when `session/close` is supported. |
| `agentCapabilities.promptCapabilities.image` | boolean | Image prompt blocks are accepted. |
| `agentCapabilities.promptCapabilities.embeddedContext` | boolean | Embedded resource prompt blocks are accepted. |
| `agentInfo` | object | Agent name, title, and version. |
| `authMethods` | array | Always empty in the current implementation. |

### session/list

Lists persisted sessions known to the authenticated backend. This is a native
ACP method.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/list",
  "params": {
    "cwd": "/home/user/project",
    "cursor": "opaque-page-token"
  }
}
```

Parameters:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `cwd` | string | no | Absolute path used to filter sessions by workspace. |
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
        "cwd": "/home/user/project",
        "title": "Fix failing tests",
        "updatedAt": "2026-06-03T10:15:00Z",
        "_meta": {
          "messageCount": 12
        }
      }
    ],
    "nextCursor": "next-page-token"
  }
}
```

Response fields:

| Field | Type | Description |
| --- | --- | --- |
| `sessions` | array | Session metadata objects. Empty when no sessions match. |
| `nextCursor` | string | Optional opaque cursor for the next page. |

Session fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `sessionId` | string | yes | Stable session identifier. |
| `cwd` | string | yes | Absolute primary workspace path. |
| `title` | string | no | User-visible session title. |
| `updatedAt` | string | no | ISO 8601 last activity timestamp. |
| `_meta` | object | no | Implementation-specific metadata such as message count. |

### session/new

Creates a new session. This is a native ACP method.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/new",
  "params": {
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

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

The returned `sessionId` must be included in prompt and cancel messages.

### session/load

Loads an existing session and replays its transcript to the client via
`session/update` notifications before the request resolves. This is a native ACP
method and should be used when the browser needs to hydrate a chat transcript
from server state.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "session/load",
  "params": {
    "sessionId": "sess_abc123",
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

During load, the server streams historical messages:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "user_message_chunk",
      "content": {
        "type": "text",
        "text": "Fix the failing tests."
      }
    }
  }
}
```

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "agent_message_chunk",
      "content": {
        "type": "text",
        "text": "I found the failing assertion."
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
  "result": null
}
```

### session/resume

Reattaches to an existing session without replaying transcript history. This is
a native ACP method and should be used after browser reconnect when the UI
already has the transcript.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "session/resume",
  "params": {
    "sessionId": "sess_abc123",
    "cwd": "/home/user/project",
    "mcpServers": []
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {}
}
```

### session/prompt

Submits one prompt to a session. The request remains open until the run
finishes, is cancelled, or fails. During the run, the server sends
`session/update` notifications.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/prompt",
  "params": {
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
  "id": 3,
  "result": {
    "stopReason": "end_turn"
  }
}
```

Response on cancellation:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "stopReason": "cancelled"
  }
}
```

Only one active prompt should run per session. A backend may allow different
sessions to run concurrently.

Prompt blocks:

Text prompt block:

```json
{
  "type": "text",
  "text": "Inspect the repository."
}
```

Image prompt block with base64 data:

```json
{
  "type": "image",
  "mimeType": "image/png",
  "data": "base64-encoded-image"
}
```

Image prompt block with a URI:

```json
{
  "type": "image",
  "mimeType": "image/png",
  "uri": "https://example.com/screenshot.png"
}
```

Embedded resource prompt block:

```json
{
  "type": "resource",
  "resource": {
    "uri": "file:///home/user/project/app.py",
    "mimeType": "text/x-python",
    "text": "print('hello')"
  }
}
```

Resource link prompt block:

```json
{
  "type": "resource_link",
  "uri": "file:///home/user/project/app.py",
  "name": "app.py"
}
```

Text blocks are passed as text. Image blocks are passed as image URLs. Resource
and resource link blocks are rendered into text context.

If the prompt contains exactly one text block, it is passed as a string to the
session. Otherwise it is passed as structured content.

### session/cancel

Cancels the current run for the session. This method may be sent as either a
notification or a request.

Notification:

```json
{
  "jsonrpc": "2.0",
  "method": "session/cancel",
  "params": {
    "sessionId": "sess_abc123"
  }
}
```

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "session/cancel",
  "params": {
    "sessionId": "sess_abc123"
  }
}
```

Response when sent as a request:

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": null
}
```

The original `session/prompt` request should later resolve with this result if
cancellation reaches the active run:

```json
{
  "stopReason": "cancelled"
}
```

### session/close

Closes an active session and frees its live resources. This is a native ACP
method.

Request:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "session/close",
  "params": {
    "sessionId": "sess_abc123"
  }
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {}
}
```

Closing a session should cancel any active run for that session before freeing
resources.

## Notifications

### session/update

The server streams run output and tool status through JSON-RPC notifications.

Assistant text delta:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "agent_message_chunk",
      "content": {
        "type": "text",
        "text": "I'll inspect the repository."
      }
    }
  }
}
```

Tool call announced:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "tool_call",
      "toolCallId": "call_1",
      "title": "shell_execute",
      "kind": "other",
      "status": "pending",
      "rawInput": {
        "command": "ls"
      }
    }
  }
}
```

Tool call updated:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "tool_call_update",
      "toolCallId": "call_1",
      "status": "completed",
      "title": "shell_execute",
      "kind": "other",
      "rawOutput": {
        "exit_code": 0
      },
      "content": [
        {
          "type": "content",
          "content": {
            "type": "text",
            "text": "Command completed."
          }
        }
      ]
    }
  }
}
```

Supported `sessionUpdate` values:

| Value | Description |
| --- | --- |
| `user_message_chunk` | A user message chunk, primarily used during `session/load` replay. |
| `agent_message_chunk` | A streamed assistant text delta. |
| `tool_call` | A tool call was requested. |
| `tool_call_update` | A tool call lifecycle update. |
| `session_info_update` | Session title, updated timestamp, or metadata changed. |

Session metadata update:

```json
{
  "jsonrpc": "2.0",
  "method": "session/update",
  "params": {
    "sessionId": "sess_abc123",
    "update": {
      "sessionUpdate": "session_info_update",
      "title": "Fix failing tests",
      "updatedAt": "2026-06-03T10:20:00Z",
      "_meta": {
        "messageCount": 14
      }
    }
  }
}
```

## Current Limitations

- No authentication or authorization.
- The current CLI worker implementation is still local-only and single-session.
- No public HTTP API.
- No fork support.
- No permission request round trip before tools execute.
- No model or configuration methods.
- No prompt queueing or steering through the remote protocol.

## Recommended Web App Extensions

For a web chat UI, prefer native ACP methods first:

| Need | Native ACP method |
| --- | --- |
| List sessions | `session/list` |
| Create a session | `session/new` |
| Hydrate transcript | `session/load` |
| Reconnect without replay | `session/resume` |
| Send prompt | `session/prompt` |
| Cancel run | `session/cancel` |
| Close live resources | `session/close` |

Add private JSON-RPC methods with a leading underscore only for behavior that
ACP does not currently define. This avoids collisions with future standard ACP
methods.

Useful additions:

| Method | Purpose |
| --- | --- |
| `_session/state` | Fetch running, queued, usage, model, and title state. |
| `_session/queue_prompt` | Queue a prompt for later execution. |
| `_session/steer` | Inject a steering prompt at the next agent boundary. |
| `_session/set_title` | Set the user-visible session title. |

For internet-facing deployment, put these methods behind a backend that owns
authentication, process lifetime, session storage, workspace access, and tool
permission policy. Do not expose the current local worker server directly to
browser users.
