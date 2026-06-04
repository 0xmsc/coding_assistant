from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import AgentSession, CompletionStreamer
from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    SessionUpdate,
    prompt_result_from_update,
    session_updates_from_agent_event,
)
from coding_assistant.llm.types import BaseMessage, Tool
from coding_assistant.remote.acp import (
    ACP_PROTOCOL_VERSION,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    JsonObject,
    initialize_result,
    jsonrpc_error,
    jsonrpc_notification,
    jsonrpc_result,
    parse_jsonrpc_message,
    prompt_content_from_acp,
)
from coding_assistant.remote.protocol import (
    messages_from_jsonrpc,
    messages_to_jsonrpc,
    session_update_to_jsonrpc_update,
)


@dataclass(frozen=True)
class WorkerServer:
    endpoint: str


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    model: str
    tools: list[Tool]
    completion_streamer: CompletionStreamer | None = None


@dataclass
class _WorkerState:
    initialized: bool = False
    session: AgentSession | None = None
    session_id: str | None = None
    base_version: int | None = None
    base_message_count: int = 0
    workspace: Path | None = None
    active_prompt_request_id: int | str | None = None
    active_prompt_source: str | None = None


def _agent_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"


def _response_id(payload: JsonObject) -> int | str | None:
    request_id = payload.get("id")
    return request_id if isinstance(request_id, int | str) else None


def _params_from_payload(payload: JsonObject) -> JsonObject:
    params = payload.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("Request params must be an object.")
    return params


def _session_id_from_params(params: JsonObject) -> str:
    session_id = params.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Request params must include sessionId.")
    return session_id


def _base_version_from_params(params: JsonObject) -> int:
    base_version = params.get("baseVersion")
    if not isinstance(base_version, int):
        raise ValueError("_session/start requires integer baseVersion.")
    return base_version


def _workspace_from_params(params: JsonObject) -> Path:
    workspace = params.get("workspace")
    if not isinstance(workspace, str) or not workspace:
        raise ValueError("_session/start requires workspace.")
    path = Path(workspace)
    if not path.is_dir():
        raise ValueError(f"Workspace does not exist: {workspace}")
    return path


def _messages_from_params(params: JsonObject) -> list[BaseMessage]:
    messages = params.get("messages")
    if not isinstance(messages, list):
        raise ValueError("_session/start requires messages array.")
    if not all(isinstance(message, dict) for message in messages):
        raise ValueError("_session/start messages must be objects.")
    return messages_from_jsonrpc(messages)


def _update_matches_active_prompt(update: SessionUpdate, source: str | None) -> bool:
    if source is None:
        return False
    return getattr(update, "source", None) == source


def _commit_notification(
    *,
    session_id: str,
    base_version: int,
    messages: list[BaseMessage],
    stop_reason: str,
) -> str:
    return jsonrpc_notification(
        "_session/commit",
        {
            "sessionId": session_id,
            "baseVersion": base_version,
            "messages": messages_to_jsonrpc(messages),
            "stopReason": stop_reason,
        },
    )


async def _send_session_update(
    *,
    websocket: ServerConnection,
    session_id: str,
    update: SessionUpdate,
) -> None:
    payload_update = session_update_to_jsonrpc_update(update)
    if payload_update is None:
        return
    await websocket.send(
        jsonrpc_notification(
            "session/update",
            {
                "sessionId": session_id,
                "update": payload_update,
            },
        ),
    )


async def _publish_session_events(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
) -> None:
    if state.session is None or state.session_id is None:
        return

    async with state.session.subscribe() as queue:
        while True:
            event = await queue.get()
            for update in session_updates_from_agent_event(event):
                if isinstance(update, AgentMessageChunkUpdate) and state.active_prompt_source is not None:
                    await _send_session_update(websocket=websocket, session_id=state.session_id, update=update)
                    continue

                if not _update_matches_active_prompt(update, state.active_prompt_source):
                    continue

                await _send_session_update(websocket=websocket, session_id=state.session_id, update=update)

                result = prompt_result_from_update(update)
                if result is None:
                    continue

                request_id = state.active_prompt_request_id
                if request_id is None or state.base_version is None:
                    continue

                if result.stop_reason is not None:
                    await websocket.send(jsonrpc_result(request_id, {"stopReason": result.stop_reason}))
                    await websocket.send(
                        _commit_notification(
                            session_id=state.session_id,
                            base_version=state.base_version,
                            messages=_new_history_messages(state.session.history, state.base_message_count),
                            stop_reason=result.stop_reason,
                        ),
                    )
                else:
                    await websocket.send(jsonrpc_error(request_id, ERROR_SERVER, result.error or "Run failed."))

                state.active_prompt_request_id = None
                state.active_prompt_source = None


def _new_history_messages(history: list[BaseMessage], base_message_count: int) -> list[BaseMessage]:
    return history[base_message_count:]


async def _handle_initialize(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
    response_id: int | str | None,
    params: JsonObject,
) -> None:
    if response_id is None:
        await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "initialize must be a request."))
        return
    protocol_version = params.get("protocolVersion")
    if not isinstance(protocol_version, int):
        await websocket.send(
            jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "initialize requires an integer protocolVersion."),
        )
        return
    state.initialized = True
    await websocket.send(
        jsonrpc_result(
            response_id,
            initialize_result(
                agent_name="coding-assistant-worker",
                agent_title="Coding Assistant Worker",
                agent_version=_agent_version(),
            )
            | {"protocolVersion": min(protocol_version, ACP_PROTOCOL_VERSION)},
        ),
    )


async def _handle_start(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
    runtime: WorkerRuntimeConfig,
    response_id: int | str | None,
    params: JsonObject,
) -> None:
    if response_id is None:
        await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "_session/start must be a request."))
        return
    if state.session is not None:
        await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Worker session is already started."))
        return

    session_id = _session_id_from_params(params)
    base_version = _base_version_from_params(params)
    workspace = _workspace_from_params(params)
    messages = _messages_from_params(params)
    state.session = AgentSession(
        history=messages,
        model=runtime.model,
        tools=runtime.tools,
        completion_streamer=runtime.completion_streamer,
    )
    state.session_id = session_id
    state.base_version = base_version
    state.base_message_count = len(messages)
    state.workspace = workspace
    await websocket.send(jsonrpc_result(response_id, {"sessionId": session_id}))


async def _handle_prompt(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
    response_id: int | str | None,
    params: JsonObject,
) -> None:
    if response_id is None:
        await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/prompt must be a request."))
        return
    if state.session is None or state.session_id is None:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "_session/start must be called first."))
        return
    if _session_id_from_params(params) != state.session_id:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
        return
    prompt_blocks = params.get("prompt")
    if not isinstance(prompt_blocks, list):
        await websocket.send(
            jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "session/prompt requires a prompt array.")
        )
        return
    if state.active_prompt_request_id is not None:
        await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Session already has an active prompt."))
        return
    try:
        prompt_content = prompt_content_from_acp(prompt_blocks)
    except ValueError as exc:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
        return

    source = f"worker:{state.session_id}:{response_id}"
    state.active_prompt_request_id = response_id
    state.active_prompt_source = source
    accepted = await state.session.enqueue_prompt_if_idle(prompt_content, source=source)
    if not accepted:
        state.active_prompt_request_id = None
        state.active_prompt_source = None
        await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Session is busy."))


async def _handle_cancel(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
    response_id: int | str | None,
    params: JsonObject,
) -> None:
    if state.session is None or state.session_id is None:
        return
    if _session_id_from_params(params) != state.session_id:
        return
    await state.session.cancel_current_run()
    if response_id is not None:
        await websocket.send(jsonrpc_result(response_id, None))


async def _handle_jsonrpc_message(
    *,
    websocket: ServerConnection,
    state: _WorkerState,
    runtime: WorkerRuntimeConfig,
    payload: JsonObject,
) -> None:
    response_id = _response_id(payload)
    method = payload.get("method")
    if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
        return

    try:
        params = _params_from_payload(payload)
    except ValueError as exc:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
        return

    if method == "initialize":
        await _handle_initialize(websocket=websocket, state=state, response_id=response_id, params=params)
        return
    if not state.initialized:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "initialize must be called first."))
        return
    if method == "_session/start":
        await _handle_start(websocket=websocket, state=state, runtime=runtime, response_id=response_id, params=params)
        return
    if method == "session/prompt":
        await _handle_prompt(websocket=websocket, state=state, response_id=response_id, params=params)
        return
    if method == "session/cancel":
        await _handle_cancel(websocket=websocket, state=state, response_id=response_id, params=params)
        return
    await websocket.send(jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported worker method: {method}"))


@asynccontextmanager
async def start_session_worker_server(
    *,
    runtime: WorkerRuntimeConfig,
    host: str = "127.0.0.1",
    port: int = 0,
) -> AsyncIterator[WorkerServer]:
    async def handle_connection(websocket: ServerConnection) -> None:
        state = _WorkerState()
        sender_task: asyncio.Task[None] | None = None
        try:
            async for raw_message in websocket:
                try:
                    payload = parse_jsonrpc_message(raw_message)
                except ValueError:
                    await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC payload."))
                    continue
                await _handle_jsonrpc_message(websocket=websocket, state=state, runtime=runtime, payload=payload)
                if state.session is not None and sender_task is None:
                    sender_task = asyncio.create_task(_publish_session_events(websocket=websocket, state=state))
        except ConnectionClosed:
            pass
        finally:
            if sender_task is not None:
                sender_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sender_task
            if state.session is not None:
                await state.session.close()

    async with serve(handle_connection, host, port) as server:
        socket = server.sockets[0]
        bound_port = socket.getsockname()[1]
        endpoint = f"ws://{host}:{bound_port}"
        yield WorkerServer(endpoint=endpoint)
