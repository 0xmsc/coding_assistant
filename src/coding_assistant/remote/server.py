from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any
from uuid import uuid4

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    ToolCallUpdateEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import ContentDeltaEvent
from coding_assistant.remote.acp import (
    ACP_PROTOCOL_VERSION,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    STOP_REASON_CANCELLED,
    STOP_REASON_END_TURN,
    agent_message_update,
    initialize_result,
    jsonrpc_error,
    jsonrpc_result,
    parse_jsonrpc_message,
    prompt_content_from_acp,
    tool_call_lifecycle_update,
    tool_call_notification,
    tool_call_update_notification,
)


@dataclass(frozen=True)
class WorkerServer:
    """Handle returned by the server context, exposing the chosen endpoint."""

    endpoint: str


@dataclass
class _ConnectionState:
    initialized: bool = False
    session_opened: bool = False
    active_prompt_request_id: int | str | None = None
    active_prompt_source: str | None = None


def _agent_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"


def _tool_call_raw_input(arguments: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _event_matches_active_prompt(event: AgentSessionEvent, source: str | None) -> bool:
    if source is None:
        return False
    event_source = getattr(event, "source", None)
    return event_source == source


def _tool_calls_to_updates(event: ToolCallsEvent) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for tool_call in event.message.tool_calls:
        updates.append(
            tool_call_notification(
                tool_call_id=tool_call.id,
                title=tool_call.function.name or "tool_call",
                kind="other",
                status="pending",
                raw_input=_tool_call_raw_input(tool_call.function.arguments),
            )
        )
    return updates


def _tool_call_update_to_payload(event: ToolCallUpdateEvent) -> dict[str, Any]:
    return tool_call_lifecycle_update(
        tool_call_id=event.event.tool_call_id,
        status=event.event.status,
        title=event.event.title,
        kind=event.event.kind,
        raw_input=event.event.raw_input,
        raw_output=event.event.raw_output,
        content_text=event.event.content,
    )


async def _publish_session_events(
    *,
    websocket: ServerConnection,
    session: AgentSession,
    session_id: str,
    state: _ConnectionState,
) -> None:
    async with session.subscribe() as queue:
        while True:
            event = await queue.get()
            if isinstance(event, ContentDeltaEvent):
                if state.active_prompt_source is None:
                    continue
                await websocket.send(agent_message_update(session_id, event.content))
                continue

            if not _event_matches_active_prompt(event, state.active_prompt_source):
                continue

            if isinstance(event, ToolCallsEvent):
                for update in _tool_calls_to_updates(event):
                    await websocket.send(tool_call_update_notification(session_id, update))
                continue

            if isinstance(event, ToolCallUpdateEvent):
                await websocket.send(tool_call_update_notification(session_id, _tool_call_update_to_payload(event)))
                continue

            request_id = state.active_prompt_request_id
            if request_id is None:
                continue

            if isinstance(event, RunFinishedEvent):
                await websocket.send(jsonrpc_result(request_id, {"stopReason": STOP_REASON_END_TURN}))
                state.active_prompt_request_id = None
                state.active_prompt_source = None
                continue

            if isinstance(event, RunCancelledEvent):
                await websocket.send(jsonrpc_result(request_id, {"stopReason": STOP_REASON_CANCELLED}))
                state.active_prompt_request_id = None
                state.active_prompt_source = None
                continue

            if isinstance(event, RunFailedEvent):
                await websocket.send(jsonrpc_error(request_id, ERROR_SERVER, event.error))
                state.active_prompt_request_id = None
                state.active_prompt_source = None


async def _handle_jsonrpc_message(
    *,
    websocket: ServerConnection,
    session: AgentSession,
    session_id: str,
    state: _ConnectionState,
    payload: dict[str, Any],
) -> None:
    request_id = payload.get("id")
    response_id = request_id if isinstance(request_id, int | str) else None
    method = payload.get("method")
    params = payload.get("params", {})

    if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
        return
    if params is None:
        params = {}
    if not isinstance(params, dict):
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Request params must be an object."))
        return

    if method == "initialize":
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "initialize must be a request."))
            return
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, int):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "initialize requires an integer protocolVersion.")
            )
            return
        negotiated_version = min(protocol_version, ACP_PROTOCOL_VERSION)
        state.initialized = True
        await websocket.send(
            jsonrpc_result(
                response_id,
                initialize_result(
                    agent_name="coding-assistant",
                    agent_title="Coding Assistant",
                    agent_version=_agent_version(),
                )
                | {"protocolVersion": negotiated_version},
            )
        )
        return

    if method == "session/new":
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/new must be a request."))
            return
        if not state.initialized:
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "initialize must be called before session/new.")
            )
            return
        state.session_opened = True
        await websocket.send(jsonrpc_result(response_id, {"sessionId": session_id}))
        return

    if method == "session/prompt":
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/prompt must be a request."))
            return
        if not state.initialized or not state.session_opened:
            await websocket.send(
                jsonrpc_error(
                    response_id,
                    ERROR_INVALID_REQUEST,
                    "initialize and session/new must be completed before session/prompt.",
                )
            )
            return
        if params.get("sessionId") != session_id:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "session/prompt requires a prompt array.")
            )
            return
        if state.active_prompt_request_id is not None:
            await websocket.send(
                jsonrpc_error(response_id, ERROR_SERVER, "This ACP connection already has an active prompt turn.")
            )
            return

        try:
            prompt_content = prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
            return

        prompt_source = f"acp:{session_id}:{response_id}:{uuid4().hex}"
        state.active_prompt_request_id = response_id
        state.active_prompt_source = prompt_source
        accepted = await session.enqueue_prompt_if_idle(prompt_content, source=prompt_source)
        if not accepted:
            state.active_prompt_request_id = None
            state.active_prompt_source = None
            await websocket.send(
                jsonrpc_error(response_id, ERROR_SERVER, "Session is busy. Wait for the current turn or cancel it.")
            )
            return
        return

    if method == "session/cancel":
        if not state.initialized or not state.session_opened:
            return
        if params.get("sessionId") != session_id:
            return
        await session.cancel_current_run()
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
        return

    await websocket.send(jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported ACP method: {method}"))


@asynccontextmanager
async def start_worker_server(
    *,
    session: AgentSession,
) -> AsyncIterator[WorkerServer]:
    """Serve one ACP websocket connection at a time for a live session."""
    connection_lock = asyncio.Lock()
    active_connection: ServerConnection | None = None
    session_id = f"sess_{uuid4().hex}"

    async def handle_connection(websocket: ServerConnection) -> None:
        nonlocal active_connection

        async with connection_lock:
            if active_connection is not None:
                await websocket.send(
                    jsonrpc_error(None, ERROR_SERVER, "Session already has a remote client connected.")
                )
                await websocket.close()
                return
            active_connection = websocket

        state = _ConnectionState()
        sender_task = asyncio.create_task(
            _publish_session_events(
                websocket=websocket,
                session=session,
                session_id=session_id,
                state=state,
            )
        )
        try:
            async for raw_message in websocket:
                try:
                    payload = parse_jsonrpc_message(raw_message)
                except ValueError:
                    await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC payload."))
                    continue
                await _handle_jsonrpc_message(
                    websocket=websocket,
                    session=session,
                    session_id=session_id,
                    state=state,
                    payload=payload,
                )
        except ConnectionClosed:
            pass
        finally:
            sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await sender_task
            async with connection_lock:
                if active_connection is websocket:
                    active_connection = None

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        yield WorkerServer(endpoint=endpoint)
