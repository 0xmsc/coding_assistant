from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import AgentSession, CompletionStreamer
from coding_assistant.llm.types import BaseMessage, Tool
from coding_assistant.remote.acp import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_SERVER,
    JsonObject,
    jsonrpc_error,
    parse_jsonrpc_message,
    session_id_from_params,
)
from coding_assistant.remote.control import RemoteAgentController, RemoteAgentInfo, RemoteControlledSession
from coding_assistant.remote.limits import WEBSOCKET_MAX_SIZE
from coding_assistant.remote.protocol import messages_from_jsonrpc


@dataclass(frozen=True)
class WorkerServer:
    endpoint: str
    _finished: asyncio.Event

    async def wait_finished(self) -> None:
        await self._finished.wait()


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    model: str
    tools: list[Tool]
    completion_streamer: CompletionStreamer | None = None
    finish_metadata_provider: Callable[[], JsonObject | None] | None = None


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


async def _handle_start(
    *,
    websocket: ServerConnection,
    controller: RemoteAgentController,
    runtime: WorkerRuntimeConfig,
    response_id: int | str | None,
    params: JsonObject,
) -> None:
    if response_id is None:
        await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "_session/start must be a request."))
        return
    if controller.has_session:
        await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Worker session is already started."))
        return

    try:
        session_id = session_id_from_params(params)
        base_version = _base_version_from_params(params)
        _workspace_from_params(params)
        messages = _messages_from_params(params)
    except ValueError as exc:
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
        return

    session = AgentSession(
        history=messages,
        model=runtime.model,
        tools=runtime.tools,
        completion_streamer=runtime.completion_streamer,
    )
    await controller.start_session(
        websocket=websocket,
        response_id=response_id,
        controlled_session=RemoteControlledSession(
            session_id=session_id,
            session=session,
            base_version=base_version,
            base_message_count=len(messages),
        ),
    )


@asynccontextmanager
async def start_session_worker_server(
    *,
    runtime: WorkerRuntimeConfig,
    host: str = "127.0.0.1",
    port: int = 0,
) -> AsyncIterator[WorkerServer]:
    finished = asyncio.Event()

    async def mark_finished() -> None:
        finished.set()

    async def handle_connection(websocket: ServerConnection) -> None:
        controller = RemoteAgentController(
            agent_info=RemoteAgentInfo(name="coding-assistant-worker", title="Coding Assistant Worker"),
            busy_message="Session already has an active prompt.",
            unopened_message="_session/start must be called first.",
            finish_metadata_provider=runtime.finish_metadata_provider,
            on_run_finished=mark_finished,
        )
        sender_task: asyncio.Task[None] | None = None

        async def handle_worker_method(method: str, response_id: int | str | None, params: JsonObject) -> bool:
            if method != "_session/start":
                return False
            await _handle_start(
                websocket=websocket,
                controller=controller,
                runtime=runtime,
                response_id=response_id,
                params=params,
            )
            return True

        try:
            async for raw_message in websocket:
                try:
                    payload = parse_jsonrpc_message(raw_message)
                except ValueError:
                    await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC payload."))
                    continue
                await controller.handle_jsonrpc_message(
                    websocket=websocket,
                    payload=payload,
                    on_unhandled=handle_worker_method,
                )
                if controller.has_session and sender_task is None:
                    sender_task = asyncio.create_task(controller.publish_session_events(websocket=websocket))
        except ConnectionClosed:
            pass
        finally:
            if sender_task is not None:
                sender_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sender_task
            session = controller.session
            if session is not None:
                await session.close()

    async with serve(handle_connection, host, port, max_size=WEBSOCKET_MAX_SIZE) as server:

        async def close_after_finished() -> None:
            await finished.wait()
            server.close()
            await server.wait_closed()

        closer_task = asyncio.create_task(close_after_finished())
        socket = server.sockets[0]
        bound_port = socket.getsockname()[1]
        endpoint = f"ws://{host}:{bound_port}"
        try:
            yield WorkerServer(endpoint=endpoint, _finished=finished)
        finally:
            closer_task.cancel()
            with suppress(asyncio.CancelledError):
                await closer_task
