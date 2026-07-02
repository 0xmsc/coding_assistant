from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from uuid import uuid4

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import AgentSession
from coding_assistant.remote.acp import ERROR_INVALID_REQUEST, ERROR_SERVER, jsonrpc_error, parse_jsonrpc_message
from coding_assistant.remote.control import RemoteAgentController, RemoteAgentInfo, RemoteControlledSession
from coding_assistant.remote.limits import WEBSOCKET_MAX_SIZE


@dataclass(frozen=True)
class WorkerServer:
    """Handle returned by the server context, exposing the chosen endpoint."""

    endpoint: str


@asynccontextmanager
async def start_worker_server(
    *,
    session: AgentSession,
) -> AsyncIterator[WorkerServer]:
    """Serve one JSON-RPC remote-control connection for a live session."""
    connection_lock = asyncio.Lock()
    active_connection: ServerConnection | None = None
    session_id = f"sess_{uuid4().hex}"

    async def handle_connection(websocket: ServerConnection) -> None:
        nonlocal active_connection

        async with connection_lock:
            if active_connection is not None:
                await websocket.send(
                    jsonrpc_error(None, ERROR_SERVER, "Session already has a remote client connected."),
                )
                await websocket.close()
                return
            active_connection = websocket

        controller = RemoteAgentController(
            agent_info=RemoteAgentInfo(name="coding-assistant", title="Coding Assistant"),
            controlled_session=RemoteControlledSession(
                session_id=session_id,
                session=session,
                base_message_count=len(session.history),
            ),
            supports_session_new=True,
            busy_message="Session is busy. Wait for the current turn or cancel it.",
        )
        sender_task = asyncio.create_task(controller.publish_session_events(websocket=websocket))
        try:
            async for raw_message in websocket:
                try:
                    payload = parse_jsonrpc_message(raw_message)
                except ValueError:
                    await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC payload."))
                    continue
                await controller.handle_jsonrpc_message(websocket=websocket, payload=payload)
        except ConnectionClosed:
            pass
        finally:
            sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await sender_task
            async with connection_lock:
                if active_connection is websocket:
                    active_connection = None

    async with serve(handle_connection, "127.0.0.1", 0, max_size=WEBSOCKET_MAX_SIZE) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        yield WorkerServer(endpoint=endpoint)
