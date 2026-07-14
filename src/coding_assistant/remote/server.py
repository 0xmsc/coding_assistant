from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from uuid import uuid4

from websockets.asyncio.server import ServerConnection

from coding_assistant.core.agent_session import AgentSession
from coding_assistant.remote.jsonrpc import ERROR_SERVER, JsonObject, jsonrpc_error
from coding_assistant.remote.control import RemoteAgentController, RemoteAgentInfo
from coding_assistant.remote.websocket_server import receive_jsonrpc_messages, serve_jsonrpc_websocket


@dataclass(frozen=True)
class RemoteServer:
    """Handle returned by the server context, exposing the chosen endpoint."""

    endpoint: str


@asynccontextmanager
async def start_worker_server(
    *,
    session: AgentSession,
) -> AsyncIterator[RemoteServer]:
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
            session_id=session_id,
            session=session,
            busy_message="Session is busy. Wait for the current turn or cancel it.",
        )
        sender_task = asyncio.create_task(controller.publish_session_events(websocket=websocket))
        try:

            async def handle_message(payload: JsonObject) -> None:
                await controller.handle_jsonrpc_message(websocket=websocket, payload=payload)

            await receive_jsonrpc_messages(websocket=websocket, on_message=handle_message)
        finally:
            sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await sender_task
            await controller.close()
            async with connection_lock:
                if active_connection is websocket:
                    active_connection = None

    async with serve_jsonrpc_websocket(handler=handle_connection) as server:
        yield RemoteServer(endpoint=server.endpoint)
