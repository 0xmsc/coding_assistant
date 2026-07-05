from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from coding_assistant.remote.jsonrpc import ERROR_INVALID_REQUEST, JsonObject, jsonrpc_error, parse_jsonrpc_message
from coding_assistant.remote.limits import WEBSOCKET_MAX_SIZE


JsonRpcMessageHandler = Callable[[JsonObject], Awaitable[None]]
WebSocketConnectionHandler = Callable[[ServerConnection], Awaitable[None]]
ProcessRequest = Callable[[ServerConnection, Request], Response | None]


@dataclass(frozen=True)
class JsonRpcWebSocketServer:
    endpoint: str


async def receive_jsonrpc_messages(
    *,
    websocket: ServerConnection,
    on_message: JsonRpcMessageHandler,
) -> None:
    try:
        async for raw_message in websocket:
            try:
                payload = parse_jsonrpc_message(raw_message)
            except ValueError:
                await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "Invalid JSON-RPC payload."))
                continue
            await on_message(payload)
    except ConnectionClosed:
        pass


@asynccontextmanager
async def serve_jsonrpc_websocket(
    *,
    handler: WebSocketConnectionHandler,
    host: str = "127.0.0.1",
    port: int = 0,
    process_request: ProcessRequest | None = None,
    close_when: asyncio.Event | None = None,
) -> AsyncIterator[JsonRpcWebSocketServer]:
    async with serve(
        handler,
        host,
        port,
        process_request=process_request,
        max_size=WEBSOCKET_MAX_SIZE,
    ) as server:

        async def close_after_event() -> None:
            if close_when is None:
                return
            await close_when.wait()
            server.close()
            await server.wait_closed()

        closer_task = asyncio.create_task(close_after_event())
        socket = server.sockets[0]
        bound_port = socket.getsockname()[1]
        try:
            yield JsonRpcWebSocketServer(endpoint=f"ws://{host}:{bound_port}")
        finally:
            closer_task.cancel()
            with suppress(asyncio.CancelledError):
                await closer_task
