from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from uuid import uuid4

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from coding_assistant.remote.protocol import (
    CancelCommand,
    CommandAcceptedMessage,
    ErrorMessage,
    NotReadyMessage,
    PromptCommand,
    StateMessage,
    WorkerToSupervisorMessage,
    message_to_json,
    parse_worker_message,
)


class RemoteWorkerConnection:
    """One live websocket connection to a worker."""

    def __init__(
        self,
        *,
        endpoint: str,
        websocket: ClientConnection,
        on_event: Callable[[WorkerToSupervisorMessage], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
    ) -> None:
        self.endpoint = endpoint
        self._websocket = websocket
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self._send_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[CommandAcceptedMessage | NotReadyMessage | ErrorMessage]] = {}
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._receive_task = asyncio.create_task(self._receive_loop())

    @classmethod
    async def open(
        cls,
        *,
        endpoint: str,
        on_event: Callable[[WorkerToSupervisorMessage], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
    ) -> RemoteWorkerConnection:
        """Open a worker connection and wait for its initial state handshake."""
        websocket = await connect(endpoint)
        connection = cls(
            endpoint=endpoint,
            websocket=websocket,
            on_event=on_event,
            on_disconnect=on_disconnect,
        )
        try:
            await connection._ready
        except Exception:
            await connection.close()
            raise
        return connection

    async def prompt(self, prompt: str) -> CommandAcceptedMessage | NotReadyMessage | ErrorMessage:
        """Send one prompt command and wait for the worker's direct reply."""
        request_id = uuid4().hex
        return await self._send_request(
            request_id=request_id,
            message=PromptCommand(request_id=request_id, prompt=prompt),
        )

    async def cancel(self) -> CommandAcceptedMessage | NotReadyMessage | ErrorMessage:
        """Request cancellation of the active worker run."""
        request_id = uuid4().hex
        return await self._send_request(
            request_id=request_id,
            message=CancelCommand(request_id=request_id),
        )

    async def close(self) -> None:
        """Close the websocket and wait for receive-loop cleanup to finish."""
        await self._websocket.close()
        with suppress(asyncio.CancelledError):
            await self._receive_task

    async def _send_request(
        self,
        *,
        request_id: str,
        message: PromptCommand | CancelCommand,
    ) -> CommandAcceptedMessage | NotReadyMessage | ErrorMessage:
        """Track one in-flight command until the matching response arrives."""
        future: asyncio.Future[CommandAcceptedMessage | NotReadyMessage | ErrorMessage] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending[request_id] = future
        async with self._send_lock:
            await self._websocket.send(message_to_json(message))
        return await future

    async def _receive_loop(self) -> None:
        """Route replies to callers and forward streamed worker events to callbacks."""
        try:
            async for raw_message in self._websocket:
                message = parse_worker_message(raw_message)
                request_id = getattr(message, "request_id", None)
                if (
                    request_id is not None
                    and request_id in self._pending
                    and isinstance(message, (CommandAcceptedMessage, NotReadyMessage, ErrorMessage))
                ):
                    future = self._pending.pop(request_id)
                    if not future.done():
                        future.set_result(message)
                    continue
                if isinstance(message, StateMessage) and not self._ready.done():
                    self._ready.set_result(None)
                elif isinstance(message, ErrorMessage) and not self._ready.done():
                    self._ready.set_exception(RuntimeError(message.message))
                await self._on_event(message)
        except ConnectionClosed:
            pass
        finally:
            if not self._ready.done():
                self._ready.set_exception(RuntimeError(f"Worker {self.endpoint} disconnected."))
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(RuntimeError(f"Worker {self.endpoint} disconnected."))
            self._pending.clear()
            await self._on_disconnect(self.endpoint)
