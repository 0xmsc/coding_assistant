from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Self

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.remote.jsonrpc import (
    ACP_PROTOCOL_VERSION,
    JsonObject,
    STOP_REASON_CANCELLED,
    STOP_REASON_END_TURN,
    jsonrpc_notification,
    jsonrpc_request,
    parse_jsonrpc_message,
)
from coding_assistant.remote.limits import WEBSOCKET_MAX_SIZE
from coding_assistant.remote.protocol import session_update_from_jsonrpc_update


def _client_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"


def _finish_title(metadata: object) -> str | None:
    if not isinstance(metadata, dict):
        return None
    title = metadata.get("title")
    if not isinstance(title, str):
        return None
    stripped = title.strip()
    return stripped or None


@dataclass(frozen=True)
class WorkerCompletion:
    stop_reason: str
    title: str | None = None


class _JsonRpcClient:
    """Shared JSON-RPC connection, request correlation, and update decoding."""

    def __init__(
        self,
        *,
        endpoint: str,
        websocket: ClientConnection,
        on_update: Callable[[SessionUpdate], Awaitable[None]] | None = None,
        on_disconnect: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._websocket = websocket
        self._on_update = on_update
        self._on_disconnect = on_disconnect
        self._send_lock = asyncio.Lock()
        self._next_request_id = 1
        self._pending_requests: dict[int, asyncio.Future[JsonObject]] = {}
        self._fatal_error: str | None = None
        self._receive_task = asyncio.create_task(self._receive_loop())

    @classmethod
    async def connect(
        cls,
        *,
        endpoint: str,
        on_update: Callable[[SessionUpdate], Awaitable[None]] | None = None,
        on_disconnect: Callable[[str], Awaitable[None]] | None = None,
        auth_token: str | None = None,
    ) -> Self:
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token is not None else None
        websocket = await connect(endpoint, additional_headers=headers, max_size=WEBSOCKET_MAX_SIZE)
        return cls(
            endpoint=endpoint,
            websocket=websocket,
            on_update=on_update,
            on_disconnect=on_disconnect,
        )

    async def initialize(self) -> None:
        await self._request(
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "coding-assistant",
                    "title": "Coding Assistant",
                    "version": _client_version(),
                },
            },
        )

    async def close(self) -> None:
        await self._websocket.close()
        with suppress(asyncio.CancelledError):
            await self._receive_task

    async def _cancel_session(self, session_id: str) -> None:
        async with self._send_lock:
            await self._websocket.send(
                jsonrpc_notification(
                    "session/cancel",
                    {"sessionId": session_id},
                ),
            )

    def _next_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def _create_request_future(self, request_id: int) -> asyncio.Future[JsonObject]:
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        return future

    async def _request(self, method: str, params: JsonObject) -> JsonObject:
        if self._fatal_error is not None:
            raise RuntimeError(self._fatal_error)
        request_id = self._next_id()
        future = self._create_request_future(request_id)
        try:
            async with self._send_lock:
                if self._fatal_error is not None:
                    raise RuntimeError(self._fatal_error)
                await self._websocket.send(jsonrpc_request(request_id, method, params))
            return await future
        except ConnectionClosed as exc:
            self._pending_requests.pop(request_id, None)
            if future.done():
                future.exception()
            raise RuntimeError(self._fatal_error or f"Remote {self.endpoint} disconnected.") from exc

    async def _receive_loop(self) -> None:
        try:
            async for raw_message in self._websocket:
                payload = parse_jsonrpc_message(raw_message)
                if "method" in payload:
                    await self._handle_notification(payload)
                    continue
                await self._handle_response(payload)
        except ConnectionClosed:
            pass
        finally:
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError(f"Remote {self.endpoint} disconnected."))
            self._pending_requests.clear()
            if self._on_disconnect is not None:
                await self._on_disconnect(self.endpoint)

    async def _handle_notification(self, payload: JsonObject) -> None:
        if payload.get("method") != "session/update":
            return
        params = payload.get("params", {})
        if not isinstance(params, dict):
            return
        update = params.get("update", {})
        if not isinstance(update, dict):
            return
        session_update = session_update_from_jsonrpc_update(update)
        if session_update is not None and self._on_update is not None:
            await self._on_update(session_update)

    async def _handle_response(self, payload: JsonObject) -> None:
        response_id = payload.get("id")
        error = payload.get("error")
        if response_id is None and isinstance(error, dict):
            message = str(error.get("message", "Unknown remote protocol error."))
            self._fatal_error = message
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError(message))
            self._pending_requests.clear()
            return
        if not isinstance(response_id, int):
            return
        response_future = self._pending_requests.pop(response_id, None)
        if response_future is None:
            return
        if isinstance(error, dict):
            response_future.set_exception(RuntimeError(str(error.get("message", "Unknown remote protocol error."))))
        else:
            response_future.set_result(payload.get("result", {}))


class WorkerClient(_JsonRpcClient):
    """Client for one manager-owned, one-shot worker run."""

    async def run(self, params: JsonObject) -> WorkerCompletion:
        if not isinstance(params.get("sessionId"), str):
            raise ValueError("_worker/run params must include sessionId.")
        result = await self._request("_worker/run", params)
        stop_reason = result.get("stopReason")
        if stop_reason not in {STOP_REASON_END_TURN, STOP_REASON_CANCELLED}:
            raise RuntimeError("_worker/run response did not include a stop reason.")
        return WorkerCompletion(
            stop_reason=stop_reason,
            title=_finish_title(result.get("_meta")),
        )

    async def cancel(self, *, session_id: str) -> None:
        await self._cancel_session(session_id)
