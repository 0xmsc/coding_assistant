from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from coding_assistant.remote.acp import (
    ACP_PROTOCOL_VERSION,
    JsonObject,
    jsonrpc_notification,
    jsonrpc_request,
    parse_jsonrpc_message,
    text_block,
)


@dataclass(frozen=True)
class RemoteContentDeltaEvent:
    content: str


@dataclass(frozen=True)
class RemoteToolCallEvent:
    title: str


@dataclass(frozen=True)
class RemoteToolCallUpdateEvent:
    status: str
    title: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class RemotePromptFinishedEvent:
    stop_reason: str


@dataclass(frozen=True)
class RemotePromptFailedEvent:
    message: str


RemoteClientEvent = (
    RemoteContentDeltaEvent
    | RemoteToolCallEvent
    | RemoteToolCallUpdateEvent
    | RemotePromptFinishedEvent
    | RemotePromptFailedEvent
)


@dataclass
class _ActivePrompt:
    request_id: int
    submission_future: asyncio.Future[None]


class RemoteWorkerConnection:
    """One live ACP websocket connection to a remote session."""

    def __init__(
        self,
        *,
        endpoint: str,
        websocket: ClientConnection,
        on_event: Callable[[RemoteClientEvent], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
    ) -> None:
        self.endpoint = endpoint
        self._websocket = websocket
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self._send_lock = asyncio.Lock()
        self._next_request_id = 1
        self._pending_requests: dict[int, asyncio.Future[JsonObject]] = {}
        self._session_id: str | None = None
        self._active_prompt: _ActivePrompt | None = None
        self._fatal_error: str | None = None
        self._receive_task = asyncio.create_task(self._receive_loop())

    @classmethod
    async def open(
        cls,
        *,
        endpoint: str,
        on_event: Callable[[RemoteClientEvent], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
    ) -> RemoteWorkerConnection:
        websocket = await connect(endpoint)
        connection = cls(
            endpoint=endpoint,
            websocket=websocket,
            on_event=on_event,
            on_disconnect=on_disconnect,
        )
        try:
            await connection._initialize()
        except Exception:
            await connection.close()
            raise
        return connection

    async def _initialize(self) -> None:
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
        result = await self._request(
            "session/new",
            {
                "cwd": os.getcwd(),
                "mcpServers": [],
            },
        )
        session_id = result.get("sessionId")
        if not isinstance(session_id, str):
            raise RuntimeError("ACP session/new response did not include a sessionId.")
        self._session_id = session_id

    async def prompt(self, prompt: str) -> str | None:
        if self._session_id is None:
            return "Remote session is not initialized."
        if self._active_prompt is not None:
            return "This remote connection already has an active prompt turn."

        request_id = self._next_id()
        response_future = self._create_request_future(request_id)
        submission_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._active_prompt = _ActivePrompt(request_id=request_id, submission_future=submission_future)

        async with self._send_lock:
            await self._websocket.send(
                jsonrpc_request(
                    request_id,
                    "session/prompt",
                    {
                        "sessionId": self._session_id,
                        "prompt": [text_block(prompt)],
                    },
                )
            )

        try:
            await asyncio.wait_for(asyncio.shield(submission_future), timeout=0.1)
        except TimeoutError:
            return None
        except Exception as exc:
            self._pending_requests.pop(request_id, None)
            self._active_prompt = None
            if not response_future.done():
                response_future.cancel()
            return str(exc)
        return None

    async def cancel(self) -> None:
        if self._session_id is None:
            raise RuntimeError("Remote session is not initialized.")
        async with self._send_lock:
            await self._websocket.send(
                jsonrpc_notification(
                    "session/cancel",
                    {
                        "sessionId": self._session_id,
                    },
                )
            )

    async def close(self) -> None:
        await self._websocket.close()
        with suppress(asyncio.CancelledError):
            await self._receive_task

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
            active_prompt = self._active_prompt
            if active_prompt is not None and not active_prompt.submission_future.done():
                active_prompt.submission_future.set_exception(RuntimeError(f"Remote {self.endpoint} disconnected."))
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError(f"Remote {self.endpoint} disconnected."))
            self._pending_requests.clear()
            await self._on_disconnect(self.endpoint)

    async def _handle_notification(self, payload: JsonObject) -> None:
        if payload.get("method") != "session/update":
            return

        active_prompt = self._active_prompt
        if active_prompt is not None and not active_prompt.submission_future.done():
            active_prompt.submission_future.set_result(None)

        params = payload.get("params", {})
        if not isinstance(params, dict):
            return
        update = params.get("update", {})
        if not isinstance(update, dict):
            return

        update_type = update.get("sessionUpdate")
        if update_type == "agent_message_chunk":
            content = update.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
                await self._on_event(RemoteContentDeltaEvent(content=content["text"]))
            return

        if update_type == "tool_call":
            title = update.get("title")
            if isinstance(title, str):
                await self._on_event(RemoteToolCallEvent(title=title))
            return

        if update_type == "tool_call_update":
            content_text: str | None = None
            content = update.get("content")
            if isinstance(content, list) and content:
                first_item = content[0]
                if (
                    isinstance(first_item, dict)
                    and isinstance(first_item.get("content"), dict)
                    and first_item["content"].get("type") == "text"
                    and isinstance(first_item["content"].get("text"), str)
                ):
                    content_text = first_item["content"]["text"]
            await self._on_event(
                RemoteToolCallUpdateEvent(
                    status=str(update.get("status", "")),
                    title=update.get("title") if isinstance(update.get("title"), str) else None,
                    content=content_text,
                )
            )

    async def _handle_response(self, payload: JsonObject) -> None:
        response_id = payload.get("id")
        error = payload.get("error")
        if response_id is None and isinstance(error, dict):
            message = str(error.get("message", "Unknown ACP error."))
            self._fatal_error = message
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError(message))
            self._pending_requests.clear()
            return
        if not isinstance(response_id, int):
            return

        response_future = self._pending_requests.pop(response_id) if response_id in self._pending_requests else None
        if response_future is not None:
            if isinstance(error, dict):
                message = error.get("message", "Unknown ACP error.")
                response_future.set_exception(RuntimeError(str(message)))
            else:
                response_future.set_result(payload.get("result", {}))

        active_prompt = self._active_prompt
        if active_prompt is None or active_prompt.request_id != response_id:
            return

        if isinstance(error, dict):
            message = str(error.get("message", "Unknown ACP error."))
            if not active_prompt.submission_future.done():
                active_prompt.submission_future.set_exception(RuntimeError(message))
            else:
                await self._on_event(RemotePromptFailedEvent(message=message))
            self._active_prompt = None
            return

        result = payload.get("result", {})
        stop_reason = ""
        if isinstance(result, dict):
            raw_stop_reason = result.get("stopReason")
            if isinstance(raw_stop_reason, str):
                stop_reason = raw_stop_reason
        if not active_prompt.submission_future.done():
            active_prompt.submission_future.set_result(None)
        await self._on_event(RemotePromptFinishedEvent(stop_reason=stop_reason))
        self._active_prompt = None


def _client_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"
