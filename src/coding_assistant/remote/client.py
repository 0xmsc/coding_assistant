from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    SessionUpdate,
    ToolCallLifecycleUpdate,
    ToolCallStartedUpdate,
)
from coding_assistant.llm.types import BaseMessage
from coding_assistant.remote.acp import (
    ACP_PROTOCOL_VERSION,
    JsonObject,
    jsonrpc_notification,
    jsonrpc_request,
    parse_jsonrpc_message,
    text_block,
)
from coding_assistant.remote.protocol import messages_from_jsonrpc, session_update_from_jsonrpc_update


def _client_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"


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


@dataclass(frozen=True)
class RemoteCommit:
    session_id: str
    base_version: int
    messages: list[BaseMessage]
    stop_reason: str


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


class RemoteSessionClient:
    """One live JSON-RPC connection to a remote agent session endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        websocket: ClientConnection,
        on_event: Callable[[RemoteClientEvent], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
        on_session_update: Callable[[SessionUpdate], Awaitable[None]] | None = None,
        on_commit: Callable[[RemoteCommit], Awaitable[None]] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._websocket = websocket
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self._on_session_update = on_session_update
        self._on_commit = on_commit
        self._send_lock = asyncio.Lock()
        self._next_request_id = 1
        self._pending_requests: dict[int, asyncio.Future[JsonObject]] = {}
        self._session_id: str | None = None
        self._active_prompt: _ActivePrompt | None = None
        self._fatal_error: str | None = None
        self._receive_task = asyncio.create_task(self._receive_loop())

    @classmethod
    async def connect(
        cls,
        *,
        endpoint: str,
        on_event: Callable[[RemoteClientEvent], Awaitable[None]],
        on_disconnect: Callable[[str], Awaitable[None]],
        on_session_update: Callable[[SessionUpdate], Awaitable[None]] | None = None,
        on_commit: Callable[[RemoteCommit], Awaitable[None]] | None = None,
    ) -> RemoteSessionClient:
        websocket = await connect(endpoint)
        return cls(
            endpoint=endpoint,
            websocket=websocket,
            on_event=on_event,
            on_disconnect=on_disconnect,
            on_session_update=on_session_update,
            on_commit=on_commit,
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

    async def new_session(self, params: JsonObject | None = None) -> str:
        result = await self._request(
            "session/new",
            params or {},
        )
        session_id = result.get("sessionId")
        if not isinstance(session_id, str):
            raise RuntimeError("session/new response did not include a sessionId.")
        self._session_id = session_id
        return session_id

    async def start_session(self, params: JsonObject) -> str:
        result = await self._request("_session/start", params)
        session_id = result.get("sessionId")
        if not isinstance(session_id, str):
            raise RuntimeError("_session/start response did not include a sessionId.")
        self._session_id = session_id
        return session_id

    async def prompt(self, prompt: str, *, session_id: str | None = None) -> str | None:
        return await self.prompt_blocks([text_block(prompt)], session_id=session_id)

    async def prompt_blocks(self, prompt: list[JsonObject], *, session_id: str | None = None) -> str | None:
        target_session_id = session_id or self._session_id
        if target_session_id is None:
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
                        "sessionId": target_session_id,
                        "prompt": prompt,
                    },
                ),
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

    async def cancel(self, *, session_id: str | None = None) -> None:
        target_session_id = session_id or self._session_id
        if target_session_id is None:
            raise RuntimeError("Remote session is not initialized.")
        async with self._send_lock:
            await self._websocket.send(
                jsonrpc_notification(
                    "session/cancel",
                    {
                        "sessionId": target_session_id,
                    },
                ),
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
        method = payload.get("method")
        if method == "_session/commit":
            await self._handle_commit_notification(payload)
            return
        if method != "session/update":
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

        session_update = session_update_from_jsonrpc_update(update)
        if session_update is None:
            return
        if self._on_session_update is not None:
            await self._on_session_update(session_update)
        await self._publish_session_update(session_update)

    async def _handle_commit_notification(self, payload: JsonObject) -> None:
        if self._on_commit is None:
            return
        params = payload.get("params", {})
        if not isinstance(params, dict):
            return
        session_id = params.get("sessionId")
        base_version = params.get("baseVersion")
        messages = params.get("messages")
        stop_reason = params.get("stopReason")
        if (
            not isinstance(session_id, str)
            or not isinstance(base_version, int)
            or not isinstance(messages, list)
            or not all(isinstance(message, dict) for message in messages)
            or not isinstance(stop_reason, str)
        ):
            return
        await self._on_commit(
            RemoteCommit(
                session_id=session_id,
                base_version=base_version,
                messages=messages_from_jsonrpc(messages),
                stop_reason=stop_reason,
            ),
        )

    async def _publish_session_update(self, update: SessionUpdate) -> None:
        if isinstance(update, AgentMessageChunkUpdate):
            await self._on_event(RemoteContentDeltaEvent(content=update.content))
            return

        if isinstance(update, ToolCallStartedUpdate):
            await self._on_event(RemoteToolCallEvent(title=update.title))
            return

        if isinstance(update, ToolCallLifecycleUpdate):
            await self._on_event(
                RemoteToolCallUpdateEvent(
                    status=update.status,
                    title=update.title,
                    content=update.content,
                ),
            )

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

        response_future = self._pending_requests.pop(response_id) if response_id in self._pending_requests else None
        if response_future is not None:
            if isinstance(error, dict):
                message = error.get("message", "Unknown remote protocol error.")
                response_future.set_exception(RuntimeError(str(message)))
            else:
                response_future.set_result(payload.get("result", {}))

        active_prompt = self._active_prompt
        if active_prompt is None or active_prompt.request_id != response_id:
            return

        if isinstance(error, dict):
            message = str(error.get("message", "Unknown remote protocol error."))
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
