from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from uuid import uuid4

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    AssistantMessageCompletedEvent,
    AssistantMessageDeltaEvent,
    RunOutcome,
    ScheduledRun,
)
from coding_assistant.core.session_updates import MessageAddedUpdate, MessageDeltaUpdate, SessionUpdate
from coding_assistant.core.tool_calls import ToolMessageProduced
from coding_assistant.remote.jsonrpc import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    JsonObject,
    initialize_response,
    jsonrpc_error,
    jsonrpc_result,
    params_from_payload,
    prompt_content_from_acp,
    response_id_from_payload,
    session_id_from_params,
)
from coding_assistant.remote.protocol import messages_to_jsonrpc, session_update_notification


@dataclass(frozen=True)
class RemoteAgentInfo:
    name: str
    title: str


@dataclass
class _RemoteControlState:
    initialized: bool = False
    session_opened: bool = False


def completed_run_result(
    *,
    outcome: RunOutcome,
    metadata: JsonObject | None = None,
) -> JsonObject:
    result: JsonObject = {
        "messages": messages_to_jsonrpc(outcome.messages),
        "stopReason": outcome.stop_reason,
    }
    if metadata:
        result["_meta"] = metadata
    return result


def worker_run_result(
    *,
    outcome: RunOutcome,
    metadata: JsonObject | None = None,
) -> JsonObject:
    result: JsonObject = {"stopReason": outcome.stop_reason}
    if metadata:
        result["_meta"] = metadata
    return result


def session_updates_from_agent_event(event: AgentSessionEvent) -> list[SessionUpdate]:
    """Project one core runtime event to transport-safe session updates."""
    if isinstance(event, AssistantMessageDeltaEvent):
        return [MessageDeltaUpdate(message_id=event.message_id, append_text=event.content)]
    if isinstance(event, AssistantMessageCompletedEvent):
        return [MessageAddedUpdate(message_id=event.message_id, message=event.completion.message)]
    if isinstance(event, ToolMessageProduced):
        return [MessageAddedUpdate(message_id=f"msg_{uuid4().hex}", message=event.message)]
    return []


async def send_session_update(
    *,
    websocket: ServerConnection,
    session_id: str,
    update: SessionUpdate,
) -> None:
    notification = session_update_notification(session_id=session_id, update=update)
    if notification is not None:
        await websocket.send(notification)


async def wait_for_session_events(
    *,
    queue: asyncio.Queue[AgentSessionEvent],
    publisher_task: asyncio.Task[None],
) -> None:
    """Wait until queued events are sent, or propagate publisher failure."""
    drain_task = asyncio.create_task(queue.join())
    try:
        done, _ = await asyncio.wait(
            {drain_task, publisher_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if publisher_task in done:
            await publisher_task
            raise RuntimeError("Session event publisher stopped before queued events were sent.")
        await drain_task
    finally:
        if not drain_task.done():
            drain_task.cancel()
        await asyncio.gather(drain_task, return_exceptions=True)


class RemoteAgentController:
    """Expose one existing live AgentSession over JSON-RPC."""

    def __init__(
        self,
        *,
        agent_info: RemoteAgentInfo,
        session_id: str,
        session: AgentSession,
        busy_message: str = "Session is busy.",
    ) -> None:
        self._agent_info = agent_info
        self._session_id = session_id
        self._session = session
        self._busy_message = busy_message
        self._state = _RemoteControlState()
        self._response_tasks: set[asyncio.Task[None]] = set()
        self._publisher_ready = asyncio.Event()
        self._publisher_queue: asyncio.Queue[AgentSessionEvent] | None = None
        self._publisher_task: asyncio.Task[None] | None = None

    async def close(self) -> None:
        for task in list(self._response_tasks):
            task.cancel()
        for task in list(self._response_tasks):
            with suppress(asyncio.CancelledError):
                await task

    async def publish_session_events(self, *, websocket: ServerConnection) -> None:
        async with self._session.subscribe() as queue:
            publisher_task = asyncio.current_task()
            assert publisher_task is not None
            self._publisher_queue = queue
            self._publisher_task = publisher_task
            self._publisher_ready.set()
            while True:
                event = await queue.get()
                try:
                    for update in session_updates_from_agent_event(event):
                        await send_session_update(websocket=websocket, session_id=self._session_id, update=update)
                finally:
                    queue.task_done()

    async def handle_jsonrpc_message(self, *, websocket: ServerConnection, payload: JsonObject) -> None:
        response_id = response_id_from_payload(payload)
        method = payload.get("method")
        if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
            return

        try:
            params = params_from_payload(payload)
            if method == "initialize":
                await self._handle_initialize(websocket=websocket, response_id=response_id, params=params)
                return
            if not self._state.initialized:
                if response_id is not None:
                    await websocket.send(
                        jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "initialize must be called first."),
                    )
                return
            if method == "session/new":
                await self._handle_session_new(websocket=websocket, response_id=response_id)
                return
            if method == "session/prompt":
                await self._handle_prompt(websocket=websocket, response_id=response_id, params=params)
                return
            if method == "session/cancel":
                await self._handle_cancel(websocket=websocket, response_id=response_id, params=params)
                return
            if response_id is not None:
                await websocket.send(
                    jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported method: {method}")
                )
        except ValueError as exc:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))

    async def _handle_initialize(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        if response_id is None:
            return
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, int):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "initialize requires an integer protocolVersion."),
            )
            return
        try:
            result = initialize_response(
                requested_protocol_version=protocol_version,
                agent_name=self._agent_info.name,
                agent_title=self._agent_info.title,
            )
        except ValueError as exc:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
            return
        self._state.initialized = True
        await websocket.send(
            jsonrpc_result(
                response_id,
                result,
            ),
        )

    async def _handle_session_new(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
    ) -> None:
        if response_id is None:
            return
        self._state.session_opened = True
        await websocket.send(jsonrpc_result(response_id, {"sessionId": self._session_id}))

    async def _handle_prompt(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        if response_id is None:
            return
        if not self._state.session_opened:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Session must be opened first."))
            return
        if session_id_from_params(params) != self._session_id:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list) or not all(isinstance(block, dict) for block in prompt_blocks):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "session/prompt requires a prompt array."),
            )
            return
        try:
            prompt_content = prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
            return

        scheduled_run = await self._session.enqueue_prompt_if_idle(prompt_content)
        if scheduled_run is None:
            await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, self._busy_message))
            return

        task = asyncio.create_task(
            self._respond_when_run_finishes(
                websocket=websocket,
                response_id=response_id,
                scheduled_run=scheduled_run,
            ),
        )
        self._response_tasks.add(task)
        task.add_done_callback(self._response_tasks.discard)

    async def _respond_when_run_finishes(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str,
        scheduled_run: ScheduledRun,
    ) -> None:
        outcome = await scheduled_run.completion
        try:
            await self._publisher_ready.wait()
            assert self._publisher_queue is not None
            assert self._publisher_task is not None
            await wait_for_session_events(
                queue=self._publisher_queue,
                publisher_task=self._publisher_task,
            )
            if outcome.stop_reason == "failed":
                await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, outcome.error or "Run failed."))
                return
            await websocket.send(jsonrpc_result(response_id, completed_run_result(outcome=outcome)))
        except ConnectionClosed:
            return

    async def _handle_cancel(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        if not self._state.session_opened:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Session must be opened first."))
            return
        if session_id_from_params(params) != self._session_id:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        await self._session.cancel_current_run()
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
