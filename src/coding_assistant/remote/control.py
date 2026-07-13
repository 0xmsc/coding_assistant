from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from websockets.asyncio.server import ServerConnection

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
)
from coding_assistant.core.session_updates import (
    MessageAddedUpdate,
    MessageDeltaUpdate,
    SessionUpdate,
)
from coding_assistant.core.tool_calls import ToolMessageProduced
from coding_assistant.llm.types import BaseMessage, CompletionEvent, ContentDeltaEvent, ModelRetryEvent
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


UnhandledMethod = Callable[[str, int | str | None, JsonObject], Awaitable[bool]]
FinishMetadataProvider = Callable[[], JsonObject | None]
PromptFinishedCallback = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class RemoteAgentInfo:
    name: str
    title: str


@dataclass(frozen=True)
class RemoteControlledSession:
    session_id: str
    session: AgentSession
    base_version: int = 0
    base_message_count: int = 0


@dataclass
class _RemoteControlState:
    initialized: bool = False
    session_opened: bool = False
    active_prompt_request_id: int | str | None = None
    assistant_message_id: str | None = None


def _prompt_result(
    *,
    base_version: int,
    messages: list[BaseMessage],
    stop_reason: str,
    metadata: JsonObject | None = None,
) -> JsonObject:
    result: JsonObject = {
        "baseVersion": base_version,
        "messages": messages_to_jsonrpc(messages),
        "stopReason": stop_reason,
    }
    if metadata:
        result["_meta"] = metadata
    return result


async def _send_session_update(
    *,
    websocket: ServerConnection,
    session_id: str,
    update: SessionUpdate,
) -> None:
    notification = session_update_notification(session_id=session_id, update=update)
    if notification is None:
        return
    await websocket.send(notification)


class RemoteAgentController:
    """Drive one AgentSession over the shared remote-control JSON-RPC methods."""

    def __init__(
        self,
        *,
        agent_info: RemoteAgentInfo,
        controlled_session: RemoteControlledSession | None = None,
        supports_session_new: bool = False,
        busy_message: str = "Session is busy.",
        unopened_message: str = "Session must be opened first.",
        finish_metadata_provider: FinishMetadataProvider | None = None,
        on_prompt_finished: PromptFinishedCallback | None = None,
    ) -> None:
        self._agent_info = agent_info
        self._controlled_session = controlled_session
        self._supports_session_new = supports_session_new
        self._busy_message = busy_message
        self._unopened_message = unopened_message
        self._finish_metadata_provider = finish_metadata_provider
        self._on_prompt_finished = on_prompt_finished
        self._state = _RemoteControlState()

    @property
    def has_session(self) -> bool:
        return self._controlled_session is not None

    @property
    def session(self) -> AgentSession | None:
        if self._controlled_session is None:
            return None
        return self._controlled_session.session

    async def start_session(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        controlled_session: RemoteControlledSession,
    ) -> None:
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "_session/start must be a request."))
            return
        if self._controlled_session is not None:
            await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, "Worker session is already started."))
            return
        self._controlled_session = controlled_session
        self._state.session_opened = True
        await websocket.send(jsonrpc_result(response_id, {"sessionId": controlled_session.session_id}))

    async def publish_session_events(self, *, websocket: ServerConnection) -> None:
        controlled_session = self._controlled_session
        if controlled_session is None:
            return

        async with controlled_session.session.subscribe() as queue:
            while True:
                event = await queue.get()
                current_session = self._controlled_session
                if current_session is None:
                    return
                for update in self._wire_updates_from_agent_event(event):
                    await _send_session_update(
                        websocket=websocket,
                        session_id=current_session.session_id,
                        update=update,
                    )
                if isinstance(event, (RunFinishedEvent, RunCancelledEvent, RunFailedEvent)):
                    await self._finish_prompt(
                        websocket=websocket,
                        controlled_session=current_session,
                        event=event,
                    )

    async def handle_jsonrpc_message(
        self,
        *,
        websocket: ServerConnection,
        payload: JsonObject,
        on_unhandled: UnhandledMethod | None = None,
    ) -> None:
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
            if on_unhandled is not None and await on_unhandled(method, response_id, params):
                return
            await websocket.send(jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported method: {method}"))
        except ValueError as exc:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))

    async def _finish_prompt(
        self,
        *,
        websocket: ServerConnection,
        controlled_session: RemoteControlledSession,
        event: RunFinishedEvent | RunCancelledEvent | RunFailedEvent,
    ) -> None:
        request_id = self._state.active_prompt_request_id
        if request_id is None:
            return
        self._state.active_prompt_request_id = None
        self._state.assistant_message_id = None

        if not isinstance(event, RunFailedEvent):
            stop_reason = "cancelled" if isinstance(event, RunCancelledEvent) else "end_turn"
            response = _prompt_result(
                base_version=controlled_session.base_version,
                messages=controlled_session.session.history[controlled_session.base_message_count :],
                stop_reason=stop_reason,
                metadata=self._finish_metadata_provider() if self._finish_metadata_provider else None,
            )
            self._controlled_session = RemoteControlledSession(
                session_id=controlled_session.session_id,
                session=controlled_session.session,
                base_version=controlled_session.base_version + 1,
                base_message_count=len(controlled_session.session.history),
            )
            await websocket.send(
                jsonrpc_result(request_id, response),
            )
            if self._on_prompt_finished is not None:
                await self._on_prompt_finished()
        else:
            await websocket.send(jsonrpc_error(request_id, ERROR_SERVER, event.error))

    def _wire_updates_from_agent_event(self, event: AgentSessionEvent) -> list[SessionUpdate]:
        if isinstance(event, ContentDeltaEvent):
            return self._append_assistant_text(event.content)
        if isinstance(event, ModelRetryEvent):
            self._state.assistant_message_id = None
            return []
        if isinstance(event, CompletionEvent):
            message_id = self._state.assistant_message_id or f"msg_{uuid4().hex}"
            self._state.assistant_message_id = None
            return [MessageAddedUpdate(message_id=message_id, message=event.completion.message)]
        if isinstance(event, ToolMessageProduced):
            return [MessageAddedUpdate(message_id=f"msg_{uuid4().hex}", message=event.message)]
        return []

    def _append_assistant_text(self, content: str) -> list[SessionUpdate]:
        if self._state.assistant_message_id is None:
            self._state.assistant_message_id = f"msg_{uuid4().hex}"
        return [MessageDeltaUpdate(message_id=self._state.assistant_message_id, append_text=content)]

    async def _handle_initialize(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "initialize must be a request."))
            return
        protocol_version = params.get("protocolVersion")
        if not isinstance(protocol_version, int):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "initialize requires an integer protocolVersion."),
            )
            return
        self._state.initialized = True
        await websocket.send(
            jsonrpc_result(
                response_id,
                initialize_response(
                    requested_protocol_version=protocol_version,
                    agent_name=self._agent_info.name,
                    agent_title=self._agent_info.title,
                ),
            ),
        )

    async def _handle_session_new(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
    ) -> None:
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/new must be a request."))
            return
        if not self._supports_session_new or self._controlled_session is None:
            await websocket.send(jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, "Unsupported method: session/new"))
            return
        self._state.session_opened = True
        await websocket.send(jsonrpc_result(response_id, {"sessionId": self._controlled_session.session_id}))

    async def _handle_prompt(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/prompt must be a request."))
            return
        controlled_session = self._controlled_session
        if controlled_session is None or not self._state.session_opened:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, self._unopened_message))
            return
        if session_id_from_params(params) != controlled_session.session_id:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list) or not all(isinstance(block, dict) for block in prompt_blocks):
            await websocket.send(
                jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "session/prompt requires a prompt array."),
            )
            return
        if self._state.active_prompt_request_id is not None:
            await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, self._busy_message))
            return

        try:
            prompt_content = prompt_content_from_acp(prompt_blocks)
        except ValueError as exc:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
            return

        self._state.active_prompt_request_id = response_id
        self._state.assistant_message_id = None
        accepted = await controlled_session.session.enqueue_prompt_if_idle(prompt_content)
        if not accepted:
            self._state.active_prompt_request_id = None
            await websocket.send(jsonrpc_error(response_id, ERROR_SERVER, self._busy_message))

    async def _handle_cancel(
        self,
        *,
        websocket: ServerConnection,
        response_id: int | str | None,
        params: JsonObject,
    ) -> None:
        controlled_session = self._controlled_session
        if controlled_session is None or not self._state.session_opened:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, self._unopened_message))
            return
        if session_id_from_params(params) != controlled_session.session_id:
            if response_id is not None:
                await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        await controlled_session.session.cancel_current_run()
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
