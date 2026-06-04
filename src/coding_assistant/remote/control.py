from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from uuid import uuid4

from websockets.asyncio.server import ServerConnection

from coding_assistant.core.agent_session import AgentSession
from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    SessionUpdate,
    prompt_result_from_update,
    session_updates_from_agent_event,
)
from coding_assistant.llm.types import BaseMessage
from coding_assistant.remote.acp import (
    ACP_PROTOCOL_VERSION,
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    JsonObject,
    initialize_result,
    jsonrpc_error,
    jsonrpc_notification,
    jsonrpc_result,
    prompt_content_from_acp,
)
from coding_assistant.remote.protocol import messages_to_jsonrpc, session_update_to_jsonrpc_update


UnhandledMethod = Callable[[str, int | str | None, JsonObject], Awaitable[bool]]


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
    active_prompt_source: str | None = None


def _agent_version() -> str:
    try:
        return version("coding-assistant-cli")
    except PackageNotFoundError:
        return "0.0.0"


def _response_id(payload: JsonObject) -> int | str | None:
    request_id = payload.get("id")
    return request_id if isinstance(request_id, int | str) else None


def _params_from_payload(payload: JsonObject) -> JsonObject:
    params = payload.get("params", {})
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("Request params must be an object.")
    return params


def _session_id_from_params(params: JsonObject) -> str:
    session_id = params.get("sessionId")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Request params must include sessionId.")
    return session_id


def _update_matches_active_prompt(update: SessionUpdate, source: str | None) -> bool:
    if source is None:
        return False
    return getattr(update, "source", None) == source


def _session_update_notification(session_id: str, update: JsonObject) -> str:
    return jsonrpc_notification(
        "session/update",
        {
            "sessionId": session_id,
            "update": update,
        },
    )


def _commit_notification(
    *,
    session_id: str,
    base_version: int,
    messages: list[BaseMessage],
    stop_reason: str,
) -> str:
    return jsonrpc_notification(
        "_session/commit",
        {
            "sessionId": session_id,
            "baseVersion": base_version,
            "messages": messages_to_jsonrpc(messages),
            "stopReason": stop_reason,
        },
    )


async def _send_session_update(
    *,
    websocket: ServerConnection,
    session_id: str,
    update: SessionUpdate,
) -> None:
    payload_update = session_update_to_jsonrpc_update(update)
    if payload_update is None:
        return
    await websocket.send(_session_update_notification(session_id, payload_update))


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
    ) -> None:
        self._agent_info = agent_info
        self._controlled_session = controlled_session
        self._supports_session_new = supports_session_new
        self._busy_message = busy_message
        self._unopened_message = unopened_message
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
                for update in session_updates_from_agent_event(event):
                    current_session = self._controlled_session
                    if current_session is None:
                        return
                    await self._handle_session_update(
                        websocket=websocket,
                        controlled_session=current_session,
                        update=update,
                    )

    async def handle_jsonrpc_message(
        self,
        *,
        websocket: ServerConnection,
        payload: JsonObject,
        on_unhandled: UnhandledMethod | None = None,
    ) -> None:
        response_id = _response_id(payload)
        method = payload.get("method")
        if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
            return

        try:
            params = _params_from_payload(payload)
        except ValueError as exc:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, str(exc)))
            return

        if method == "initialize":
            await self._handle_initialize(websocket=websocket, response_id=response_id, params=params)
            return
        if not self._state.initialized:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "initialize must be called first."))
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

    async def _handle_session_update(
        self,
        *,
        websocket: ServerConnection,
        controlled_session: RemoteControlledSession,
        update: SessionUpdate,
    ) -> None:
        if isinstance(update, AgentMessageChunkUpdate) and self._state.active_prompt_source is not None:
            await _send_session_update(websocket=websocket, session_id=controlled_session.session_id, update=update)
            return

        if not _update_matches_active_prompt(update, self._state.active_prompt_source):
            return

        await _send_session_update(websocket=websocket, session_id=controlled_session.session_id, update=update)

        result = prompt_result_from_update(update)
        if result is None:
            return

        request_id = self._state.active_prompt_request_id
        if request_id is None:
            return

        if result.stop_reason is not None:
            await websocket.send(jsonrpc_result(request_id, {"stopReason": result.stop_reason}))
            await websocket.send(
                _commit_notification(
                    session_id=controlled_session.session_id,
                    base_version=controlled_session.base_version,
                    messages=controlled_session.session.history[controlled_session.base_message_count :],
                    stop_reason=result.stop_reason,
                ),
            )
            self._controlled_session = RemoteControlledSession(
                session_id=controlled_session.session_id,
                session=controlled_session.session,
                base_version=controlled_session.base_version + 1,
                base_message_count=len(controlled_session.session.history),
            )
        else:
            await websocket.send(jsonrpc_error(request_id, ERROR_SERVER, result.error or "Run failed."))

        self._state.active_prompt_request_id = None
        self._state.active_prompt_source = None

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
                initialize_result(
                    agent_name=self._agent_info.name,
                    agent_title=self._agent_info.title,
                    agent_version=_agent_version(),
                )
                | {"protocolVersion": min(protocol_version, ACP_PROTOCOL_VERSION)},
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
        if _session_id_from_params(params) != controlled_session.session_id:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_PARAMS, "Unknown sessionId."))
            return
        prompt_blocks = params.get("prompt")
        if not isinstance(prompt_blocks, list):
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

        prompt_source = f"remote:{controlled_session.session_id}:{response_id}:{uuid4().hex}"
        self._state.active_prompt_request_id = response_id
        self._state.active_prompt_source = prompt_source
        accepted = await controlled_session.session.enqueue_prompt_if_idle(prompt_content, source=prompt_source)
        if not accepted:
            self._state.active_prompt_request_id = None
            self._state.active_prompt_source = None
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
            return
        if _session_id_from_params(params) != controlled_session.session_id:
            return
        await controlled_session.session.cancel_current_run()
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
