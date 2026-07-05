from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from http import HTTPStatus

from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.service import ManagerError, ManagerService, SessionBusyError
from coding_assistant.manager.store import SessionNotFoundError, StaleSessionCommitError
from coding_assistant.manager.workspace import WorkspaceMissingError
from coding_assistant.remote.acp import (
    ERROR_INVALID_PARAMS,
    ERROR_INVALID_REQUEST,
    ERROR_METHOD_NOT_FOUND,
    ERROR_SERVER,
    JsonObject,
    initialize_response,
    jsonrpc_error,
    jsonrpc_result,
    jsonrpc_result_required,
    params_from_payload,
    response_id_from_payload,
    session_id_from_params,
)
from coding_assistant.remote.protocol import session_update_notification
from coding_assistant.remote.websocket_server import receive_jsonrpc_messages, serve_jsonrpc_websocket


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManagerServer:
    endpoint: str


@dataclass
class _ConnectionState:
    initialized: bool = False
    subscribed_session_ids: set[str] = field(default_factory=set)


class _SessionUpdateBroker:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[ServerConnection]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, *, websocket: ServerConnection, session_id: str, state: _ConnectionState) -> None:
        async with self._lock:
            self._subscribers.setdefault(session_id, set()).add(websocket)
            state.subscribed_session_ids.add(session_id)

    async def unsubscribe_all(self, *, websocket: ServerConnection, state: _ConnectionState) -> None:
        async with self._lock:
            for session_id in state.subscribed_session_ids:
                subscribers = self._subscribers.get(session_id)
                if subscribers is None:
                    continue
                subscribers.discard(websocket)
                if not subscribers:
                    self._subscribers.pop(session_id, None)
            state.subscribed_session_ids.clear()

    async def publish(self, *, session_id: str, update: SessionUpdate) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(session_id, ()))
        for websocket in subscribers:
            try:
                await _send_session_update(websocket=websocket, session_id=session_id, update=update)
            except ConnectionClosed:
                async with self._lock:
                    self._subscribers.get(session_id, set()).discard(websocket)


def _error_code_for_exception(exc: Exception) -> int:
    if isinstance(exc, (SessionBusyError, StaleSessionCommitError)):
        return ERROR_SERVER
    if isinstance(exc, (ValueError, ManagerError, SessionNotFoundError, WorkspaceMissingError)):
        return ERROR_INVALID_PARAMS
    return ERROR_SERVER


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


async def _run_prompt_request(
    *,
    websocket: ServerConnection,
    broker: _SessionUpdateBroker,
    service: ManagerService,
    response_id: int | str,
    params: JsonObject,
) -> None:
    session_id = "<unknown>"
    try:
        session_id = session_id_from_params(params)

        async def send_update(update: SessionUpdate) -> None:
            await broker.publish(session_id=session_id, update=update)

        result = await service.prompt(params=params, on_update=send_update)
        with suppress(ConnectionClosed):
            await websocket.send(jsonrpc_result(response_id, {"stopReason": result.stop_reason}))
    except Exception as exc:
        logger.exception("Manager prompt request failed for session %s.", session_id)
        with suppress(ConnectionClosed):
            await websocket.send(jsonrpc_error(response_id, _error_code_for_exception(exc), str(exc)))


async def _handle_initialize(
    *,
    websocket: ServerConnection,
    state: _ConnectionState,
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
    state.initialized = True
    await websocket.send(
        jsonrpc_result(
            response_id,
            initialize_response(
                requested_protocol_version=protocol_version,
                agent_name="coding-assistant",
                agent_title="Coding Assistant",
                capabilities={
                    "loadSession": True,
                    "promptCapabilities": {
                        "image": True,
                        "embeddedContext": True,
                    },
                },
            ),
        ),
    )


async def _handle_session_method(
    *,
    websocket: ServerConnection,
    service: ManagerService,
    broker: _SessionUpdateBroker,
    prompt_tasks: set[asyncio.Task[None]],
    method: str,
    response_id: int | str | None,
    params: JsonObject,
    state: _ConnectionState,
) -> None:
    if method == "session/list":
        await websocket.send(jsonrpc_result_required(response_id, service.list_sessions(params=params)))
        return

    if method == "session/new":
        await websocket.send(
            jsonrpc_result_required(
                response_id,
                service.new_session(params=params),
            ),
        )
        return

    if method == "session/load":
        session_id = session_id_from_params(params)
        await broker.subscribe(websocket=websocket, session_id=session_id, state=state)

        async def send_update(update: SessionUpdate) -> None:
            await _send_session_update(websocket=websocket, session_id=session_id, update=update)

        result = await service.load_session(params=params, on_update=send_update)
        await websocket.send(jsonrpc_result_required(response_id, result))
        return

    if method == "session/upload_file":
        session_id = session_id_from_params(params)
        await broker.subscribe(websocket=websocket, session_id=session_id, state=state)

        async def send_update(update: SessionUpdate) -> None:
            await broker.publish(session_id=session_id, update=update)

        result = await service.upload_file(params=params, on_update=send_update)
        await websocket.send(jsonrpc_result_required(response_id, result))
        return

    if method == "session/download_attachment":
        await websocket.send(jsonrpc_result_required(response_id, await service.download_attachment(params=params)))
        return

    if method == "session/rename":
        session_id = session_id_from_params(params)
        await broker.subscribe(websocket=websocket, session_id=session_id, state=state)

        async def send_update(update: SessionUpdate) -> None:
            await broker.publish(session_id=session_id, update=update)

        result = await service.rename_session(params=params, on_update=send_update)
        await websocket.send(jsonrpc_result_required(response_id, result))
        return

    if method == "session/set_model":
        session_id = session_id_from_params(params)
        await broker.subscribe(websocket=websocket, session_id=session_id, state=state)

        async def send_update(update: SessionUpdate) -> None:
            await broker.publish(session_id=session_id, update=update)

        result = await service.set_session_model(params=params, on_update=send_update)
        await websocket.send(jsonrpc_result_required(response_id, result))
        return

    if method == "session/prompt":
        if response_id is None:
            await websocket.send(jsonrpc_error(None, ERROR_INVALID_REQUEST, "session/prompt must be a request."))
            return
        session_id = session_id_from_params(params)
        await broker.subscribe(websocket=websocket, session_id=session_id, state=state)
        task = asyncio.create_task(
            _run_prompt_request(
                websocket=websocket,
                broker=broker,
                service=service,
                response_id=response_id,
                params=params,
            ),
        )
        prompt_tasks.add(task)
        task.add_done_callback(prompt_tasks.discard)
        return

    if method == "session/cancel":
        await service.cancel(params=params)
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
        return

    if method == "session/delete":
        await service.delete_session(params=params)
        if response_id is not None:
            await websocket.send(jsonrpc_result(response_id, None))
        return

    await websocket.send(jsonrpc_error(response_id, ERROR_METHOD_NOT_FOUND, f"Unsupported manager method: {method}"))


async def _handle_jsonrpc_message(
    *,
    websocket: ServerConnection,
    service: ManagerService,
    broker: _SessionUpdateBroker,
    prompt_tasks: set[asyncio.Task[None]],
    state: _ConnectionState,
    payload: JsonObject,
) -> None:
    response_id = response_id_from_payload(payload)
    method = payload.get("method")
    if payload.get("jsonrpc") != "2.0" or not isinstance(method, str):
        await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "Invalid JSON-RPC request."))
        return

    try:
        params = params_from_payload(payload)
        if method == "initialize":
            await _handle_initialize(websocket=websocket, state=state, response_id=response_id, params=params)
            return

        if not state.initialized:
            await websocket.send(jsonrpc_error(response_id, ERROR_INVALID_REQUEST, "initialize must be called first."))
            return

        if method == "model/list":
            await websocket.send(jsonrpc_result_required(response_id, await service.list_models()))
            return

        await _handle_session_method(
            websocket=websocket,
            service=service,
            broker=broker,
            prompt_tasks=prompt_tasks,
            method=method,
            response_id=response_id,
            params=params,
            state=state,
        )
    except Exception as exc:
        logger.exception("Manager method %s failed.", method)
        await websocket.send(jsonrpc_error(response_id, _error_code_for_exception(exc), str(exc)))


def _is_authorized_request(request: Request, auth_secret: str) -> bool:
    expected = f"Bearer {auth_secret}"
    provided = request.headers.get("Authorization")
    return provided is not None and hmac.compare_digest(provided, expected)


@asynccontextmanager
async def start_manager_server(
    *,
    service: ManagerService,
    host: str = "127.0.0.1",
    port: int = 0,
    auth_secret: str | None = None,
) -> AsyncIterator[ManagerServer]:
    broker = _SessionUpdateBroker()
    prompt_tasks: set[asyncio.Task[None]] = set()

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        if auth_secret is None or _is_authorized_request(request, auth_secret):
            return None
        return connection.respond(HTTPStatus.UNAUTHORIZED, "Unauthorized\n")

    async def handle_connection(websocket: ServerConnection) -> None:
        state = _ConnectionState()

        async def handle_message(payload: JsonObject) -> None:
            await _handle_jsonrpc_message(
                websocket=websocket,
                service=service,
                broker=broker,
                prompt_tasks=prompt_tasks,
                state=state,
                payload=payload,
            )

        try:
            await receive_jsonrpc_messages(websocket=websocket, on_message=handle_message)
        finally:
            await broker.unsubscribe_all(websocket=websocket, state=state)

    async with serve_jsonrpc_websocket(
        handler=handle_connection,
        host=host,
        port=port,
        process_request=process_request,
    ) as server:
        try:
            yield ManagerServer(endpoint=server.endpoint)
        finally:
            for task in list(prompt_tasks):
                task.cancel()
            for task in list(prompt_tasks):
                with suppress(asyncio.CancelledError):
                    await task
