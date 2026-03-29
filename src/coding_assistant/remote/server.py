from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.app.session_control import (
    REMOTE_CONTROLLER,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionControlSurface,
    SessionEvent,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import ContentDeltaEvent
from coding_assistant.remote.discovery import advertise_worker
from coding_assistant.remote.protocol import (
    CancelCommand,
    CommandAcceptedMessage,
    ContentDeltaMessage,
    ErrorMessage,
    NotReadyMessage,
    PromptCommand,
    RunCancelledMessage,
    RunFailedMessage,
    RunFinishedMessage,
    StateMessage,
    ToolCallPayload,
    ToolCallsMessage,
    message_to_json,
    parse_supervisor_message,
)


@dataclass(frozen=True)
class WorkerServer:
    endpoint: str


class RemoteController:
    """Expose one worker websocket endpoint that can temporarily own session input."""

    def __init__(
        self,
        *,
        cwd: Path,
        set_local_worker_endpoint: Callable[[str], None],
    ) -> None:
        self._cwd = cwd
        self._set_local_worker_endpoint = set_local_worker_endpoint

    async def run(self, session: SessionControlSurface) -> None:
        async with start_worker_server(session=session, cwd=self._cwd) as worker_server:
            self._set_local_worker_endpoint(worker_server.endpoint)
            await asyncio.Future()


class RemoteOutput:
    """Forward session events over one supervisor websocket connection."""

    def __init__(self, *, websocket: ServerConnection) -> None:
        self._websocket = websocket

    async def run(self, session: SessionControlSurface) -> None:
        async with session.subscribe() as queue:
            while True:
                event = await queue.get()
                message = _session_event_to_message(event)
                if message is None:
                    continue
                await self._websocket.send(message_to_json(message))


def _session_event_to_message(
    event: SessionEvent,
) -> (
    StateMessage
    | ContentDeltaMessage
    | ToolCallsMessage
    | RunFinishedMessage
    | RunCancelledMessage
    | RunFailedMessage
    | None
):
    if isinstance(event, StateChangedEvent):
        return StateMessage(
            promptable=event.state.promptable,
            remote_connected=event.state.remote_connected,
            running=event.state.running,
        )
    if isinstance(event, ContentDeltaEvent):
        return ContentDeltaMessage(content=event.content)
    if isinstance(event, ToolCallsEvent):
        return ToolCallsMessage(
            tool_calls=[
                ToolCallPayload(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
                for tool_call in event.message.tool_calls
            ]
        )
    if isinstance(event, RunFinishedEvent):
        return RunFinishedMessage(summary=event.summary)
    if isinstance(event, RunCancelledEvent):
        return RunCancelledMessage()
    if isinstance(event, RunFailedEvent):
        return RunFailedMessage(error=event.error)
    return None


async def _handle_supervisor_message(
    websocket: ServerConnection,
    session: SessionControlSurface,
    message: PromptCommand | CancelCommand,
) -> None:
    if isinstance(message, PromptCommand):
        result = await session.submit_prompt(controller=REMOTE_CONTROLLER, content=message.prompt)
        if result.accepted:
            await websocket.send(message_to_json(CommandAcceptedMessage(request_id=message.request_id)))
        else:
            state = session.state
            await websocket.send(
                message_to_json(
                    NotReadyMessage(
                        request_id=message.request_id,
                        promptable=state.promptable,
                        remote_connected=state.remote_connected,
                        running=state.running,
                    )
                )
            )
        return

    cancelled = await session.cancel_current_run()
    if cancelled:
        await websocket.send(message_to_json(CommandAcceptedMessage(request_id=message.request_id)))
        return

    state = session.state
    await websocket.send(
        message_to_json(
            NotReadyMessage(
                request_id=message.request_id,
                promptable=state.promptable,
                remote_connected=state.remote_connected,
                running=state.running,
            )
        )
    )


@asynccontextmanager
async def start_worker_server(
    *,
    session: SessionControlSurface,
    cwd: Path,
) -> AsyncIterator[WorkerServer]:
    async def handle_connection(websocket: ServerConnection) -> None:
        if not await session.activate_controller(REMOTE_CONTROLLER):
            await websocket.send(message_to_json(ErrorMessage(message="Worker is already remotely controlled.")))
            await websocket.close()
            return

        sender_task = asyncio.create_task(RemoteOutput(websocket=websocket).run(session))
        try:
            async for raw_message in websocket:
                message = parse_supervisor_message(raw_message)
                await _handle_supervisor_message(websocket, session, message)
        except ConnectionClosed:
            pass
        finally:
            sender_task.cancel()
            with suppress(asyncio.CancelledError):
                await sender_task
            await session.activate_default_controller(cancel_current_run=True)

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        async with advertise_worker(endpoint=endpoint, cwd=cwd):
            yield WorkerServer(endpoint=endpoint)
