from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.app.session_host import (
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


async def _forward_session_events(
    websocket: ServerConnection,
    queue: asyncio.Queue[SessionEvent],
) -> None:
    while True:
        event = await queue.get()
        message: (
            StateMessage
            | ContentDeltaMessage
            | ToolCallsMessage
            | RunFinishedMessage
            | RunCancelledMessage
            | RunFailedMessage
        )
        if isinstance(event, StateChangedEvent):
            message = StateMessage(
                promptable=event.state.promptable,
                remote_connected=event.state.remote_connected,
                running=event.state.running,
            )
        elif isinstance(event, ContentDeltaEvent):
            message = ContentDeltaMessage(content=event.content)
        elif isinstance(event, ToolCallsEvent):
            message = ToolCallsMessage(
                tool_calls=[
                    ToolCallPayload(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    for tool_call in event.message.tool_calls
                ]
            )
        elif isinstance(event, RunFinishedEvent):
            message = RunFinishedMessage(summary=event.summary)
        elif isinstance(event, RunCancelledEvent):
            message = RunCancelledMessage()
        elif isinstance(event, RunFailedEvent):
            message = RunFailedMessage(error=event.error)
        else:
            continue
        await websocket.send(message_to_json(message))


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

        async with session.subscribe() as queue:
            sender_task = asyncio.create_task(_forward_session_events(websocket, queue))
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
                await session.restore_cli_controller()

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        async with advertise_worker(endpoint=endpoint, cwd=cwd):
            yield WorkerServer(endpoint=endpoint)
