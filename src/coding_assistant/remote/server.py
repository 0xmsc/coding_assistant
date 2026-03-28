from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.app.session_host import (
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionEvent,
    SessionHost,
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


@asynccontextmanager
async def start_worker_server(
    *,
    session_host: SessionHost,
    cwd: Path,
) -> AsyncIterator[WorkerServer]:
    async def handle_connection(websocket: ServerConnection) -> None:
        if not await session_host.attach_remote_controller():
            await websocket.send(message_to_json(ErrorMessage(message="Worker is already remotely controlled.")))
            await websocket.close()
            return

        async with session_host.subscribe() as queue:
            sender_task = asyncio.create_task(_forward_session_events(websocket, queue))
            try:
                async for raw_message in websocket:
                    message = parse_supervisor_message(raw_message)
                    await _handle_supervisor_message(websocket, session_host, message)
            except ConnectionClosed:
                pass
            finally:
                sender_task.cancel()
                with suppress(asyncio.CancelledError):
                    await sender_task
                await session_host.detach_remote_controller()

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        async with advertise_worker(endpoint=endpoint, cwd=cwd):
            yield WorkerServer(endpoint=endpoint)


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
    session_host: SessionHost,
    message: PromptCommand | CancelCommand,
) -> None:
    if isinstance(message, PromptCommand):
        result = await session_host.submit_remote_prompt(message.prompt)
        if result.accepted:
            await websocket.send(message_to_json(CommandAcceptedMessage(request_id=message.request_id)))
        else:
            state = session_host.state
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

    cancelled = await session_host.cancel_current_run()
    if cancelled:
        await websocket.send(message_to_json(CommandAcceptedMessage(request_id=message.request_id)))
        return

    state = session_host.state
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
