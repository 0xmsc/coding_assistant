from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.llm.types import ContentDeltaEvent
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
from coding_assistant.remote.worker_session import (
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
    WorkerSession,
    WorkerSessionEvent,
)


@dataclass(frozen=True)
class WorkerServer:
    """Handle returned by the server context, exposing the chosen endpoint."""

    endpoint: str


class RemoteOutput:
    """Forward session events over one supervisor websocket connection."""

    def __init__(self, *, websocket: ServerConnection) -> None:
        self._websocket = websocket

    async def run(self, session: WorkerSession) -> None:
        """Subscribe to worker-session events and forward the wire-compatible ones."""
        async with session.subscribe() as queue:
            while True:
                event = await queue.get()
                message = _session_event_to_message(event)
                if message is None:
                    continue
                await self._websocket.send(message_to_json(message))


def _session_event_to_message(
    event: WorkerSessionEvent,
) -> (
    StateMessage
    | ContentDeltaMessage
    | ToolCallsMessage
    | RunFinishedMessage
    | RunCancelledMessage
    | RunFailedMessage
    | None
):
    """Translate internal session events into protocol messages for the supervisor."""
    if isinstance(event, StateChangedEvent):
        return StateMessage(
            promptable=event.state.promptable,
            remote_connected=True,
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
    session: WorkerSession,
    message: PromptCommand | CancelCommand,
) -> None:
    """Execute one supervisor command and send its immediate accepted/not-ready reply."""
    if isinstance(message, PromptCommand):
        if await session.submit_prompt(message.prompt):
            await websocket.send(message_to_json(CommandAcceptedMessage(request_id=message.request_id)))
        else:
            state = session.state
            await websocket.send(
                message_to_json(
                    NotReadyMessage(
                        request_id=message.request_id,
                        promptable=state.promptable,
                        remote_connected=True,
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
                remote_connected=True,
                running=state.running,
            )
        )
    )


@asynccontextmanager
async def start_worker_server(
    *,
    session: WorkerSession,
) -> AsyncIterator[WorkerServer]:
    """Serve one controlling supervisor connection at a time for a worker session."""
    connection_lock = asyncio.Lock()
    active_connection: ServerConnection | None = None

    async def handle_connection(websocket: ServerConnection) -> None:
        """Own one websocket connection for as long as it is the active controller."""
        nonlocal active_connection

        async with connection_lock:
            if active_connection is not None:
                await websocket.send(message_to_json(ErrorMessage(message="Worker is already remotely controlled.")))
                await websocket.close()
                return
            active_connection = websocket

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
            await session.cancel_current_run()
            async with connection_lock:
                if active_connection is websocket:
                    active_connection = None

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        yield WorkerServer(endpoint=endpoint)
