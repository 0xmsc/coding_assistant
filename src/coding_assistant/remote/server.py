from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TypedDict

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import ContentDeltaEvent
from coding_assistant.remote.protocol import (
    CancelCommand,
    ContentDeltaMessage,
    ErrorMessage,
    PromptCommand,
    RequestOkMessage,
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
    """Handle returned by the server context, exposing the chosen endpoint."""

    endpoint: str


class RemoteOutput:
    """Forward session events over one remote websocket connection."""

    def __init__(self, *, websocket: ServerConnection) -> None:
        self._websocket = websocket

    async def run(self, session: AgentSession) -> None:
        """Subscribe to session events and forward the wire-compatible ones."""
        async with session.subscribe() as queue:
            while True:
                event = await queue.get()
                message = _session_event_to_message(event, state=session.state)
                if message is None:
                    continue
                await self._websocket.send(message_to_json(message))


class _StatePayload(TypedDict):
    accepting_prompts: bool
    running: bool


def _session_event_to_message(
    event: AgentSessionEvent,
    *,
    state: SessionState,
) -> (
    StateMessage
    | ContentDeltaMessage
    | ToolCallsMessage
    | RunFinishedMessage
    | RunCancelledMessage
    | RunFailedMessage
    | None
):
    """Translate internal session events into protocol messages for the remote client."""
    if isinstance(event, StateChangedEvent):
        return StateMessage(**_state_payload(event.state))
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
        return RunFinishedMessage(summary=event.summary, **_state_payload(state))
    if isinstance(event, RunCancelledEvent):
        return RunCancelledMessage(**_state_payload(state))
    if isinstance(event, RunFailedEvent):
        return RunFailedMessage(error=event.error, **_state_payload(state))
    return None


def _state_payload(state: SessionState) -> _StatePayload:
    """Serialize one session state snapshot into protocol fields."""
    return {
        "accepting_prompts": state.promptable and not state.running and state.queued_prompt_count == 0,
        "running": state.running,
    }


async def _handle_supervisor_message(
    websocket: ServerConnection,
    session: AgentSession,
    message: PromptCommand | CancelCommand,
) -> None:
    """Execute one remote command and send its immediate success or error reply."""
    if isinstance(message, PromptCommand):
        accepted = await session.enqueue_prompt_if_idle(message.prompt)
        if accepted:
            await websocket.send(message_to_json(RequestOkMessage(request_id=message.request_id)))
            return

        state = session.state
        if not state.promptable:
            message_text = "Remote session is not accepting prompts."
        else:
            message_text = "Remote session is busy. Wait for it to finish or cancel the current run."
        await websocket.send(
            message_to_json(
                ErrorMessage(
                    request_id=message.request_id,
                    message=message_text,
                )
            )
        )
        return

    cancelled = await session.cancel_current_run()
    if cancelled:
        await websocket.send(message_to_json(RequestOkMessage(request_id=message.request_id)))
        return

    state = session.state
    if not state.promptable:
        message_text = "Remote session is closed."
    else:
        message_text = "Remote session has no active run to cancel."
    await websocket.send(
        message_to_json(
            ErrorMessage(
                request_id=message.request_id,
                message=message_text,
            )
        )
    )


@asynccontextmanager
async def start_worker_server(
    *,
    session: AgentSession,
) -> AsyncIterator[WorkerServer]:
    """Serve one remote connection at a time for a live session."""
    connection_lock = asyncio.Lock()
    active_connection: ServerConnection | None = None

    async def handle_connection(websocket: ServerConnection) -> None:
        """Own one websocket connection for as long as it is the active remote client."""
        nonlocal active_connection

        async with connection_lock:
            if active_connection is not None:
                await websocket.send(
                    message_to_json(ErrorMessage(message="Session already has a remote client connected."))
                )
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
            async with connection_lock:
                if active_connection is websocket:
                    active_connection = None

    async with serve(handle_connection, "127.0.0.1", 0) as server:
        socket = server.sockets[0]
        port = socket.getsockname()[1]
        endpoint = f"ws://127.0.0.1:{port}"
        yield WorkerServer(endpoint=endpoint)
