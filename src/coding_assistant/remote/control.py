from __future__ import annotations

import asyncio
from uuid import uuid4

from websockets.asyncio.server import ServerConnection

from coding_assistant.core.agent_session import (
    AgentSessionEvent,
    AssistantMessageCompletedEvent,
    AssistantMessageDeltaEvent,
    HistoryResetEvent,
    RunOutcome,
)
from coding_assistant.core.session_updates import (
    HistoryResetUpdate,
    MessageAddedUpdate,
    MessageDeltaUpdate,
    SessionUpdate,
)
from coding_assistant.core.tool_calls import ToolMessageProduced
from coding_assistant.remote.jsonrpc import JsonObject
from coding_assistant.remote.protocol import session_update_notification


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
    if isinstance(event, HistoryResetEvent):
        return [
            HistoryResetUpdate(),
            *(MessageAddedUpdate(message_id=f"msg_{uuid4().hex}", message=message) for message in event.history),
        ]
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
