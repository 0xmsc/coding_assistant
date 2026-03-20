from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    FinishedEvent,
    SessionEvent,
    WaitingForUserEvent,
)


@dataclass(frozen=True)
class WebsocketCommand:
    name: Literal["start", "send_user_message", "cancel"]
    payload: dict[str, Any]


def session_event_to_json(event: SessionEvent) -> dict[str, Any]:
    if isinstance(event, AssistantDeltaEvent):
        return {"type": event.type, "delta": event.delta}
    if isinstance(event, AssistantMessageEvent):
        return {
            "type": event.type,
            "message": {"content": event.message.content, "tool_calls": event.message.tool_calls},
        }
    if isinstance(event, WaitingForUserEvent):
        return {"type": event.type}
    if isinstance(event, FinishedEvent):
        return {"type": event.type, "result": event.result, "summary": event.summary}
    if isinstance(event, FailedEvent):
        return {"type": event.type, "error": event.error}
    if isinstance(event, CancelledEvent):
        return {"type": event.type}
    raise TypeError(f"Unsupported event type: {type(event).__name__}")


def websocket_command_from_json(payload: dict[str, Any]) -> WebsocketCommand:
    message_type = payload.get("type")
    if message_type == "start":
        return WebsocketCommand(
            name="start",
            payload={
                "mode": payload.get("mode", "chat"),
                "task": payload.get("task"),
                "instructions": payload.get("instructions"),
            },
        )
    if message_type == "send_user_message":
        return WebsocketCommand(name="send_user_message", payload={"content": payload["content"]})
    if message_type == "cancel":
        return WebsocketCommand(name="cancel", payload={})
    raise ValueError(f"Unsupported websocket command type: {message_type}")
