from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    InputRequestedEvent,
    SessionEvent,
    ToolCallRequestedEvent,
)


@dataclass(frozen=True)
class WebsocketCommand:
    name: Literal["start", "send_user_message", "submit_tool_result", "submit_tool_error", "cancel"]
    payload: dict[str, Any]


def session_event_to_json(event: SessionEvent) -> dict[str, Any]:
    if isinstance(event, AssistantDeltaEvent):
        return {"type": event.type, "delta": event.delta}
    if isinstance(event, AssistantMessageEvent):
        return {
            "type": event.type,
            "message": {"content": event.message.content, "tool_calls": event.message.tool_calls},
        }
    if isinstance(event, ToolCallRequestedEvent):
        return {
            "type": event.type,
            "tool_call": event.tool_call,
            "arguments": event.arguments,
        }
    if isinstance(event, InputRequestedEvent):
        return {"type": event.type}
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
                "initial_user_message": payload.get("initial_user_message"),
                "instructions": payload.get("instructions"),
                "use_expert_model": payload.get("use_expert_model", False),
            },
        )
    if message_type == "send_user_message":
        return WebsocketCommand(name="send_user_message", payload={"content": payload["content"]})
    if message_type == "submit_tool_result":
        return WebsocketCommand(
            name="submit_tool_result",
            payload={"tool_call_id": payload["tool_call_id"], "result": payload["result"]},
        )
    if message_type == "submit_tool_error":
        return WebsocketCommand(
            name="submit_tool_error",
            payload={"tool_call_id": payload["tool_call_id"], "error": payload["error"]},
        )
    if message_type == "cancel":
        return WebsocketCommand(name="cancel", payload={})
    raise ValueError(f"Unsupported websocket command type: {message_type}")
