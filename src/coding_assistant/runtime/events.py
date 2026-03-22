from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from coding_assistant.llm.types import AssistantMessage, ToolCall


@dataclass(frozen=True, kw_only=True)
class AssistantDeltaEvent:
    delta: str
    type: Literal["assistant_delta"] = "assistant_delta"


@dataclass(frozen=True, kw_only=True)
class AssistantMessageEvent:
    message: AssistantMessage
    type: Literal["assistant_message"] = "assistant_message"


@dataclass(frozen=True, kw_only=True)
class ToolCallRequestedEvent:
    tool_call: ToolCall
    arguments: dict[str, Any]
    type: Literal["tool_call_requested"] = "tool_call_requested"


@dataclass(frozen=True, kw_only=True)
class InputRequestedEvent:
    type: Literal["input_requested"] = "input_requested"


@dataclass(frozen=True, kw_only=True)
class FailedEvent:
    error: str
    type: Literal["failed"] = "failed"


@dataclass(frozen=True, kw_only=True)
class CancelledEvent:
    type: Literal["cancelled"] = "cancelled"


SessionEvent = (
    AssistantDeltaEvent
    | AssistantMessageEvent
    | ToolCallRequestedEvent
    | InputRequestedEvent
    | FailedEvent
    | CancelledEvent
)
