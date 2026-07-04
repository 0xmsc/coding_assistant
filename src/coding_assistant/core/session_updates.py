from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from coding_assistant.core.agent_session import (
    AgentSessionEvent,
    PromptContent,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    ToolCallsEvent,
    ToolCallUpdateEvent,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
)

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SessionAttachment:
    attachment_id: str
    name: str
    mime_type: str
    size: int
    path: str
    sha256: str

    def to_json(self) -> JsonObject:
        return {
            "id": self.attachment_id,
            "name": self.name,
            "mimeType": self.mime_type,
            "size": self.size,
            "path": self.path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, value: JsonObject) -> SessionAttachment | None:
        attachment_id = value.get("id")
        name = value.get("name")
        mime_type = value.get("mimeType")
        size = value.get("size")
        path = value.get("path")
        sha256 = value.get("sha256")
        if not (
            isinstance(attachment_id, str)
            and isinstance(name, str)
            and isinstance(mime_type, str)
            and isinstance(size, int)
            and isinstance(path, str)
            and isinstance(sha256, str)
        ):
            return None
        return cls(
            attachment_id=attachment_id,
            name=name,
            mime_type=mime_type,
            size=size,
            path=path,
            sha256=sha256,
        )


@dataclass(frozen=True)
class HistoryResetUpdate:
    pass


@dataclass(frozen=True)
class HistoryCompleteUpdate:
    version: int


@dataclass(frozen=True)
class MessageAddedUpdate:
    message_id: str
    message: BaseMessage
    created_at: str | None = None


@dataclass(frozen=True)
class MessageDeltaUpdate:
    message_id: str
    append_text: str


@dataclass(frozen=True)
class AttachmentAddedUpdate:
    attachment: SessionAttachment
    created_at: str | None = None


@dataclass(frozen=True)
class SessionUpdatedUpdate:
    session: JsonObject


@dataclass(frozen=True)
class RunUpdatedUpdate:
    run: JsonObject


@dataclass(frozen=True)
class AgentMessageChunkUpdate:
    content: str


@dataclass(frozen=True)
class ReasoningMessageChunkUpdate:
    content: str


@dataclass(frozen=True)
class StatusUpdate:
    content: str


@dataclass(frozen=True)
class PromptStartedUpdate:
    source: str
    prompt: PromptContent


@dataclass(frozen=True)
class ToolCallStartedUpdate:
    source: str
    tool_call_id: str
    tool_name: str
    status: str
    arguments: JsonObject | None
    message: AssistantMessage | None


@dataclass(frozen=True)
class ToolCallLifecycleUpdate:
    source: str | None
    tool_call_id: str
    tool_name: str
    status: str
    arguments: JsonObject | None = None
    error: str | None = None


@dataclass(frozen=True)
class RunFinishedUpdate:
    source: str
    summary: str


@dataclass(frozen=True)
class RunCancelledUpdate:
    source: str


@dataclass(frozen=True)
class RunFailedUpdate:
    source: str
    error: str


SessionUpdate = (
    HistoryResetUpdate
    | HistoryCompleteUpdate
    | MessageAddedUpdate
    | MessageDeltaUpdate
    | AttachmentAddedUpdate
    | SessionUpdatedUpdate
    | RunUpdatedUpdate
    | AgentMessageChunkUpdate
    | ReasoningMessageChunkUpdate
    | StatusUpdate
    | PromptStartedUpdate
    | ToolCallStartedUpdate
    | ToolCallLifecycleUpdate
    | RunFinishedUpdate
    | RunCancelledUpdate
    | RunFailedUpdate
)


@dataclass(frozen=True)
class PromptResult:
    """Terminal result for one active prompt request."""

    source: str
    stop_reason: Literal["end_turn", "cancelled"] | None = None
    error: str | None = None


def tool_call_arguments(arguments: str) -> JsonObject | None:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _committed_content_text(content: str | list[JsonObject] | None) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    if not text_parts:
        return None
    return "\n".join(text_parts)


def content_text(content: str | list[JsonObject] | None) -> str | None:
    return _committed_content_text(content)


def session_updates_from_agent_event(event: AgentSessionEvent) -> list[SessionUpdate]:
    """Convert raw AgentSession events to normalized updates."""
    if isinstance(event, ContentDeltaEvent):
        return [AgentMessageChunkUpdate(content=event.content)]

    if isinstance(event, ReasoningDeltaEvent):
        return [ReasoningMessageChunkUpdate(content=event.content)]

    if isinstance(event, StatusEvent):
        return [StatusUpdate(content=event.message)]

    if isinstance(event, PromptStartedEvent):
        return [PromptStartedUpdate(source=event.source, prompt=event.content)]

    if isinstance(event, ToolCallsEvent):
        return [
            ToolCallStartedUpdate(
                source=event.source,
                tool_call_id=tool_call.id,
                tool_name=tool_call.function.name,
                status="pending",
                arguments=tool_call_arguments(tool_call.function.arguments),
                message=event.message,
            )
            for tool_call in event.message.tool_calls
        ]

    if isinstance(event, ToolCallUpdateEvent):
        return [
            ToolCallLifecycleUpdate(
                source=event.source,
                tool_call_id=event.event.tool_call_id,
                tool_name=event.event.tool_name,
                status=event.event.status,
                arguments=event.event.arguments,
                error=event.event.error,
            ),
        ]

    if isinstance(event, RunFinishedEvent):
        return [RunFinishedUpdate(source=event.source, summary=event.summary)]

    if isinstance(event, RunCancelledEvent):
        return [RunCancelledUpdate(source=event.source)]

    if isinstance(event, RunFailedEvent):
        return [RunFailedUpdate(source=event.source, error=event.error)]

    return []


def prompt_result_from_update(update: SessionUpdate) -> PromptResult | None:
    if isinstance(update, RunFinishedUpdate):
        return PromptResult(source=update.source, stop_reason="end_turn")
    if isinstance(update, RunCancelledUpdate):
        return PromptResult(source=update.source, stop_reason="cancelled")
    if isinstance(update, RunFailedUpdate):
        return PromptResult(source=update.source, error=update.error)
    return None
