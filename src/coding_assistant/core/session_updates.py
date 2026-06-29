from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal
from uuid import uuid4

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
    ToolCall,
    ToolMessage,
    UserMessage,
)

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SessionItem:
    kind: str
    payload: JsonObject
    item_id: str = field(default_factory=lambda: f"item_{uuid4().hex}")
    sequence: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class HistoryResetUpdate:
    pass


@dataclass(frozen=True)
class HistoryCompleteUpdate:
    version: int


@dataclass(frozen=True)
class SessionItemAddedUpdate:
    item: SessionItem


@dataclass(frozen=True)
class SessionItemDeltaUpdate:
    item_id: str
    append_text: str


@dataclass(frozen=True)
class SessionItemUpdatedUpdate:
    item_id: str
    patch: JsonObject


@dataclass(frozen=True)
class UserMessageChunkUpdate:
    content: str


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
    title: str
    tool_kind: str
    status: str
    raw_input: JsonObject | None
    message: AssistantMessage | None


@dataclass(frozen=True)
class ToolCallLifecycleUpdate:
    source: str | None
    tool_call_id: str
    status: str
    title: str | None = None
    tool_kind: str | None = None
    raw_input: JsonObject | None = None
    raw_output: Any | None = None
    content: str | None = None
    display_content: list[JsonObject] | None = None


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
    | SessionItemAddedUpdate
    | SessionItemDeltaUpdate
    | SessionItemUpdatedUpdate
    | UserMessageChunkUpdate
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
class CommittedMessage:
    """One model-visible message committed to durable history."""

    role: str
    content: str | list[JsonObject] | None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class PromptResult:
    """Terminal result for one active prompt request."""

    source: str
    stop_reason: Literal["end_turn", "cancelled"] | None = None
    error: str | None = None


def _tool_call_raw_input(arguments: str) -> JsonObject | None:
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


def _display_content_from_tool_content(
    *,
    tool_name: str | None,
    content: str | list[JsonObject] | None,
) -> list[JsonObject] | None:
    if tool_name != "load_image" or not isinstance(content, list):
        return None
    blocks = [
        block
        for block in content
        if block.get("type") == "image_url"
        and isinstance(block.get("image_url"), dict)
        and isinstance(block["image_url"].get("url"), str)
    ]
    return blocks or None


def content_text(content: str | list[JsonObject] | None) -> str | None:
    return _committed_content_text(content)


def session_items_from_history_messages(messages: Sequence[BaseMessage]) -> list[SessionItem]:
    items: list[SessionItem] = []
    tool_item_indexes: dict[str, int] = {}

    def append_item(kind: str, payload: JsonObject) -> None:
        items.append(SessionItem(kind=kind, payload=payload))

    for message in messages:
        if isinstance(message, UserMessage):
            text = _committed_content_text(message.content)
            if text is not None:
                append_item("message", {"role": "user", "content": text})
            continue

        if isinstance(message, AssistantMessage):
            text = _committed_content_text(message.content)
            if text is not None:
                append_item("message", {"role": "assistant", "content": text})
            for tool_call in message.tool_calls:
                payload: JsonObject = {
                    "toolCallId": tool_call.id,
                    "title": tool_call.function.name or "tool_call",
                    "kind": "other",
                    "status": "pending",
                }
                raw_input = _tool_call_raw_input(tool_call.function.arguments)
                if raw_input is not None:
                    payload["rawInput"] = raw_input
                tool_item_indexes[tool_call.id] = len(items)
                append_item("tool_call", payload)
            continue

        if isinstance(message, ToolMessage):
            text = _committed_content_text(message.content)
            payload = {
                "toolCallId": message.tool_call_id,
                "status": "completed",
                "kind": "other",
                "rawOutput": message.content,
            }
            if message.name is not None:
                payload["title"] = message.name
            if text is not None:
                payload["content"] = text
            display_content = _display_content_from_tool_content(tool_name=message.name, content=message.content)
            if display_content is not None:
                payload["displayContent"] = display_content
            item_index = tool_item_indexes.get(message.tool_call_id)
            if item_index is None:
                append_item("tool_call", payload)
                continue
            existing = items[item_index]
            items[item_index] = replace(existing, payload={**existing.payload, **payload})

    return items


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
                title=tool_call.function.name or "tool_call",
                tool_kind="other",
                status="pending",
                raw_input=_tool_call_raw_input(tool_call.function.arguments),
                message=event.message,
            )
            for tool_call in event.message.tool_calls
        ]

    if isinstance(event, ToolCallUpdateEvent):
        return [
            ToolCallLifecycleUpdate(
                source=event.source,
                tool_call_id=event.event.tool_call_id,
                title=event.event.title,
                tool_kind=event.event.kind,
                status=event.event.status,
                raw_input=event.event.raw_input,
                raw_output=event.event.raw_output,
                content=event.event.content,
                display_content=event.event.display_content,
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


def committed_message_from_history_message(message: BaseMessage) -> CommittedMessage | None:
    if isinstance(message, UserMessage):
        return CommittedMessage(role="user", content=message.content)
    if isinstance(message, AssistantMessage):
        return CommittedMessage(role="assistant", content=message.content, tool_calls=message.tool_calls)
    if isinstance(message, ToolMessage):
        return CommittedMessage(
            role="tool",
            content=message.content,
            tool_call_id=message.tool_call_id,
            name=message.name,
        )
    return None


def replay_updates_from_committed_message(message: CommittedMessage) -> list[SessionUpdate]:
    if message.role == "user":
        text = _committed_content_text(message.content)
        if text is None:
            return []
        return [UserMessageChunkUpdate(content=text)]
    if message.role == "assistant":
        updates: list[SessionUpdate] = []
        text = _committed_content_text(message.content)
        if text is not None:
            updates.append(AgentMessageChunkUpdate(content=text))
        for tool_call in message.tool_calls or []:
            updates.append(
                ToolCallStartedUpdate(
                    source="history",
                    tool_call_id=tool_call.id,
                    title=tool_call.function.name or "tool_call",
                    tool_kind="other",
                    status="pending",
                    raw_input=_tool_call_raw_input(tool_call.function.arguments),
                    message=None,
                )
            )
        return updates
    if message.role == "tool" and message.tool_call_id:
        text = _committed_content_text(message.content)
        return [
            ToolCallLifecycleUpdate(
                source="history",
                tool_call_id=message.tool_call_id,
                status="completed",
                title=message.name,
                tool_kind="other",
                raw_output=message.content,
                content=text,
                display_content=_display_content_from_tool_content(
                    tool_name=message.name,
                    content=message.content,
                ),
            )
        ]
    return []
