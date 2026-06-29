from __future__ import annotations

from typing import Any, TypeGuard

from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    SessionUpdate,
    SessionItem,
    SessionItemAddedUpdate,
    SessionItemDeltaUpdate,
    SessionItemUpdatedUpdate,
    ToolCallLifecycleUpdate,
    ToolCallStartedUpdate,
    UserMessageChunkUpdate,
)
from coding_assistant.llm.types import BaseMessage, message_from_dict, message_to_dict
from coding_assistant.remote.acp import JsonObject, jsonrpc_notification, text_block, tool_content_text


def _is_json_object_list(value: Any) -> TypeGuard[list[JsonObject]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _content_text_from_tool_content(content: list[JsonObject] | None) -> str | None:
    if not content:
        return None
    first_item = content[0]
    if not (
        isinstance(first_item, dict)
        and isinstance(first_item.get("content"), dict)
        and first_item["content"].get("type") == "text"
        and isinstance(first_item["content"].get("text"), str)
    ):
        return None
    text = first_item["content"]["text"]
    if not isinstance(text, str):
        return None
    return text


def _session_item_to_json(item: SessionItem) -> JsonObject:
    payload: JsonObject = {
        "id": item.item_id,
        "kind": item.kind,
        "payload": item.payload,
    }
    if item.sequence is not None:
        payload["sequence"] = item.sequence
    if item.created_at is not None:
        payload["createdAt"] = item.created_at
    if item.updated_at is not None:
        payload["updatedAt"] = item.updated_at
    return payload


def _session_item_from_json(value: JsonObject) -> SessionItem | None:
    item_id = value.get("id")
    kind = value.get("kind")
    payload = value.get("payload")
    if not isinstance(item_id, str) or not isinstance(kind, str) or not isinstance(payload, dict):
        return None
    sequence = value.get("sequence")
    created_at = value.get("createdAt")
    updated_at = value.get("updatedAt")
    return SessionItem(
        item_id=item_id,
        kind=kind,
        payload=payload,
        sequence=sequence if isinstance(sequence, int) else None,
        created_at=created_at if isinstance(created_at, str) else None,
        updated_at=updated_at if isinstance(updated_at, str) else None,
    )


def session_update_to_jsonrpc_update(update: SessionUpdate) -> JsonObject | None:
    """Serialize one normalized update to a session/update payload update object."""
    if isinstance(update, HistoryResetUpdate):
        return {"sessionUpdate": "history_reset"}

    if isinstance(update, HistoryCompleteUpdate):
        return {
            "sessionUpdate": "history_complete",
            "version": update.version,
        }

    if isinstance(update, SessionItemAddedUpdate):
        return {
            "sessionUpdate": "item_added",
            "item": _session_item_to_json(update.item),
        }

    if isinstance(update, SessionItemDeltaUpdate):
        return {
            "sessionUpdate": "item_delta",
            "itemId": update.item_id,
            "appendText": update.append_text,
        }

    if isinstance(update, SessionItemUpdatedUpdate):
        return {
            "sessionUpdate": "item_updated",
            "itemId": update.item_id,
            "patch": update.patch,
        }

    if isinstance(update, UserMessageChunkUpdate):
        return {
            "sessionUpdate": "user_message_chunk",
            "content": text_block(update.content),
        }

    if isinstance(update, AgentMessageChunkUpdate):
        return {
            "sessionUpdate": "agent_message_chunk",
            "content": text_block(update.content),
        }

    if isinstance(update, ToolCallStartedUpdate):
        payload: JsonObject = {
            "sessionUpdate": "tool_call",
            "toolCallId": update.tool_call_id,
            "title": update.title,
            "kind": update.tool_kind,
            "status": update.status,
        }
        if update.raw_input is not None:
            payload["rawInput"] = update.raw_input
        return payload

    if isinstance(update, ToolCallLifecycleUpdate):
        payload = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": update.tool_call_id,
            "status": update.status,
        }
        if update.title is not None:
            payload["title"] = update.title
        if update.tool_kind is not None:
            payload["kind"] = update.tool_kind
        if update.raw_input is not None:
            payload["rawInput"] = update.raw_input
        if update.raw_output is not None:
            payload["rawOutput"] = update.raw_output
        if update.content:
            payload["content"] = tool_content_text(update.content)
        return payload

    return None


def session_update_notification(*, session_id: str, update: SessionUpdate) -> str | None:
    payload_update = session_update_to_jsonrpc_update(update)
    if payload_update is None:
        return None
    return jsonrpc_notification(
        "session/update",
        {
            "sessionId": session_id,
            "update": payload_update,
        },
    )


def messages_to_jsonrpc(messages: list[BaseMessage]) -> list[JsonObject]:
    return [message_to_dict(message) for message in messages]


def messages_from_jsonrpc(messages: list[JsonObject]) -> list[BaseMessage]:
    return [message_from_dict(message) for message in messages]


def session_update_from_jsonrpc_update(update: JsonObject) -> SessionUpdate | None:
    """Parse a session/update payload update object into a normalized update."""
    update_type = update.get("sessionUpdate")
    if update_type == "history_reset":
        return HistoryResetUpdate()

    if update_type == "history_complete":
        version = update.get("version")
        return HistoryCompleteUpdate(version=version if isinstance(version, int) else 0)

    if update_type == "item_added":
        item = update.get("item")
        if not isinstance(item, dict):
            return None
        parsed_item = _session_item_from_json(item)
        return SessionItemAddedUpdate(item=parsed_item) if parsed_item is not None else None

    if update_type == "item_delta":
        item_id = update.get("itemId")
        append_text = update.get("appendText")
        if isinstance(item_id, str) and isinstance(append_text, str):
            return SessionItemDeltaUpdate(item_id=item_id, append_text=append_text)
        return None

    if update_type == "item_updated":
        item_id = update.get("itemId")
        patch = update.get("patch")
        if isinstance(item_id, str) and isinstance(patch, dict):
            return SessionItemUpdatedUpdate(item_id=item_id, patch=patch)
        return None

    if update_type == "user_message_chunk":
        content = update.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
            return UserMessageChunkUpdate(content=content["text"])
        return None

    if update_type == "agent_message_chunk":
        content = update.get("content", {})
        if isinstance(content, dict) and content.get("type") == "text" and isinstance(content.get("text"), str):
            return AgentMessageChunkUpdate(content=content["text"])
        return None

    if update_type == "tool_call":
        title = update.get("title")
        if not isinstance(title, str):
            return None
        return ToolCallStartedUpdate(
            source="remote",
            tool_call_id=str(update.get("toolCallId", "")),
            title=title,
            tool_kind=str(update.get("kind", "other")),
            status=str(update.get("status", "pending")),
            raw_input=update.get("rawInput") if isinstance(update.get("rawInput"), dict) else None,
            message=None,
        )

    if update_type == "tool_call_update":
        content = update.get("content")
        raw_input = update.get("rawInput")
        return ToolCallLifecycleUpdate(
            source="remote",
            tool_call_id=str(update.get("toolCallId", "")),
            status=str(update.get("status", "")),
            title=update.get("title") if isinstance(update.get("title"), str) else None,
            tool_kind=update.get("kind") if isinstance(update.get("kind"), str) else None,
            raw_input=raw_input if isinstance(raw_input, dict) else None,
            raw_output=update.get("rawOutput"),
            content=_content_text_from_tool_content(content if _is_json_object_list(content) else None),
        )

    return None
