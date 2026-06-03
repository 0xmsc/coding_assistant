from __future__ import annotations

from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    SessionUpdate,
    ToolCallLifecycleUpdate,
    ToolCallStartedUpdate,
    UserMessageChunkUpdate,
)
from coding_assistant.remote.acp import JsonObject, text_block, tool_content_text


def _content_text_from_tool_content(content: object) -> str | None:
    if not isinstance(content, list) or not content:
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


def session_update_to_jsonrpc_update(update: SessionUpdate) -> JsonObject | None:
    """Serialize one normalized update to a session/update payload update object."""
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


def session_update_from_jsonrpc_update(update: JsonObject) -> SessionUpdate | None:
    """Parse a session/update payload update object into a normalized update."""
    update_type = update.get("sessionUpdate")
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
        return ToolCallLifecycleUpdate(
            source="remote",
            tool_call_id=str(update.get("toolCallId", "")),
            status=str(update.get("status", "")),
            title=update.get("title") if isinstance(update.get("title"), str) else None,
            tool_kind=update.get("kind") if isinstance(update.get("kind"), str) else None,
            content=_content_text_from_tool_content(update.get("content")),
        )

    return None
