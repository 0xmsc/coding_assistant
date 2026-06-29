from __future__ import annotations

from coding_assistant.core.session_updates import (
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    SessionUpdate,
    SessionItem,
    SessionItemAddedUpdate,
    SessionItemDeltaUpdate,
    SessionItemUpdatedUpdate,
)
from coding_assistant.llm.types import BaseMessage, message_from_dict, message_to_dict
from coding_assistant.remote.acp import JsonObject, jsonrpc_notification


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

    return None
