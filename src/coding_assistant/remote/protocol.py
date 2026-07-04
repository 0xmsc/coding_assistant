from __future__ import annotations

from coding_assistant.core.session_updates import (
    AttachmentAddedUpdate,
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    MessageAddedUpdate,
    MessageDeltaUpdate,
    RunUpdatedUpdate,
    SessionAttachment,
    SessionUpdate,
    SessionUpdatedUpdate,
)
from coding_assistant.llm.types import BaseMessage, message_from_dict, message_to_dict
from coding_assistant.remote.acp import JsonObject, jsonrpc_notification


def _message_added_payload(update: MessageAddedUpdate) -> JsonObject:
    message = message_to_dict(update.message)
    message["id"] = update.message_id
    if update.created_at is not None:
        message["createdAt"] = update.created_at
    return message


def _message_added_from_payload(value: JsonObject) -> MessageAddedUpdate | None:
    message_id = value.get("id")
    created_at = value.get("createdAt")
    if not isinstance(message_id, str):
        return None
    message_payload = {key: item for key, item in value.items() if key not in {"id", "createdAt"}}
    return MessageAddedUpdate(
        message_id=message_id,
        message=message_from_dict(message_payload),
        created_at=created_at if isinstance(created_at, str) else None,
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

    if isinstance(update, MessageAddedUpdate):
        return {
            "sessionUpdate": "message_added",
            "message": _message_added_payload(update),
        }

    if isinstance(update, MessageDeltaUpdate):
        return {
            "sessionUpdate": "message_delta",
            "messageId": update.message_id,
            "appendText": update.append_text,
        }

    if isinstance(update, AttachmentAddedUpdate):
        attachment = update.attachment.to_json()
        if update.created_at is not None:
            attachment["createdAt"] = update.created_at
        return {
            "sessionUpdate": "attachment_added",
            "attachment": attachment,
        }

    if isinstance(update, SessionUpdatedUpdate):
        return {
            "sessionUpdate": "session_updated",
            "session": update.session,
        }

    if isinstance(update, RunUpdatedUpdate):
        return {
            "sessionUpdate": "run_updated",
            "run": update.run,
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

    if update_type == "message_added":
        message = update.get("message")
        if not isinstance(message, dict):
            return None
        return _message_added_from_payload(message)

    if update_type == "message_delta":
        message_id = update.get("messageId")
        append_text = update.get("appendText")
        if isinstance(message_id, str) and isinstance(append_text, str):
            return MessageDeltaUpdate(message_id=message_id, append_text=append_text)
        return None

    if update_type == "attachment_added":
        attachment = update.get("attachment")
        if not isinstance(attachment, dict):
            return None
        parsed_attachment = SessionAttachment.from_json(attachment)
        if parsed_attachment is None:
            return None
        created_at = attachment.get("createdAt")
        return AttachmentAddedUpdate(
            attachment=parsed_attachment,
            created_at=created_at if isinstance(created_at, str) else None,
        )

    if update_type == "session_updated":
        session = update.get("session")
        if not isinstance(session, dict):
            return None
        return SessionUpdatedUpdate(session=dict(session))

    if update_type == "run_updated":
        run = update.get("run")
        if not isinstance(run, dict):
            return None
        return RunUpdatedUpdate(run=dict(run))

    return None
