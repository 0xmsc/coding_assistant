from __future__ import annotations

from coding_assistant.core.session_updates import (
    AttachmentAddedUpdate,
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    MessageAddedUpdate,
    MessageDeltaUpdate,
    SessionAttachment,
    SessionUpdatedUpdate,
)
from coding_assistant.llm.types import AssistantMessage, UserMessage
from coding_assistant.remote.protocol import (
    session_update_from_jsonrpc_update,
    session_update_notification,
    session_update_to_jsonrpc_update,
)


def test_session_update_notification_wraps_message_update() -> None:
    notification = session_update_notification(
        session_id="sess_1",
        update=MessageDeltaUpdate(message_id="msg_1", append_text="hello"),
    )

    assert notification is not None
    assert '"method": "session/update"' in notification
    assert '"sessionId": "sess_1"' in notification
    assert '"sessionUpdate": "message_delta"' in notification


def test_session_updates_convert_to_jsonrpc_payloads_and_parse_back() -> None:
    attachment = SessionAttachment(
        attachment_id="att_1",
        name="image.png",
        mime_type="image/png",
        size=4,
        path="/attachments/att_1-image.png",
        sha256="abc",
    )

    assert session_update_to_jsonrpc_update(HistoryResetUpdate()) == {"sessionUpdate": "history_reset"}
    assert session_update_to_jsonrpc_update(
        MessageAddedUpdate(message_id="msg_1", message=AssistantMessage(content="hello"), created_at="now")
    ) == {
        "sessionUpdate": "message_added",
        "message": {
            "id": "msg_1",
            "role": "assistant",
            "content": "hello",
            "createdAt": "now",
        },
    }
    assert session_update_to_jsonrpc_update(MessageDeltaUpdate(message_id="msg_1", append_text="hello")) == {
        "sessionUpdate": "message_delta",
        "messageId": "msg_1",
        "appendText": "hello",
    }
    assert session_update_to_jsonrpc_update(AttachmentAddedUpdate(attachment=attachment, created_at="now")) == {
        "sessionUpdate": "attachment_added",
        "attachment": {
            "id": "att_1",
            "name": "image.png",
            "mimeType": "image/png",
            "size": 4,
            "path": "/attachments/att_1-image.png",
            "sha256": "abc",
            "createdAt": "now",
        },
    }
    assert session_update_to_jsonrpc_update(HistoryCompleteUpdate()) == {"sessionUpdate": "history_complete"}
    assert session_update_to_jsonrpc_update(
        SessionUpdatedUpdate(session={"sessionId": "sess_1", "title": "Updated"})
    ) == {
        "sessionUpdate": "session_updated",
        "session": {"sessionId": "sess_1", "title": "Updated"},
    }

    parsed = session_update_from_jsonrpc_update(
        {
            "sessionUpdate": "message_added",
            "message": {
                "id": "msg_2",
                "role": "user",
                "content": "hello",
            },
        },
    )

    assert parsed == MessageAddedUpdate(message_id="msg_2", message=UserMessage(content="hello"))
    assert session_update_from_jsonrpc_update(
        {"sessionUpdate": "session_updated", "session": {"sessionId": "sess_1"}}
    ) == SessionUpdatedUpdate(session={"sessionId": "sess_1"})


def test_unknown_session_update_shape_is_ignored() -> None:
    assert session_update_from_jsonrpc_update({"sessionUpdate": "unknown"}) is None
