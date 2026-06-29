from __future__ import annotations

from coding_assistant.core.agent_session import RunFinishedEvent
from coding_assistant.core.session_updates import (
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    SessionItem,
    SessionItemAddedUpdate,
    SessionItemDeltaUpdate,
    SessionItemUpdatedUpdate,
    prompt_result_from_update,
    session_updates_from_agent_event,
)
from coding_assistant.remote.protocol import (
    session_update_from_jsonrpc_update,
    session_update_notification,
    session_update_to_jsonrpc_update,
)


def test_session_update_notification_wraps_item_update() -> None:
    notification = session_update_notification(
        session_id="sess_1",
        update=SessionItemDeltaUpdate(item_id="item_1", append_text="hello"),
    )

    assert notification is not None
    assert '"method": "session/update"' in notification
    assert '"sessionId": "sess_1"' in notification
    assert '"sessionUpdate": "item_delta"' in notification


def test_item_updates_convert_to_jsonrpc_payloads_and_parse_back() -> None:
    item = SessionItem(item_id="item_1", kind="message", payload={"role": "assistant", "content": ""})

    assert session_update_to_jsonrpc_update(HistoryResetUpdate()) == {"sessionUpdate": "history_reset"}
    assert session_update_to_jsonrpc_update(SessionItemAddedUpdate(item=item)) == {
        "sessionUpdate": "item_added",
        "item": {
            "id": "item_1",
            "kind": "message",
            "payload": {"role": "assistant", "content": ""},
        },
    }
    assert session_update_to_jsonrpc_update(SessionItemDeltaUpdate(item_id="item_1", append_text="hello")) == {
        "sessionUpdate": "item_delta",
        "itemId": "item_1",
        "appendText": "hello",
    }
    assert session_update_to_jsonrpc_update(
        SessionItemUpdatedUpdate(item_id="item_1", patch={"payload": {"content": "hello"}})
    ) == {
        "sessionUpdate": "item_updated",
        "itemId": "item_1",
        "patch": {"payload": {"content": "hello"}},
    }
    assert session_update_to_jsonrpc_update(HistoryCompleteUpdate(version=3)) == {
        "sessionUpdate": "history_complete",
        "version": 3,
    }

    parsed = session_update_from_jsonrpc_update(
        {
            "sessionUpdate": "item_added",
            "item": {
                "id": "item_1",
                "kind": "message",
                "payload": {"role": "assistant", "content": ""},
            },
        },
    )

    assert parsed == SessionItemAddedUpdate(item=item)


def test_unknown_session_update_shape_is_ignored() -> None:
    assert session_update_from_jsonrpc_update({"sessionUpdate": "unknown"}) is None


def test_run_finished_update_converts_to_prompt_result() -> None:
    updates = session_updates_from_agent_event(RunFinishedEvent(summary="done", source="remote:test"))

    assert len(updates) == 1
    result = prompt_result_from_update(updates[0])

    assert result is not None
    assert result.source == "remote:test"
    assert result.stop_reason == "end_turn"
    assert result.error is None
