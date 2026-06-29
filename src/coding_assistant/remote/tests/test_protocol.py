from __future__ import annotations

from coding_assistant.core.agent_session import RunFinishedEvent, ToolCallsEvent, ToolCallUpdateEvent
from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    HistoryCompleteUpdate,
    HistoryResetUpdate,
    SessionItem,
    SessionItemAddedUpdate,
    SessionItemDeltaUpdate,
    SessionItemUpdatedUpdate,
    ToolCallLifecycleUpdate,
    ToolCallStartedUpdate,
    UserMessageChunkUpdate,
    committed_message_from_history_message,
    prompt_result_from_update,
    replay_updates_from_committed_message,
    session_updates_from_agent_event,
)
from coding_assistant.core.tool_calls import ToolCallLifecycleEvent
from coding_assistant.llm.types import (
    AssistantMessage,
    ContentDeltaEvent,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.remote.protocol import (
    session_update_notification,
    session_update_from_jsonrpc_update,
    session_update_to_jsonrpc_update,
)


IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def test_content_delta_converts_to_normalized_update_and_jsonrpc_payload() -> None:
    updates = session_updates_from_agent_event(ContentDeltaEvent(content="hello"))

    assert len(updates) == 1
    assert isinstance(updates[0], AgentMessageChunkUpdate)
    assert updates[0].content == "hello"
    assert session_update_to_jsonrpc_update(updates[0]) == {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "hello"},
    }


def test_session_update_notification_wraps_serialized_update() -> None:
    notification = session_update_notification(
        session_id="sess_1",
        update=AgentMessageChunkUpdate(content="hello"),
    )

    assert notification is not None
    assert '"method": "session/update"' in notification
    assert '"sessionId": "sess_1"' in notification


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


def test_tool_calls_convert_to_normalized_updates_and_jsonrpc_payloads() -> None:
    message = AssistantMessage(
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="shell_execute", arguments='{"command": "pwd"}'),
            ),
        ],
    )

    updates = session_updates_from_agent_event(ToolCallsEvent(message=message, source="remote:test"))

    assert len(updates) == 1
    assert isinstance(updates[0], ToolCallStartedUpdate)
    assert updates[0].source == "remote:test"
    assert updates[0].message is message
    assert session_update_to_jsonrpc_update(updates[0]) == {
        "sessionUpdate": "tool_call",
        "toolCallId": "call-1",
        "title": "shell_execute",
        "kind": "other",
        "status": "pending",
        "rawInput": {"command": "pwd"},
    }


def test_tool_call_lifecycle_update_converts_to_normalized_update_and_jsonrpc_payload() -> None:
    event = ToolCallLifecycleEvent(
        tool_call_id="call-1",
        tool_name="shell_execute",
        status="completed",
        title="shell_execute",
        kind="other",
        raw_input={"command": "pwd"},
        raw_output="/workspace",
        content="/workspace",
    )

    updates = session_updates_from_agent_event(ToolCallUpdateEvent(event=event, source="remote:test"))

    assert len(updates) == 1
    assert isinstance(updates[0], ToolCallLifecycleUpdate)
    assert updates[0].source == "remote:test"
    assert session_update_to_jsonrpc_update(updates[0]) == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "call-1",
        "status": "completed",
        "title": "shell_execute",
        "kind": "other",
        "rawInput": {"command": "pwd"},
        "rawOutput": "/workspace",
        "content": [{"type": "content", "content": {"type": "text", "text": "/workspace"}}],
    }


def test_jsonrpc_tool_call_update_parses_to_normalized_update() -> None:
    update = session_update_from_jsonrpc_update(
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "status": "completed",
            "title": "shell_execute",
            "rawInput": {"command": "pwd"},
            "rawOutput": "/workspace",
            "content": [{"type": "content", "content": {"type": "text", "text": "done"}}],
            "displayContent": [IMAGE_BLOCK],
        },
    )

    assert update is not None
    assert isinstance(update, ToolCallLifecycleUpdate)
    assert update.tool_call_id == "call-1"
    assert update.status == "completed"
    assert update.title == "shell_execute"
    assert update.raw_input == {"command": "pwd"}
    assert update.raw_output == "/workspace"
    assert update.content == "done"
    assert update.display_content == [IMAGE_BLOCK]


def test_load_image_lifecycle_update_serializes_display_content() -> None:
    update = ToolCallLifecycleUpdate(
        source="remote:test",
        tool_call_id="call-1",
        title="load_image",
        tool_kind="read",
        status="completed",
        raw_input={"path": "/attachments/att_1-meal.png"},
        raw_output="loaded image",
        content="loaded image",
        display_content=[IMAGE_BLOCK],
    )

    assert session_update_to_jsonrpc_update(update) == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "call-1",
        "status": "completed",
        "title": "load_image",
        "kind": "read",
        "rawInput": {"path": "/attachments/att_1-meal.png"},
        "rawOutput": "loaded image",
        "content": [{"type": "content", "content": {"type": "text", "text": "loaded image"}}],
        "displayContent": [IMAGE_BLOCK],
    }


def test_run_finished_update_converts_to_prompt_result() -> None:
    updates = session_updates_from_agent_event(RunFinishedEvent(summary="done", source="remote:test"))

    assert len(updates) == 1
    result = prompt_result_from_update(updates[0])

    assert result is not None
    assert result.source == "remote:test"
    assert result.stop_reason == "end_turn"
    assert result.error is None


def test_committed_user_message_converts_to_replay_update_and_jsonrpc_payload() -> None:
    committed = committed_message_from_history_message(UserMessage(content="hello"))

    assert committed is not None
    updates = replay_updates_from_committed_message(committed)

    assert len(updates) == 1
    assert isinstance(updates[0], UserMessageChunkUpdate)
    assert updates[0].content == "hello"
    assert session_update_to_jsonrpc_update(updates[0]) == {
        "sessionUpdate": "user_message_chunk",
        "content": {"type": "text", "text": "hello"},
    }


def test_committed_assistant_message_converts_to_replay_update_and_jsonrpc_payload() -> None:
    committed = committed_message_from_history_message(AssistantMessage(content="answer"))

    assert committed is not None
    updates = replay_updates_from_committed_message(committed)

    assert len(updates) == 1
    assert isinstance(updates[0], AgentMessageChunkUpdate)
    assert updates[0].content == "answer"
    assert session_update_to_jsonrpc_update(updates[0]) == {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": "answer"},
    }


def test_committed_tool_call_history_converts_to_replay_updates_and_jsonrpc_payloads() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="shell_execute", arguments='{"command": "cat smoke.txt"}'),
    )

    assistant = committed_message_from_history_message(AssistantMessage(tool_calls=[tool_call]))
    tool_result = committed_message_from_history_message(
        ToolMessage(tool_call_id="call-1", name="shell_execute", content="smoke output")
    )

    assert assistant is not None
    assert tool_result is not None
    updates = [
        *replay_updates_from_committed_message(assistant),
        *replay_updates_from_committed_message(tool_result),
    ]

    assert [session_update_to_jsonrpc_update(update) for update in updates] == [
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call-1",
            "title": "shell_execute",
            "kind": "other",
            "status": "pending",
            "rawInput": {"command": "cat smoke.txt"},
        },
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "status": "completed",
            "title": "shell_execute",
            "kind": "other",
            "rawOutput": "smoke output",
            "content": [{"type": "content", "content": {"type": "text", "text": "smoke output"}}],
        },
    ]


def test_committed_load_image_tool_call_replays_display_content() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="load_image", arguments='{"path": "/attachments/att_1-meal.png"}'),
    )

    assistant = committed_message_from_history_message(AssistantMessage(tool_calls=[tool_call]))
    tool_result = committed_message_from_history_message(
        ToolMessage(
            tool_call_id="call-1",
            name="load_image",
            content=[
                {"type": "text", "text": "loaded image"},
                IMAGE_BLOCK,
            ],
        )
    )

    assert assistant is not None
    assert tool_result is not None
    updates = [
        *replay_updates_from_committed_message(assistant),
        *replay_updates_from_committed_message(tool_result),
    ]

    assert [session_update_to_jsonrpc_update(update) for update in updates] == [
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call-1",
            "title": "load_image",
            "kind": "other",
            "status": "pending",
            "rawInput": {"path": "/attachments/att_1-meal.png"},
        },
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "status": "completed",
            "title": "load_image",
            "kind": "other",
            "rawOutput": [
                {"type": "text", "text": "loaded image"},
                IMAGE_BLOCK,
            ],
            "content": [{"type": "content", "content": {"type": "text", "text": "loaded image"}}],
            "displayContent": [IMAGE_BLOCK],
        },
    ]
