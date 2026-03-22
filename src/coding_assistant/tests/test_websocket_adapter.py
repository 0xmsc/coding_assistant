import pytest

from coding_assistant.adapters.websocket import session_event_to_json, websocket_command_from_json
from coding_assistant.llm.types import AssistantMessage, FunctionCall, ToolCall
from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    InputRequestedEvent,
    ToolCallRequestedEvent,
)


def test_session_event_to_json() -> None:
    assert session_event_to_json(AssistantDeltaEvent(delta="hi")) == {"type": "assistant_delta", "delta": "hi"}
    assert session_event_to_json(AssistantMessageEvent(message=AssistantMessage(content="hello"))) == {
        "type": "assistant_message",
        "message": {"content": "hello", "tool_calls": []},
    }
    assert session_event_to_json(
        ToolCallRequestedEvent(
            tool_call=ToolCall(id="1", function=FunctionCall(name="mock_tool", arguments="{}")),
            arguments={},
        )
    ) == {
        "type": "tool_call_requested",
        "tool_call": ToolCall(id="1", function=FunctionCall(name="mock_tool", arguments="{}")),
        "arguments": {},
    }
    assert session_event_to_json(InputRequestedEvent()) == {"type": "input_requested"}
    assert session_event_to_json(FailedEvent(error="boom")) == {"type": "failed", "error": "boom"}
    assert session_event_to_json(CancelledEvent()) == {"type": "cancelled"}


def test_websocket_command_from_json() -> None:
    assert websocket_command_from_json(
        {
            "type": "start",
            "initial_user_message": "Do it",
            "instructions": "Be precise.",
            "use_expert_model": True,
        }
    ).payload == {
        "initial_user_message": "Do it",
        "instructions": "Be precise.",
        "use_expert_model": True,
    }
    assert websocket_command_from_json({"type": "send_user_message", "content": "hello"}).payload == {
        "content": "hello"
    }
    assert websocket_command_from_json({"type": "submit_tool_result", "tool_call_id": "1", "result": "ok"}).payload == {
        "tool_call_id": "1",
        "result": "ok",
    }
    assert websocket_command_from_json({"type": "submit_tool_error", "tool_call_id": "1", "error": "boom"}).payload == {
        "tool_call_id": "1",
        "error": "boom",
    }
    assert websocket_command_from_json({"type": "cancel"}).name == "cancel"


def test_websocket_command_from_json_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported websocket command type"):
        websocket_command_from_json({"type": "unknown"})
