import pytest

from coding_assistant.adapters.websocket import session_event_to_json, websocket_command_from_json
from coding_assistant.llm.types import AssistantMessage
from coding_assistant.runtime import (
    AssistantDeltaEvent,
    AssistantMessageEvent,
    CancelledEvent,
    FailedEvent,
    FinishedEvent,
    WaitingForUserEvent,
)


def test_session_event_to_json() -> None:
    assert session_event_to_json(AssistantDeltaEvent(delta="hi")) == {"type": "assistant_delta", "delta": "hi"}
    assert session_event_to_json(AssistantMessageEvent(message=AssistantMessage(content="hello"))) == {
        "type": "assistant_message",
        "message": {"content": "hello", "tool_calls": []},
    }
    assert session_event_to_json(WaitingForUserEvent()) == {"type": "waiting_for_user"}
    assert session_event_to_json(FinishedEvent(result="done", summary="sum")) == {
        "type": "finished",
        "result": "done",
        "summary": "sum",
    }
    assert session_event_to_json(FailedEvent(error="boom")) == {"type": "failed", "error": "boom"}
    assert session_event_to_json(CancelledEvent()) == {"type": "cancelled"}


def test_websocket_command_from_json() -> None:
    assert websocket_command_from_json({"type": "start", "mode": "agent", "task": "Do it"}).name == "start"
    assert websocket_command_from_json({"type": "send_user_message", "content": "hello"}).payload == {
        "content": "hello"
    }
    assert websocket_command_from_json({"type": "cancel"}).name == "cancel"


def test_websocket_command_from_json_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported websocket command type"):
        websocket_command_from_json({"type": "unknown"})
