from pathlib import Path

from coding_assistant.history_store import (
    FileHistoryStore,
    get_history_path,
    get_project_cache_dir,
    sanitize_history,
)
from coding_assistant.llm.types import AssistantMessage, FunctionCall, ToolCall, ToolMessage, UserMessage


def test_sanitize_history_with_empty_list() -> None:
    assert sanitize_history([]) == []


def test_sanitize_history_with_valid_history() -> None:
    history = [UserMessage(content="Hello"), AssistantMessage(content="Hi there!")]
    assert sanitize_history(history) == history


def test_sanitize_history_with_trailing_assistant_message_with_tool_calls() -> None:
    history = [
        UserMessage(content="Hello"),
        AssistantMessage(
            content="Thinking...",
            tool_calls=[ToolCall(id="123", function=FunctionCall(name="test", arguments="{}"))],
        ),
    ]
    assert sanitize_history(history) == [UserMessage(content="Hello")]


def test_sanitize_history_with_no_trailing_assistant_message() -> None:
    history = [
        UserMessage(content="Hello"),
        AssistantMessage(
            content="Thinking...",
            tool_calls=[ToolCall(id="123", function=FunctionCall(name="test", arguments="{}"))],
        ),
        ToolMessage(content="Result", tool_call_id="123"),
    ]
    assert sanitize_history(history) == history


def test_file_history_store_roundtrip(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)
    assert get_project_cache_dir(tmp_path).exists()
    assert store.path == get_history_path(tmp_path)

    store.save([UserMessage(content="msg-1")])
    data = store.load()
    assert data is not None
    assert data[-1].content == "msg-1"

    store.save([UserMessage(content="msg-2")])
    data = store.load()
    assert data is not None
    assert data[-1].content == "msg-2"


def test_file_history_store_strips_trailing_tool_calls(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)
    invalid = [
        UserMessage(content="hi"),
        AssistantMessage(
            content="Thinking...",
            tool_calls=[ToolCall(id="1", function=FunctionCall(name="x", arguments="{}"))],
        ),
    ]

    store.save(invalid)
    loaded = store.load()
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].content == "hi"
