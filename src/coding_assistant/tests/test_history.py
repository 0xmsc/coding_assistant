from pathlib import Path


from coding_assistant.history import (
    _fix_invalid_history,
    get_latest_orchestrator_history_file,
    get_orchestrator_history_file,
    get_project_cache_dir,
    load_orchestrator_history,
    save_orchestrator_history,
)


def test_fix_invalid_history_with_empty_list():
    assert _fix_invalid_history([]) == []


def test_fix_invalid_history_with_valid_history():
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    assert _fix_invalid_history(history) == history


def test_fix_invalid_history_with_trailing_assistant_message_with_tool_calls():
    history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Thinking...",
            "tool_calls": [{"id": "123", "function": {"name": "test", "arguments": "{}"}}],
        },
    ]
    assert _fix_invalid_history(history) == [{"role": "user", "content": "Hello"}]


def test_fix_invalid_history_with_no_trailing_assistant_message():
    history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Thinking...",
            "tool_calls": [{"id": "123", "function": {"name": "test", "arguments": "{}"}}],
        },
        {"role": "tool", "content": "Result", "tool_call_id": "123"},
    ]
    assert _fix_invalid_history(history) == history


def test_fix_invalid_history_with_multiple_trailing_assistant_messages():
    history = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Thinking...",
            "tool_calls": [{"id": "123", "function": {"name": "test", "arguments": "{}"}}],
        },
        {
            "role": "assistant",
            "content": "Thinking...",
            "tool_calls": [{"id": "456", "function": {"name": "test", "arguments": "{}"}}],
        },
    ]
    assert _fix_invalid_history(history) == [{"role": "user", "content": "Hello"}]


def test_orchestrator_history_roundtrip(tmp_path: Path):
    wd = tmp_path

    # Cache dir should exist
    cache_dir = get_project_cache_dir(wd)
    assert cache_dir.exists()

    # Save and read back
    save_orchestrator_history(wd, [{"role": "user", "content": "msg-1"}])
    latest = get_latest_orchestrator_history_file(wd)
    assert latest is not None and latest.exists()
    assert latest == get_orchestrator_history_file(wd)

    data = load_orchestrator_history(latest)
    assert isinstance(data, list) and data[-1]["content"] == "msg-1"

    # Overwrite
    save_orchestrator_history(wd, [{"role": "user", "content": "msg-2"}])
    data = load_orchestrator_history(latest)
    assert isinstance(data, list) and data[-1]["content"] == "msg-2"


def test_save_orchestrator_history_strips_trailing_assistant_tool_calls(tmp_path: Path):
    wd = tmp_path
    # An invalid history with trailing assistant tool_call should be fixed on save
    invalid = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Thinking...",
            "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}],
        },
    ]

    save_orchestrator_history(wd, invalid)
    latest = get_latest_orchestrator_history_file(wd)
    assert latest is not None
    fixed = load_orchestrator_history(latest)
    assert fixed == [{"role": "user", "content": "hi"}]
