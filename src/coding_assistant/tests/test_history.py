import json
from pathlib import Path
from src.coding_assistant.history import _fix_invalid_history


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
