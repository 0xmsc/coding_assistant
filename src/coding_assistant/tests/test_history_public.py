from pathlib import Path
from datetime import datetime, timedelta

import pytest

from coding_assistant.history import (
    get_project_cache_dir,
    get_conversation_history_file,
    get_conversation_summaries,
    save_conversation_summary,
    get_orchestrator_history_dir,
    save_orchestrator_history,
    get_latest_orchestrator_history_file,
    load_orchestrator_history,
    clear_orchestrator_history,
    trim_orchestrator_history,
)


def test_conversation_summaries_roundtrip(tmp_path: Path):
    wd = tmp_path

    # Initially none
    assert get_conversation_summaries(wd) == []

    # Save two summaries and read back
    save_conversation_summary(wd, "first")
    save_conversation_summary(wd, "second")
    assert get_conversation_summaries(wd) == ["first", "second"]

    # Files exist in expected location
    path = get_conversation_history_file(wd)
    assert path.exists()


def test_orchestrator_history_roundtrip_and_trim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    wd = tmp_path

    # Directory creation helpers should work
    cache_dir = get_project_cache_dir(wd)
    assert cache_dir.exists()
    hist_dir = get_orchestrator_history_dir(wd)
    assert hist_dir.exists()

    # Ensure unique filenames by faking datetime.now()
    base = datetime(2024, 1, 1, 0, 0, 0)
    times = [base + timedelta(seconds=i) for i in range(12)]

    class _FakeDT:
        def now(self):
            return times.pop(0)

    monkeypatch.setattr("coding_assistant.history.datetime", _FakeDT(), raising=False)

    # Save multiple histories
    for i in range(12):
        save_orchestrator_history(wd, [{"role": "user", "content": f"msg-{i}"}])

    # Latest file available
    latest = get_latest_orchestrator_history_file(wd)
    assert latest is not None
    assert latest.exists()
    data = load_orchestrator_history(latest)
    assert isinstance(data, list) and data[-1]["content"].startswith("msg-")

    # Trim to keep 5 most recent
    trim_orchestrator_history(wd, keep=5)
    assert len(list(hist_dir.glob("history_*.json"))) == 5

    # Clear all
    clear_orchestrator_history(wd)
    assert len(list(hist_dir.glob("history_*.json"))) == 0


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
