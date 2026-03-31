from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

from prompt_toolkit.document import Document

from coding_assistant.app.terminal_ui import (
    SlashCompleter,
    TerminalUiMode,
    create_terminal_application,
    format_queued_prompts,
)
from coding_assistant.core.agent_session import SessionState


def test_slash_completer() -> None:
    words = ["/exit", "/compact", "/clear"]
    completer = SlashCompleter(words)

    doc = Document("/", cursor_position=1)
    completions = list(completer.get_completions(doc, cast(Any, None)))
    assert [c.text for c in completions] == ["/exit", "/compact", "/clear"]

    doc = Document("/e", cursor_position=2)
    completions = list(completer.get_completions(doc, cast(Any, None)))
    assert [c.text for c in completions] == ["/exit"]

    doc = Document(" /", cursor_position=2)
    completions = list(completer.get_completions(doc, cast(Any, None)))
    assert [c.text for c in completions] == []

    doc = Document("Hello /e", cursor_position=8)
    completions = list(completer.get_completions(doc, cast(Any, None)))
    assert [c.text for c in completions] == []

    doc = Document("e", cursor_position=1)
    completions = list(completer.get_completions(doc, cast(Any, None)))
    assert [c.text for c in completions] == []


def test_create_terminal_application_builds_non_fullscreen_interactive_ui(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(
        promptable=True,
        running=False,
        queued_prompt_count=2,
        pending_prompts=("first queued prompt", "second queued prompt"),
    )

    application, answer_queue = create_terminal_application(
        session=session,
        mode=TerminalUiMode.INTERACTIVE,
        history_path=tmp_path / "history",
        words=[],
    )

    assert application.full_screen is False
    assert application.refresh_interval == 0.1
    assert answer_queue is not None
    assert answer_queue.empty()


def test_create_terminal_application_builds_non_fullscreen_readonly_ui() -> None:
    session = Mock()
    session.state = SessionState(
        promptable=True,
        running=False,
        queued_prompt_count=0,
    )

    application, answer_queue = create_terminal_application(
        session=session,
        mode=TerminalUiMode.READONLY,
        history_path=None,
    )

    assert application.full_screen is False
    assert application.refresh_interval == 0.1
    assert answer_queue is None


def test_format_queued_prompts_shows_pending_prompts() -> None:
    session = Mock()
    session.state = SessionState(
        promptable=True,
        running=True,
        queued_prompt_count=3,
        pending_prompts=("first queued prompt", "second queued prompt", "third queued prompt"),
    )

    assert format_queued_prompts(session) == "first queued prompt\nsecond queued prompt\n+1 more"
