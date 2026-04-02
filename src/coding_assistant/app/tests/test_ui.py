import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import FormattedTextControl

from coding_assistant.app.terminal_ui import (
    SlashCompleter,
    create_terminal_application,
    format_queued_prompts,
    run_terminal_ui,
)
from coding_assistant.core.agent_session import SessionState
from coding_assistant.llm.types import SystemMessage


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
        running=False,
        queued_prompt_count=2,
        pending_prompts=("first queued prompt", "second queued prompt"),
    )

    application, answer_queue = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    assert application.full_screen is False
    assert application.refresh_interval == 0.1
    assert answer_queue is not None
    assert answer_queue.empty()
    container = application.layout.container
    assert isinstance(container, HSplit)
    layout_children = container.children
    assert isinstance(layout_children[0], ConditionalContainer)
    input_row = layout_children[1]
    assert isinstance(input_row, VSplit)
    prompt_window = input_row.children[0]
    assert isinstance(prompt_window, Window)
    assert isinstance(prompt_window.content, FormattedTextControl)
    assert prompt_window.content.text == [("class:prompt", "> ")]


def test_format_queued_prompts_shows_pending_prompts() -> None:
    session = Mock()
    session.state = SessionState(
        running=True,
        queued_prompt_count=3,
        pending_prompts=("first queued prompt", "second queued prompt", "third queued prompt"),
    )

    assert format_queued_prompts(session) == "↳ first queued prompt\n↳ second queued prompt\n↳ +1 more"


@pytest.mark.asyncio
async def test_run_terminal_ui_prints_accepted_prompts_in_transcript() -> None:
    async def run_async() -> None:
        await asyncio.sleep(0)

    application = Mock()
    application.run_async = AsyncMock(side_effect=run_async)
    application.exit = Mock()
    application.is_done = True
    session = Mock()
    system_message = SystemMessage(content="System")
    answer_queue: asyncio.Queue[str] = asyncio.Queue()
    submit_handler = AsyncMock(return_value=False)

    with (
        patch("coding_assistant.app.terminal_ui.create_terminal_application", return_value=(application, answer_queue)),
        patch("coding_assistant.app.terminal_ui.run_session_output", new=AsyncMock()) as mock_run_session_output,
    ):
        await run_terminal_ui(
            session=session,
            system_message=system_message,
            history_path=Path("history"),
            submit_handler=submit_handler,
        )

    assert mock_run_session_output.await_args is not None
    assert mock_run_session_output.await_args.kwargs == {
        "session": session,
        "system_message": system_message,
    }
    submit_handler.assert_not_awaited()
