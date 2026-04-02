import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from coding_assistant.app.terminal_ui import (
    PromptSubmission,
    PromptSubmitType,
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

    application, answer_queue, _ = create_terminal_application(
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
    input_window = input_row.children[1]
    assert isinstance(input_window, Window)
    assert input_window.wrap_lines() is True


def test_create_terminal_application_submits_as_steering_with_enter(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, answer_queue, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    layout = cast(HSplit, application.layout.container)
    input_row = cast(VSplit, layout.children[1])
    assert isinstance(input_row, VSplit)
    input_window = input_row.children[1]
    assert isinstance(input_window, Window)
    assert isinstance(input_window.content, BufferControl)
    input_buffer = input_window.content.buffer
    input_buffer.text = "hello world"

    key_bindings = cast(KeyBindings, application.key_bindings)
    enter_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-m",))
    enter_binding.handler(cast(Any, None))

    submission = answer_queue.get_nowait()
    assert submission.content == "hello world"
    assert submission.submit_type == PromptSubmitType.STEERING


def test_create_terminal_application_submits_as_queued_with_tab(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, answer_queue, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    layout = cast(HSplit, application.layout.container)
    input_row = cast(VSplit, layout.children[1])
    assert isinstance(input_row, VSplit)
    input_window = input_row.children[1]
    assert isinstance(input_window, Window)
    assert isinstance(input_window.content, BufferControl)
    input_buffer = input_window.content.buffer
    input_buffer.text = "queued prompt"

    key_bindings = cast(KeyBindings, application.key_bindings)
    tab_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-i",))
    tab_binding.handler(cast(Any, None))

    submission = answer_queue.get_nowait()
    assert submission.content == "queued prompt"
    assert submission.submit_type == PromptSubmitType.QUEUED


def test_create_terminal_application_inserts_newline_with_ctrl_j(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, _, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    layout = cast(HSplit, application.layout.container)
    input_row = cast(VSplit, layout.children[1])
    assert isinstance(input_row, VSplit)
    input_window = input_row.children[1]
    assert isinstance(input_window, Window)
    assert isinstance(input_window.content, BufferControl)
    input_buffer = input_window.content.buffer

    key_bindings = cast(KeyBindings, application.key_bindings)
    newline_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-j",))
    with patch.object(input_buffer, "insert_text") as mock_insert_text:
        newline_binding.handler(cast(Any, None))

    mock_insert_text.assert_called_once_with("\n")


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
    answer_queue: asyncio.Queue[PromptSubmission] = asyncio.Queue()
    submit_handler = AsyncMock(return_value=False)

    with (
        patch(
            "coding_assistant.app.terminal_ui.create_terminal_application",
            return_value=(application, answer_queue, asyncio.Queue()),
        ),
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


@pytest.mark.asyncio
async def test_create_terminal_application_ctrl_c_cancels_active_run(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=True, queued_prompt_count=1, pending_prompts=("queued",))
    session.cancel_current_run = AsyncMock(return_value=True)

    application, _, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    key_bindings = cast(KeyBindings, application.key_bindings)
    ctrl_c_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-c",))
    event = Mock()
    event.app = Mock()
    event.app.invalidate = Mock()

    ctrl_c_binding.handler(event)
    await asyncio.sleep(0)

    session.cancel_current_run.assert_awaited_once_with(pause_queue=True)
    event.app.invalidate.assert_called_once()


@pytest.mark.asyncio
async def test_create_terminal_application_ctrl_c_resumes_paused_queue(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=1, paused=True, pending_prompts=("queued",))
    session.resume = AsyncMock(return_value=True)

    application, _, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    key_bindings = cast(KeyBindings, application.key_bindings)
    ctrl_c_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-c",))
    event = Mock()
    event.app = Mock()
    event.app.invalidate = Mock()

    ctrl_c_binding.handler(event)
    await asyncio.sleep(0)

    session.resume.assert_awaited_once_with()
    event.app.invalidate.assert_called_once()


@pytest.mark.asyncio
async def test_create_terminal_application_ctrl_c_clears_input_buffer_when_idle(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, _, _ = create_terminal_application(
        session=session,
        history_path=tmp_path / "history",
        words=[],
    )

    layout = cast(HSplit, application.layout.container)
    input_row = cast(VSplit, layout.children[1])
    input_window = input_row.children[1]
    assert isinstance(input_window, Window)
    assert isinstance(input_window.content, BufferControl)
    input_buffer = input_window.content.buffer
    input_buffer.text = "draft prompt"
    input_buffer.cursor_position = len(input_buffer.text)

    key_bindings = cast(KeyBindings, application.key_bindings)
    ctrl_c_binding = next(binding for binding in key_bindings.bindings if binding.keys == ("c-c",))
    event = Mock()
    event.app = Mock()
    event.app.invalidate = Mock()

    ctrl_c_binding.handler(event)
    await asyncio.sleep(0)

    assert input_buffer.text == ""
    assert input_buffer.cursor_position == 0
    event.app.invalidate.assert_called_once()
