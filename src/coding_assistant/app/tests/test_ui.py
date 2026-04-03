from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock, patch

from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from coding_assistant.app.cli import (
    PromptSubmitType,
    SlashCompleter,
    _create_application,
    _format_queued_prompts,
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


def test_create_application_builds_non_fullscreen_interactive_ui(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(
        running=False,
        queued_prompt_count=2,
        pending_prompts=("first queued prompt", "second queued prompt"),
    )

    application, answer_queue = _create_application(
        session=session,
        history_path=tmp_path / "history",
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


def test_create_application_submits_as_steering_with_enter(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, answer_queue = _create_application(
        session=session,
        history_path=tmp_path / "history",
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
    assert submission.type == PromptSubmitType.STEERING


def test_create_application_submits_as_queued_with_tab(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, answer_queue = _create_application(
        session=session,
        history_path=tmp_path / "history",
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
    assert submission.type == PromptSubmitType.QUEUED


def test_create_application_inserts_newline_with_ctrl_j(tmp_path: Path) -> None:
    session = Mock()
    session.state = SessionState(running=False, queued_prompt_count=0)

    application, _ = _create_application(
        session=session,
        history_path=tmp_path / "history",
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

    assert _format_queued_prompts(session) == "↳ first queued prompt\n↳ second queued prompt\n↳ +1 more"
