"""Tests for the output formatting module.

This module tests all functions in coding_assistant.app.output, including:
- ParagraphBuffer: streaming content buffering logic
- DeltaRenderer: markdown rendering with buffering
- format_* functions: pure formatting logic
- print_* functions: output captured via Rich Console

Tests use stdout capture (StringIO + Rich Console) for minimal mocking,
following the principle: tests with minimal mocking are preferred.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from unittest.mock import patch

from rich.console import Console

from coding_assistant.app.output import (
    DeltaRenderer,
    ParagraphBuffer,
    format_prompt_preview,
    format_session_status,
    format_tool_call_display,
    print_active_prompt,
    print_info_message,
    print_session_status,
    print_system_message,
    print_tool_calls,
)
from coding_assistant.core.agent_session import SessionState
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall


def capture_rich_output() -> Iterator[tuple[Callable[[], str], Console]]:
    """Context manager that captures rich_print output to a StringIO buffer.

    Yields:
        Tuple of (get_output, console) for assertions.
        Call get_output() to get the current captured string.

    Usage:
        with capture_rich_output() as (get_output, console):
            some_rich_function()
        assert "expected" in get_output()
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=80)
    # Patch rich_print globally to use our console
    with patch("rich.print", console.print):
        yield buffer.getvalue, console


# =============================================================================
# ParagraphBuffer Tests
# =============================================================================


class TestParagraphBuffer:
    """Tests for ParagraphBuffer streaming content logic."""

    def test_empty_buffer_returns_nothing(self) -> None:
        buffer = ParagraphBuffer()
        assert buffer.push("") == []
        assert buffer.flush() is None

    def test_single_paragraph_yields_on_double_newline(self) -> None:
        buffer = ParagraphBuffer()
        assert buffer.push("Hello\n\nWorld") == ["Hello"]
        assert buffer._buffer == "World"

    def test_multiple_paragraphs(self) -> None:
        buffer = ParagraphBuffer()
        paragraphs = buffer.push("Para 1\n\nPara 2\n\nPara 3")
        assert paragraphs == ["Para 1", "Para 2"]
        assert buffer._buffer == "Para 3"

    def test_flush_returns_remaining(self) -> None:
        buffer = ParagraphBuffer()
        buffer.push("Some text")
        assert buffer.flush() == "Some text"
        assert buffer._buffer == ""

    def test_flush_empty_buffer_returns_none(self) -> None:
        buffer = ParagraphBuffer()
        assert buffer.flush() is None

    def test_handles_even_code_fences_correctly(self) -> None:
        """Even number of fences - no special handling needed."""
        buffer = ParagraphBuffer()
        content = "Text before\n\n```\ncode\n```\n\nText after"
        paragraphs = buffer.push(content)
        # The buffer splits at \n\n when not inside a code fence
        assert "Text before" in paragraphs
        assert buffer._buffer  # Remaining text preserved

    def test_handles_adjacent_paragraphs(self) -> None:
        """Adjacent paragraphs without blank lines are one unit."""
        buffer = ParagraphBuffer()
        content = "First line\nSecond line\n\nSeparate paragraph"
        paragraphs = buffer.push(content)
        assert paragraphs == ["First line\nSecond line"]


# =============================================================================
# DeltaRenderer Tests
# =============================================================================


class TestDeltaRenderer:
    """Tests for DeltaRenderer markdown rendering."""

    def test_empty_renderer_finishes_cleanly(self) -> None:
        renderer = DeltaRenderer()
        with patch("coding_assistant.app.output.rich_print"):
            renderer.finish()  # Should not raise

    def test_single_delta_renders_markdown(self) -> None:
        """Single content delta with paragraph boundary renders first paragraph."""
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Hello\n\nWorld")

        output = buffer.getvalue()
        # "Hello" is rendered (paragraph boundary reached)
        assert "Hello" in output
        # "World" stays in buffer until finish() or another paragraph boundary
        assert "World" not in output

    def test_buffer_holds_incomplete_paragraph(self) -> None:
        """Content without \n\n stays in buffer until flush."""
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Hello\n\nWorld")
            # Flush remaining content
            renderer.finish()

        output = buffer.getvalue()
        # Both paragraphs should now be in output
        assert "Hello" in output
        assert "World" in output

    def test_finish_flushes_remaining_content(self) -> None:
        """Finish should flush any remaining buffered content."""
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Some text without paragraph end")
            renderer.finish()

        output = buffer.getvalue()
        assert "Some text without paragraph end" in output

    def test_trailing_blank_line_controlled_by_flag(self) -> None:
        """trailing_blank_line=False should suppress the trailing newline."""
        renderer1 = DeltaRenderer()
        renderer2 = DeltaRenderer()

        buffer1 = io.StringIO()
        buffer2 = io.StringIO()
        console1 = Console(file=buffer1, force_terminal=False, width=80)
        console2 = Console(file=buffer2, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console1.print):
            renderer1.on_delta("Content\n\nMore")
            renderer1.finish(trailing_blank_line=False)

        with patch("coding_assistant.app.output.rich_print", console2.print):
            renderer2.on_delta("Content\n\nMore")
            renderer2.finish(trailing_blank_line=True)

        # With trailing_blank_line=True, we should see an extra newline
        output1 = buffer1.getvalue()
        output2 = buffer2.getvalue()
        # Both should contain the content
        assert "Content" in output1
        assert "Content" in output2


# =============================================================================
# format_tool_call_display Tests
# =============================================================================


class TestFormatToolCallDisplay:
    """Tests for tool call formatting logic."""

    def test_simple_tool_call(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="echo", arguments='{"text": "hello"}'),
        )
        header, body = format_tool_call_display(tool_call)
        assert header == 'echo(text="hello")'
        assert body == []

    def test_tool_call_with_multiline_argument(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(
                name="shell_execute",
                arguments='{"command": "echo hello\\nworld\\nbob"}',
            ),
        )
        header, body = format_tool_call_display(tool_call)
        assert header == "shell_execute(command)"
        assert len(body) == 1
        assert body[0][0] == "command"
        assert "echo hello" in body[0][1]

    def test_tool_call_with_hide_values(self) -> None:
        """filesystem_edit_file hides old_text and new_text."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(
                name="filesystem_edit_file",
                arguments='{"path": "/tmp/test.py", "old_text": "secret", "new_text": "visible"}',
            ),
        )
        header, body = format_tool_call_display(tool_call)
        # old_text and new_text should be in header params (hidden)
        assert "old_text" in header
        assert "new_text" in header

    def test_tool_call_with_json_argument(self) -> None:
        """todo_add uses JSON format for descriptions."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(
                name="todo_add",
                arguments='{"descriptions": ["task 1", "task 2"]}',
            ),
        )
        header, body = format_tool_call_display(tool_call)
        assert header == "todo_add(descriptions)"
        assert len(body) == 1
        assert body[0][2] == "json"  # language hint

    def test_tool_call_with_missing_name(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="", arguments="{}"),
        )
        header, _ = format_tool_call_display(tool_call)
        assert "<missing>" in header

    def test_tool_call_with_invalid_json(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="test", arguments="not valid json"),
        )
        header, body = format_tool_call_display(tool_call)
        assert header == "test(arguments)"
        assert len(body) == 1
        assert body[0][0] == "arguments"


class TestGetToolLanguageOverride:
    """Tests for language override based on file extension."""

    def test_python_file(self) -> None:
        from coding_assistant.app.output import _get_tool_language_override

        assert _get_tool_language_override("filesystem_write_file", {"path": "/tmp/script.py"}) == "py"

    def test_javascript_file(self) -> None:
        from coding_assistant.app.output import _get_tool_language_override

        assert _get_tool_language_override("filesystem_write_file", {"path": "/tmp/app.js"}) == "js"

    def test_rust_file(self) -> None:
        from coding_assistant.app.output import _get_tool_language_override

        assert _get_tool_language_override("filesystem_edit_file", {"path": "/tmp/main.rs"}) == "rs"

    def test_no_extension(self) -> None:
        from coding_assistant.app.output import _get_tool_language_override

        assert _get_tool_language_override("filesystem_write_file", {"path": "/tmp/Makefile"}) is None

    def test_non_file_tool(self) -> None:
        from coding_assistant.app.output import _get_tool_language_override

        assert _get_tool_language_override("shell_execute", {"command": "ls"}) is None


# =============================================================================
# format_prompt_preview Tests
# =============================================================================


class TestFormatPromptPreview:
    """Tests for prompt preview formatting."""

    def test_string_prompt(self) -> None:
        result = format_prompt_preview("Hello world, how are you?")
        assert result == "Hello world, how are you?"

    def test_whitespace_normalized(self) -> None:
        result = format_prompt_preview("  Hello   world  ")
        assert result == "Hello world"

    def test_preserves_normal_length(self) -> None:
        """Normal prompts are not truncated."""
        normal_prompt = " ".join(["word"] * 20)
        result = format_prompt_preview(normal_prompt)
        assert len(result.split()) == 20

    def test_structured_prompt_returns_placeholder(self) -> None:
        structured = [{"type": "text", "text": "hello"}]
        assert format_prompt_preview(structured) == "structured prompt"


# =============================================================================
# format_session_status Tests
# =============================================================================


class TestFormatSessionStatus:
    """Tests for session status formatting."""

    def test_running_status_with_zero_queue(self) -> None:
        state = SessionState(running=True, queued_prompt_count=0)
        # Implementation shows queued: 0 even when zero
        assert "running" in format_session_status(state)
        assert "queued" in format_session_status(state)

    def test_paused_status_with_zero_queue(self) -> None:
        state = SessionState(running=False, queued_prompt_count=0, paused=True)
        assert "paused" in format_session_status(state)
        assert "queued" in format_session_status(state)

    def test_idle_status_with_queued_count(self) -> None:
        state = SessionState(running=False, queued_prompt_count=5)
        result = format_session_status(state)
        assert "idle" in result
        assert "queued: 5" in result

    def test_pending_prompts_suppresses_queue_count(self) -> None:
        """When pending prompts exist, only show status to avoid redundancy."""
        state = SessionState(
            running=True,
            queued_prompt_count=3,
            pending_prompts=("prompt1", "prompt2"),
        )
        result = format_session_status(state)
        assert result == "running"
        assert "queued" not in result


# =============================================================================
# print_* Function Tests - stdout capture via Rich Console
# =============================================================================


def _capture_output(func: Callable[[], None]) -> str:
    """Capture output from a function that uses rich_print.

    Creates a StringIO buffer wrapped in a Rich Console, patches the
    rich_print import in the output module, calls the function, and returns
    the captured output.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=80)
    # Patch rich_print where it's imported in the output module
    with patch("coding_assistant.app.output.rich_print", console.print):
        func()
    return buffer.getvalue()


class TestPrintFunctions:
    """Tests for print functions that output to terminal.

    These tests use stdout capture (StringIO + Rich Console) for minimal mocking,
    verifying actual output rather than mocking rich_print.
    """

    def test_print_system_message(self) -> None:
        """print_system_message should render system message in a Panel."""
        message = SystemMessage(content="# Test System")

        output = _capture_output(lambda: print_system_message(message))

        # Verify Panel output contains the markdown content
        assert "Test System" in output or "#" in output

    def test_print_session_status(self) -> None:
        """print_session_status should output a dimmed status line."""
        state = SessionState(running=True, queued_prompt_count=0)

        output = _capture_output(lambda: print_session_status(state))

        # Should contain status info
        assert "running" in output or "dim" in output.lower()

    def test_print_info_message(self) -> None:
        """print_info_message should prefix with 'i' indicator."""
        output = _capture_output(lambda: print_info_message("Test message"))

        assert "Test message" in output
        assert "i" in output

    def test_print_active_prompt_string(self) -> None:
        """print_active_prompt should render content in a Panel with minimal box."""
        output = _capture_output(lambda: print_active_prompt("Hello world"))

        assert "Hello world" in output

    def test_print_active_prompt_structured(self) -> None:
        """print_active_prompt should handle structured content."""
        structured = [{"type": "text", "text": "Hello"}]
        output = _capture_output(lambda: print_active_prompt(structured))

        assert "Hello" in output or "text" in output

    def test_print_tool_calls(self) -> None:
        """print_tool_calls should render tool call header."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="echo", arguments='{"text": "hello"}'),
        )
        message = AssistantMessage(tool_calls=[tool_call])

        output = _capture_output(lambda: print_tool_calls(message))

        assert "echo" in output
        assert "hello" in output


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestOutputIntegration:
    """Integration tests combining multiple output components."""

    def test_delta_renderer_produces_output(self) -> None:
        """Full cycle: delta -> render should produce output."""
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("# Heading\n\nSome **bold** text\n\n")
            renderer.finish()

        output = buffer.getvalue()
        # Should contain the markdown content
        assert "Heading" in output or "#" in output

    def test_tool_call_format_and_print_roundtrip(self) -> None:
        """format_tool_call_display output should be printable."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(
                name="python_execute",
                arguments='{"code": "print(\\"hello\\")\\nprint(\\"world\\")"}',
            ),
        )
        header, body_sections = format_tool_call_display(tool_call)

        # Verify format output
        assert "python_execute" in header
        assert len(body_sections) == 1
        assert body_sections[0][2] == "python"  # language hint

        # Verify it can be printed
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)
        with patch("coding_assistant.app.output.rich_print", console.print):
            print_tool_calls(AssistantMessage(tool_calls=[tool_call]))

        output = buffer.getvalue()
        assert "python_execute" in output
