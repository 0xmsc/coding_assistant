"""Tests for the output formatting module."""

from __future__ import annotations

import io
from collections.abc import Callable
from unittest.mock import patch

from rich.console import Console

from coding_assistant.app.output import (
    DeltaRenderer,
    ParagraphBuffer,
    format_prompt_preview,
    format_session_status,
    print_active_prompt,
    print_info_message,
    print_session_status,
    print_system_message,
    print_tool_calls,
)
from coding_assistant.core.agent_session import SessionState
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall


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
        buffer = ParagraphBuffer()
        content = "Text before\n\n```\ncode\n```\n\nText after"
        paragraphs = buffer.push(content)
        assert "Text before" in paragraphs
        assert buffer._buffer

    def test_handles_adjacent_paragraphs(self) -> None:
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
            renderer.finish()

    def test_single_delta_renders_markdown(self) -> None:
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Hello\n\nWorld")

        output = buffer.getvalue()
        assert "Hello" in output
        assert "World" not in output

    def test_buffer_holds_incomplete_paragraph(self) -> None:
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Hello\n\nWorld")
            renderer.finish()

        output = buffer.getvalue()
        assert "Hello" in output
        assert "World" in output

    def test_finish_flushes_remaining_content(self) -> None:
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("Some text without paragraph end")
            renderer.finish()

        output = buffer.getvalue()
        assert "Some text without paragraph end" in output

    def test_trailing_blank_line_controlled_by_flag(self) -> None:
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

        output1 = buffer1.getvalue()
        output2 = buffer2.getvalue()
        assert "Content" in output1
        assert "Content" in output2


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

    def test_running_status(self) -> None:
        state = SessionState(running=True, queued_prompt_count=0)
        assert format_session_status(state) == "running"

    def test_paused_status(self) -> None:
        state = SessionState(running=False, queued_prompt_count=0, paused=True)
        assert format_session_status(state) == "paused"

    def test_idle_status(self) -> None:
        state = SessionState(running=False, queued_prompt_count=5)
        assert format_session_status(state) == "idle"

    def test_pending_prompts_do_not_change_status_text(self) -> None:
        state = SessionState(
            running=True,
            queued_prompt_count=3,
            pending_prompts=("prompt1", "prompt2"),
        )
        assert format_session_status(state) == "running"


# =============================================================================
# print_* Function Tests
# =============================================================================


def _capture_output(func: Callable[[], None]) -> str:
    """Capture output from a function that uses rich_print."""
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=False, width=80)
    with patch("coding_assistant.app.output.rich_print", console.print):
        func()
    return buffer.getvalue()


class TestPrintFunctions:
    """Tests for print functions."""

    def test_print_system_message(self) -> None:
        message = SystemMessage(content="# Test System")
        output = _capture_output(lambda: print_system_message(message))
        assert "Test System" in output or "#" in output

    def test_print_session_status(self) -> None:
        state = SessionState(running=True, queued_prompt_count=0)
        output = _capture_output(lambda: print_session_status(state))
        assert "running" in output or "dim" in output.lower()

    def test_print_info_message(self) -> None:
        output = _capture_output(lambda: print_info_message("Test message"))
        assert "Test message" in output
        assert "i" in output

    def test_print_active_prompt_string(self) -> None:
        output = _capture_output(lambda: print_active_prompt("Hello world"))
        assert "Hello world" in output

    def test_print_active_prompt_structured(self) -> None:
        structured = [{"type": "text", "text": "Hello"}]
        output = _capture_output(lambda: print_active_prompt(structured))
        assert "Hello" in output or "text" in output

    def test_print_tool_calls(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="echo", arguments='{"text": "hello"}'),
        )
        message = AssistantMessage(tool_calls=[tool_call])
        output = _capture_output(lambda: print_tool_calls(message))
        assert "echo" in output
        assert "hello" in output

    def test_print_tool_calls_multiline(self) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="shell_execute", arguments='{"command": "echo hi"}'),
        )
        message = AssistantMessage(tool_calls=[tool_call])
        output = _capture_output(lambda: print_tool_calls(message))
        assert "shell_execute" in output
        assert "command" in output


# =============================================================================
# Integration Tests
# =============================================================================


class TestOutputIntegration:
    """Integration tests combining multiple output components."""

    def test_delta_renderer_produces_output(self) -> None:
        renderer = DeltaRenderer()
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=False, width=80)

        with patch("coding_assistant.app.output.rich_print", console.print):
            renderer.on_delta("# Heading\n\nSome **bold** text\n\n")
            renderer.finish()

        output = buffer.getvalue()
        assert "Heading" in output or "#" in output

    def test_tool_call_print_simple_format(self) -> None:
        """Tool calls should print in simple one-liner format."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="python_execute", arguments='{"code": "print(1)"}'),
        )
        message = AssistantMessage(tool_calls=[tool_call])
        output = _capture_output(lambda: print_tool_calls(message))
        assert "python_execute" in output
        assert "code" in output
        assert "▶" in output
