"""Tests for the output formatting module.

This module tests all functions in coding_assistant.app.output, including:
- ParagraphBuffer: streaming content buffering logic
- DeltaRenderer: markdown rendering with buffering
- format_* functions: pure formatting logic
- print_* functions: output with rich (patched in tests)
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from rich.markdown import Markdown
from rich.panel import Panel

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

    @pytest.fixture
    def mock_rich_print(self) -> Generator[MagicMock, None, None]:
        """Fixture that patches rich_print for each test."""
        with patch("coding_assistant.app.output.rich_print") as mock:
            yield mock

    def test_empty_renderer_finishes_cleanly(self, mock_rich_print: MagicMock) -> None:
        renderer = DeltaRenderer()
        renderer.finish()
        mock_rich_print.assert_not_called()

    def test_single_delta_renders_markdown(self, mock_rich_print: MagicMock) -> None:
        renderer = DeltaRenderer()
        renderer.on_delta("Hello\n\nWorld")
        # Two paragraphs should produce two rich_print calls
        assert mock_rich_print.call_count == 2
        # Verify Markdown was created
        args = mock_rich_print.call_args_list
        assert all(isinstance(arg[0][0], Markdown) for arg in args if arg[0])

    def test_finish_flushes_remaining_content(self, mock_rich_print: MagicMock) -> None:
        renderer = DeltaRenderer()
        renderer.on_delta("Some text without paragraph end")
        mock_rich_print.reset_mock()
        renderer.finish()
        # finish() prints: blank line + content + blank line = 3 calls
        assert mock_rich_print.call_count == 3
        # Second call should be the Markdown content
        assert isinstance(mock_rich_print.call_args_list[1][0][0], Markdown)

    def test_trailing_blank_line_controlled_by_flag(self, mock_rich_print: MagicMock) -> None:
        renderer = DeltaRenderer()
        renderer.on_delta("Content\n\nMore")
        mock_rich_print.reset_mock()
        renderer.finish(trailing_blank_line=False)
        # Only the two paragraphs, no trailing blank line = 2 calls
        assert mock_rich_print.call_count == 2

    def test_multiple_deltas_accumulate(self, mock_rich_print: MagicMock) -> None:
        renderer = DeltaRenderer()
        renderer.on_delta("First para")
        renderer.on_delta("\n\nSecond para")
        mock_rich_print.reset_mock()
        renderer.finish()
        # Should have printed both paragraphs plus trailing blank line = 3 calls
        assert mock_rich_print.call_count == 3


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
# print_* Function Tests (patching rich_print)
# =============================================================================


class TestPrintFunctions:
    """Tests for print functions that output to terminal."""

    @pytest.fixture
    def mock_rich_print(self) -> Generator[MagicMock, None, None]:
        """Fixture that patches rich_print for each test."""
        with patch("coding_assistant.app.output.rich_print") as mock:
            yield mock

    def test_print_system_message(self, mock_rich_print: MagicMock) -> None:
        message = SystemMessage(content="# Test System")
        print_system_message(message)
        mock_rich_print.assert_called_once()
        # Verify Panel was created with Markdown
        call_args = mock_rich_print.call_args[0][0]
        assert isinstance(call_args, Panel)

    def test_print_session_status(self, mock_rich_print: MagicMock) -> None:
        state = SessionState(running=True, queued_prompt_count=0)
        print_session_status(state)
        mock_rich_print.assert_called_once()
        call_args = mock_rich_print.call_args[0][0]
        assert "[dim]" in call_args

    def test_print_info_message(self, mock_rich_print: MagicMock) -> None:
        print_info_message("Test message")
        mock_rich_print.assert_called_once()
        call_args = mock_rich_print.call_args[0][0]
        assert "[bold blue]i[/bold blue]" in call_args

    def test_print_active_prompt_string(self, mock_rich_print: MagicMock) -> None:
        print_active_prompt("Hello world")
        # Function prints: blank line + Panel + blank line = 3 calls
        assert mock_rich_print.call_count == 3
        # Second call should be a Panel
        assert isinstance(mock_rich_print.call_args_list[1][0][0], Panel)

    def test_print_active_prompt_structured(self, mock_rich_print: MagicMock) -> None:
        structured = [{"type": "text", "text": "Hello"}]
        print_active_prompt(structured)
        # Function prints: blank line + Panel + blank line = 3 calls
        assert mock_rich_print.call_count == 3

    def test_print_tool_calls(self, mock_rich_print: MagicMock) -> None:
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(name="echo", arguments='{"text": "hello"}'),
        )
        message = AssistantMessage(tool_calls=[tool_call])
        print_tool_calls(message)
        # Should print header + body sections
        assert mock_rich_print.call_count >= 2


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestOutputIntegration:
    """Integration tests combining multiple output components."""

    @pytest.fixture
    def mock_rich_print(self) -> Generator[MagicMock, None, None]:
        """Fixture that patches rich_print for each test."""
        with patch("coding_assistant.app.output.rich_print") as mock:
            yield mock

    def test_delta_renderer_produces_markdown_output(self, mock_rich_print: MagicMock) -> None:
        """Full cycle: delta -> markdown render."""
        renderer = DeltaRenderer()
        renderer.on_delta("# Heading\n\nSome **bold** text\n\n")
        mock_rich_print.assert_called()
        # Check that markdown was rendered (just verify it's called with Markdown objects)
        calls = mock_rich_print.call_args_list
        assert any(isinstance(call[0][0], Markdown) for call in calls if call[0])

    def test_tool_call_to_print_roundtrip(self, mock_rich_print: MagicMock) -> None:
        """Test that format_tool_call_display output can be printed."""
        tool_call = ToolCall(
            id="1",
            function=FunctionCall(
                name="python_execute",
                arguments='{"code": "print(\\"hello\\")\\nprint(\\"world\\")"}',
            ),
        )
        header, body_sections = format_tool_call_display(tool_call)
        # This would be printed by _print_tool_call
        assert "python_execute" in header
        assert len(body_sections) == 1
        assert body_sections[0][2] == "python"  # language hint
