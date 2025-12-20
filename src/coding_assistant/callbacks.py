from __future__ import annotations
from rich.styled import Styled

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from rich import print
from rich.markdown import Markdown
from rich.padding import Padding

from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.results import TextResult, ToolResult

logger = logging.getLogger(__name__)


class ParagraphBuffer:
    """Buffers text and returns full paragraphs (separated by double newlines).

    Handles code fences (```) by buffering the entire fence before splitting.
    """

    def __init__(self):
        self._buffer = ""

    def _is_inside_code_fence(self, text: str) -> bool:
        return text.count("```") % 2 != 0

    def push(self, chunk: str) -> list[str]:
        """Push a chunk of text and return any complete paragraphs found."""
        self._buffer += chunk

        # If we are inside a code fence, we don't split yet
        if self._is_inside_code_fence(self._buffer):
            return []

        parts = self._buffer.split("\n\n")
        if len(parts) > 1:
            # We need to make sure none of the parts (except maybe the last one which stays in buffer)
            # start a code fence that isn't closed within that part.
            # This is complex if a code fence spans multiple paragraphs.
            # Let's refine: find double newlines only outside of code fences.

            paragraphs = []
            current_temp = ""
            remaining = self._buffer

            while "\n\n" in remaining:
                prefix, suffix = remaining.split("\n\n", 1)
                current_temp += prefix
                if self._is_inside_code_fence(current_temp):
                    # The double newline is inside a code fence, keep it
                    current_temp += "\n\n"
                    remaining = suffix
                else:
                    # Found a real paragraph boundary
                    paragraphs.append(current_temp)
                    current_temp = ""
                    remaining = suffix

            self._buffer = current_temp + remaining
            return paragraphs

        return []

    def flush(self) -> Optional[str]:
        """Flush the remaining buffer and return it as a paragraph if not empty."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None


@dataclass
class ReasoningState:
    buffer: ParagraphBuffer = field(default_factory=ParagraphBuffer)


@dataclass
class ContentState:
    buffer: ParagraphBuffer = field(default_factory=ParagraphBuffer)


@dataclass
class ToolState:
    tool_call_id: str | None = None


@dataclass
class IdleState:
    pass


ProgressState = Union[ReasoningState, ContentState, ToolState, IdleState, None]


async def confirm_tool_if_needed(*, tool_name: str, arguments: dict, patterns: list[str], ui) -> Optional[TextResult]:
    for pat in patterns:
        if re.search(pat, tool_name):
            question = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Tool execution denied.")
            break
    return None


async def confirm_shell_if_needed(*, tool_name: str, arguments: dict, patterns: list[str], ui) -> Optional[TextResult]:
    if tool_name != "mcp_coding_assistant_mcp_shell_execute":
        return None

    command = arguments.get("command")
    if not isinstance(command, str):
        return None

    for pat in patterns:
        if re.search(pat, command):
            question = f"Execute shell command `{command}` for tool `{tool_name}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Shell command execution denied.")
            break
    return None


class DenseProgressCallbacks(ProgressCallbacks):
    """Dense progress callbacks with minimal formatting."""

    def __init__(self):
        self._state: ProgressState = None

    def on_user_message(self, context_name: str, content: str, force: bool = False):
        if not force:
            return

        self._finalize_state()
        print()
        print(Markdown(f"## User\n\n{content}"))
        self._state = IdleState()

    def on_assistant_message(self, context_name: str, content: str, force: bool = False):
        if not force:
            return

        self._finalize_state()
        print()
        print(Markdown(f"## Assistant\n\n{content}"))
        self._state = IdleState()

    def on_assistant_reasoning(self, context_name: str, content: str):
        # Don't print - reasoning is already printed via chunks
        pass

    def _print_tool_start(self, symbol, tool_name: str, arguments: dict):
        args_str = self._format_arguments(arguments)
        print(f"[bold yellow]{symbol}[/bold yellow] {tool_name}{args_str}")

    def on_tool_start(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        self._finalize_state()
        print()
        self._print_tool_start("▶", tool_name, arguments)
        self._state = ToolState(tool_call_id=tool_call_id)

    def _special_handle_full_result(self, tool_call_id: str, tool_name: str, result: str) -> bool:
        left_padding = (0, 0, 0, 1)

        if tool_name == "mcp_coding_assistant_mcp_filesystem_edit_file":
            diff_body = result.strip("\n")
            rendered_result = Markdown(f"```diff\n{diff_body}\n```")
            print()
            print(Padding(rendered_result, left_padding))
            return True
        elif tool_name.startswith("mcp_coding_assistant_mcp_todo_"):
            print(Padding(Markdown(result), left_padding))
            return True

        return False

    def _format_arguments(self, arguments: dict) -> str:
        if not arguments:
            return ""

        formatted = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
        return f"({formatted})"

    def on_tool_message(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict, result: str):
        if not isinstance(self._state, ToolState) or self._state.tool_call_id != tool_call_id:
            print()
            self._print_tool_start("◀", tool_name, arguments)

        if not self._special_handle_full_result(tool_call_id, tool_name, result):
            print(f"  [dim]→ {len(result.splitlines())} lines[/dim]")

        # Reset state
        self._state = ToolState()

    def on_reasoning_chunk(self, chunk: str):
        if not isinstance(self._state, ReasoningState):
            self._finalize_state()
            print()
            self._state = ReasoningState()

        for paragraph in self._state.buffer.push(chunk):
            print()
            print(Styled(Markdown(paragraph), "dim cyan"))

    def on_content_chunk(self, chunk: str):
        if not isinstance(self._state, ContentState):
            self._finalize_state()
            print()
            self._state = ContentState()

        for paragraph in self._state.buffer.push(chunk):
            print()
            print(Markdown(paragraph))

    def _finalize_state(self):
        if isinstance(self._state, ContentState):
            if flushed := self._state.buffer.flush():
                print()
                print(Markdown(flushed))
        elif isinstance(self._state, ReasoningState):
            if flushed := self._state.buffer.flush():
                print()
                print(Styled(Markdown(flushed), "dim cyan"))
            else:
                print()

    def on_chunks_end(self):
        self._finalize_state()
        self._state = IdleState()


class ConfirmationToolCallbacks(ToolCallbacks):
    def __init__(
        self,
        *,
        tool_confirmation_patterns: list[str] | None = None,
        shell_confirmation_patterns: list[str] | None = None,
    ):
        self._tool_patterns = tool_confirmation_patterns or []
        self._shell_patterns = shell_confirmation_patterns or []

    async def before_tool_execution(
        self,
        context_name: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        *,
        ui,
    ) -> Optional[ToolResult]:
        if result := await confirm_tool_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._tool_patterns,
            ui=ui,
        ):
            return result

        if result := await confirm_shell_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._shell_patterns,
            ui=ui,
        ):
            return result

        return None
