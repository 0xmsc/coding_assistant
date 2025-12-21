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

    # Configuration for special handling of multiline arguments
    _SPECIAL_TOOLS = {
        "mcp_coding_assistant_mcp_shell_execute": {"lang_map": {"command": "bash"}, "order": ["command"]},
        "mcp_coding_assistant_mcp_python_execute": {"lang_map": {"code": "python"}, "order": ["code"]},
        "mcp_coding_assistant_mcp_filesystem_write_file": {"order": ["path", "content"]},
        "mcp_coding_assistant_mcp_filesystem_edit_file": {"order": ["path", "old_text", "new_text"]},
    }

    def __init__(self):
        self._state: ProgressState = None
        self._left_padding = (0, 0, 0, 2)

    def on_user_message(self, context_name: str, content: str, force: bool = False):
        if force:
            self._print_banner("User", content)

    def on_assistant_message(self, context_name: str, content: str, force: bool = False):
        if force:
            self._print_banner("Assistant", content)

    def _print_banner(self, role: str, content: str):
        self._finalize_state()
        print()
        print(Markdown(f"## {role}\n\n{content}"))
        self._state = IdleState()

    def on_assistant_reasoning(self, context_name: str, content: str):
        pass

    def _print_arguments_multiline(
        self,
        symbol: str,
        tool_name: str,
        arguments: dict,
        lang_map: dict[str, str] = dict(),
        order: list[str] | None = None,
    ):
        print(f"[bold yellow]{symbol}[/bold yellow] {tool_name}")

        keys = list(arguments.keys())
        if order:
            keys.sort(key=lambda k: order.index(k) if k in order else len(order))

        for key in keys:
            value = arguments[key]
            if isinstance(value, str) and "\n" in value:
                lang = lang_map.get(key, "")
                print()
                print(Padding(f"[dim]{key}:[/dim]", self._left_padding))
                print(Padding(Markdown(f"```{lang}\n{value}\n```"), self._left_padding))
            else:
                print(Padding(f"[dim]{key}:[/dim] {json.dumps(value)}", self._left_padding))
        print()

    def _special_handle_arguments(self, symbol: str, tool_name: str, arguments: dict) -> bool:
        config = self._SPECIAL_TOOLS.get(tool_name)
        if not config:
            return False

        if any("\n" in str(v) for v in arguments.values()):
            self._print_arguments_multiline(
                symbol, tool_name, arguments, lang_map=config.get("lang_map", {}), order=config.get("order")
            )
            return True

        return False

    def _print_tool_start(self, symbol: str, tool_name: str, arguments: dict):
        if not self._special_handle_arguments(symbol, tool_name, arguments):
            formatted = ", ".join(f"{k}={json.dumps(v)}" for k, v in arguments.items())
            args_str = f"({formatted})" if formatted else ""
            print(f"[bold yellow]{symbol}[/bold yellow] {tool_name}{args_str}")

    def on_tool_start(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        self._finalize_state()
        print()
        self._print_tool_start("▶", tool_name, arguments)
        self._state = ToolState(tool_call_id=tool_call_id)

    def _special_handle_full_result(self, tool_name: str, result: str) -> bool:
        if tool_name == "mcp_coding_assistant_mcp_filesystem_edit_file":
            print()
            print(Padding(Markdown(f"```diff\n{result.strip('\\n')}\n```"), self._left_padding))
            return True
        if tool_name.startswith("mcp_coding_assistant_mcp_todo_"):
            print(Padding(Markdown(result), self._left_padding))
            return True
        return False

    def on_tool_message(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict, result: str):
        if not isinstance(self._state, ToolState) or self._state.tool_call_id != tool_call_id:
            print()
            self._print_tool_start("◀", tool_name, arguments)

        if not self._special_handle_full_result(tool_name, result):
            print(f"  [dim]→ {len(result.splitlines())} lines[/dim]")

        self._state = ToolState()

    def on_reasoning_chunk(self, chunk: str):
        self._handle_chunk(chunk, ReasoningState, "dim cyan")

    def on_content_chunk(self, chunk: str):
        self._handle_chunk(chunk, ContentState)

    def _handle_chunk(self, chunk: str, state_class: type, style: str | None = None):
        if not isinstance(self._state, state_class):
            self._finalize_state()
            print()
            self._state = state_class()

        for paragraph in self._state.buffer.push(chunk):
            print()
            md = Markdown(paragraph)
            print(Styled(md, style) if style else md)

    def _finalize_state(self):
        if isinstance(self._state, (ContentState, ReasoningState)):
            if flushed := self._state.buffer.flush():
                print()
                md = Markdown(flushed)
                style = "dim cyan" if isinstance(self._state, ReasoningState) else None
                print(Styled(md, style) if style else md)
            elif isinstance(self._state, ReasoningState):
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
