from __future__ import annotations
from rich.styled import Styled

import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Union

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.results import TextResult, ToolResult

console = Console()
print = console.print

logger = logging.getLogger(__name__)


class ParagraphBuffer:
    def __init__(self):
        self._buffer = ""

    def _is_inside_code_fence(self, text: str) -> bool:
        return text.count("```") % 2 != 0

    def push(self, chunk: str) -> list[str]:
        self._buffer += chunk
        paragraphs = []

        search_from = 0
        while (pos := self._buffer.find("\n\n", search_from)) != -1:
            candidate = self._buffer[:pos]

            if not self._is_inside_code_fence(candidate):
                paragraphs.append(candidate)
                self._buffer = self._buffer[pos + 2 :]
                search_from = 0
            else:
                search_from = pos + 2

        return paragraphs

    def flush(self) -> Optional[str]:
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
    if tool_name != "shell_execute":
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
    _SPECIAL_TOOLS: dict[str, dict[str, str]] = {
        "shell_execute": {"command": "bash"},
        "python_execute": {"code": "python"},
        "filesystem_write_file": {"content": ""},
        "filesystem_edit_file": {"old_text": "", "new_text": ""},
        "todo_add": {"descriptions": "json"},
        "compact_conversation": {"summary": "markdown"},
    }

    def __init__(self, print_reasoning: bool = True):
        self._state: ProgressState = None
        self._left_padding = (0, 0, 0, 2)
        self._print_reasoning = print_reasoning

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

    def _print_tool_start(self, symbol: str, tool_name: str, arguments: dict):
        multiline_config = self._SPECIAL_TOOLS.get(tool_name, {})

        header_params = []
        multi_line_params = []

        for key, value in arguments.items():
            if key in multiline_config:
                if isinstance(value, str):
                    formatted_value = value
                else:
                    formatted_value = json.dumps(value, indent=2)

                if "\n" in formatted_value:
                    multi_line_params.append((key, formatted_value))
                    header_params.append(key)
                    continue

            header_params.append(f"{key}={json.dumps(value)}")

        args_str = f"({', '.join(header_params)})"
        print(f"[bold yellow]{symbol}[/bold yellow] {tool_name}{args_str}")

        if multi_line_params:
            lang_override = self._get_lang_override(tool_name, arguments)

            for key, value in multi_line_params:
                lang = lang_override or multiline_config[key]
                print()
                print(Padding(f"[dim]{key}:[/dim]", self._left_padding))
                if lang == "markdown":
                    print(Padding(Markdown(value), self._left_padding))
                else:
                    print(Padding(Markdown(f"````{lang}\n{value}\n````"), self._left_padding))
            print()

    def _get_lang_override(self, tool_name: str, arguments: dict) -> Optional[str]:
        file_tools = {
            "filesystem_write_file",
            "filesystem_edit_file",
        }
        if tool_name in file_tools and "path" in arguments and isinstance(arguments["path"], str):
            path = arguments["path"]
            basename = os.path.basename(path)
            _, ext = os.path.splitext(basename)
            if ext:
                return ext[1:]
        return None

    def on_tool_start(self, context_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        self._finalize_state()
        print()
        self._print_tool_start("▶", tool_name, arguments)
        self._state = ToolState(tool_call_id=tool_call_id)

    def _special_handle_full_result(self, tool_name: str, result: str) -> bool:
        if tool_name == "filesystem_edit_file":
            print()
            print(Padding(Markdown(f"````diff\n{result.strip()}\n````"), self._left_padding))
            return True
        if tool_name.startswith("todo_"):
            print(Padding(Markdown(result.strip()), self._left_padding))
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
        if self._print_reasoning:
            self._handle_chunk(chunk, ReasoningState, "dim cyan")

    def on_content_chunk(self, chunk: str):
        self._handle_chunk(chunk, ContentState)

    def _handle_chunk(
        self, chunk: str, state_class: type[Union[ContentState, ReasoningState]], style: str | None = None
    ):
        if not isinstance(self._state, state_class):
            self._finalize_state()
            print()
            self._state = state_class()

        assert isinstance(self._state, (ContentState, ReasoningState))
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
