from __future__ import annotations
from rich.styled import Styled

import os
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, NotRequired, Optional, Union, TypedDict

from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding

from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks, StatusLevel
from coding_assistant.framework.results import TextResult, ToolResult
from coding_assistant.llm.types import UserMessage, AssistantMessage, ToolCall, ToolMessage

console = Console()
print = console.print

logger = logging.getLogger(__name__)


class SpecialToolParameterConfig(TypedDict):
    """Configuration for a single parameter's display behavior."""

    language: str  # Language hint for syntax highlighting (empty string = no highlighting)


class SpecialToolConfig(TypedDict):
    """Configuration for special tool display handling."""

    languages: dict[str, str]  # param_name -> language hint
    hide_value: NotRequired[list[str]]  # Parameters to hide from display


class ParagraphBuffer:
    def __init__(self) -> None:
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


async def confirm_tool_if_needed(
    *, tool_name: str, arguments: dict[str, Any], patterns: list[str], ui: Any
) -> Optional[TextResult]:
    for pat in patterns:
        if re.search(pat, tool_name):
            question = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Tool execution denied.")
            break
    return None


async def confirm_shell_if_needed(
    *, tool_name: str, arguments: dict[str, Any], patterns: list[str], ui: Any
) -> Optional[TextResult]:
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
    """Callbacks for displaying progress and tool calls with rich formatting."""

    _SPECIAL_TOOLS: dict[str, SpecialToolConfig] = {
        "shell_execute": {
            "languages": {"command": "bash"},
        },
        "python_execute": {
            "languages": {"code": "python"},
        },
        "filesystem_write_file": {
            "languages": {"content": ""},
        },
        "filesystem_edit_file": {
            "languages": {"old_text": "", "new_text": ""},
            "hide_value": ["old_text", "new_text"],
        },
        "todo_add": {
            "languages": {"descriptions": "json"},
        },
        "compact_conversation": {
            "languages": {"summary": "markdown"},
        },
    }

    def __init__(self, print_reasoning: bool = True):
        self._state: ProgressState = None
        self._left_padding = (0, 0, 0, 2)
        self._print_reasoning = print_reasoning

    def on_status_message(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        self._finalize_state()
        symbol = {
            StatusLevel.INFO: "ℹ️",
            StatusLevel.SUCCESS: "✅",
            StatusLevel.WARNING: "⚠️",
            StatusLevel.ERROR: "❌",
        }.get(level, "•")
        print(f"{symbol} {message}")

    def on_user_message(self, context_name: str, message: UserMessage, force: bool = False) -> None:
        if force:
            content = message.content if isinstance(message.content, str) else str(message.content)
            self._print_banner("User", content)

    def on_assistant_message(self, context_name: str, message: AssistantMessage, force: bool = False) -> None:
        if force:
            content = message.content if isinstance(message.content, str) else ""
            self._print_banner("Assistant", content)

    def _print_banner(self, role: str, content: str) -> None:
        self._finalize_state()
        print()
        print(Markdown(f"## {role}\n\n{content}"))
        print()
        self._state = IdleState()

    def _print_tool_start(self, symbol: str, tool_name: str, arguments: dict[str, Any]) -> None:
        config: SpecialToolConfig = self._SPECIAL_TOOLS.get(tool_name, {"languages": {}})
        hide_value_keys = config.get("hide_value", [])
        lang_hints = config.get("languages", {})

        header_params = []
        multi_line_params = []

        for key, value in arguments.items():
            if key in hide_value_keys:
                header_params.append(key)
                continue

            # Only treat as multi-line if explicitly configured in _SPECIAL_TOOLS
            if key in lang_hints:
                lang_hint = lang_hints[key]
                if isinstance(value, str):
                    formatted_value = value
                else:
                    formatted_value = json.dumps(value, indent=2)

                if "\n" in formatted_value:
                    multi_line_params.append((key, formatted_value, lang_hint))
                    header_params.append(key)
                    continue

            header_params.append(f"{key}={json.dumps(value)}")

        args_str = f"({', '.join(header_params)})"
        print(f"[bold yellow]{symbol}[/bold yellow] {tool_name}{args_str}")

        if multi_line_params:
            lang_override = self._get_lang_override(tool_name, arguments)

            for key, value, lang_hint in multi_line_params:
                lang = lang_override or lang_hint
                print()
                print(Padding(f"[dim]{key}:[/dim]", self._left_padding))
                if lang == "markdown":
                    print(Padding(Markdown(value), self._left_padding))
                else:
                    print(Padding(Markdown(f"````{lang}\n{value}\n````"), self._left_padding))
            print()

    def _get_lang_override(self, tool_name: str, arguments: dict[str, Any]) -> Optional[str]:
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

    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        self._finalize_state()
        print()
        self._print_tool_start("▶", tool_call.function.name, arguments)
        self._state = ToolState(tool_call_id=tool_call.id)

    def _special_handle_full_result(self, tool_name: str, result: str) -> bool:
        if tool_name == "filesystem_edit_file":
            print()
            print(Padding(Markdown(f"````diff\n{result.strip()}\n````"), self._left_padding))
            return True
        if tool_name.startswith("todo_"):
            print(Padding(Markdown(result.strip()), self._left_padding))
            return True
        return False

    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        if not isinstance(self._state, ToolState) or self._state.tool_call_id != message.tool_call_id:
            print()
            self._print_tool_start("◀", tool_name, arguments)

        if not self._special_handle_full_result(tool_name, message.content):
            print(f"  [dim]→ {len(message.content.splitlines())} lines[/dim]")

        self._state = ToolState()

    def on_reasoning_chunk(self, chunk: str) -> None:
        if self._print_reasoning:
            self._handle_chunk(chunk, ReasoningState, "dim cyan")

    def on_content_chunk(self, chunk: str) -> None:
        self._handle_chunk(chunk, ContentState)

    def _handle_chunk(
        self, chunk: str, state_class: type[Union[ContentState, ReasoningState]], style: str | None = None
    ) -> None:
        if not isinstance(self._state, state_class):
            self._finalize_state()
            print()
            self._state = state_class()

        assert isinstance(self._state, (ContentState, ReasoningState))
        for paragraph in self._state.buffer.push(chunk):
            print()
            md = Markdown(paragraph)
            print(Styled(md, style) if style else md)

    def _finalize_state(self) -> None:
        if isinstance(self._state, (ContentState, ReasoningState)):
            if flushed := self._state.buffer.flush():
                print()
                md = Markdown(flushed)
                style = "dim cyan" if isinstance(self._state, ReasoningState) else None
                print(Styled(md, style) if style else md)
            print()

    def on_chunks_end(self) -> None:
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
        arguments: dict[str, Any],
        *,
        ui: Any,
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
