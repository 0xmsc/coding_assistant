from __future__ import annotations

import json
import os
from typing import Any

from rich import print as rich_print
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

from coding_assistant.core.agent_session import SessionState
from coding_assistant.llm.types import AssistantMessage, SystemMessage, ToolCall


SPECIAL_TOOL_FORMATS: dict[str, dict[str, Any]] = {
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
        "hide_value": ["old_text", "new_text"],
    },
    "todo_add": {
        "languages": {"descriptions": "json"},
    },
    "compact_conversation": {
        "languages": {"summary": "markdown"},
    },
}


class ParagraphBuffer:
    """Buffer streamed content until paragraph boundaries are safe to render."""

    def __init__(self) -> None:
        self._buffer = ""

    def _is_inside_code_fence(self, text: str) -> bool:
        return text.count("```") % 2 != 0

    def push(self, chunk: str) -> list[str]:
        self._buffer += chunk
        paragraphs: list[str] = []

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

    def flush(self) -> str | None:
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining if remaining else None


class DeltaRenderer:
    """Render streamed assistant text as markdown paragraphs."""

    def __init__(self) -> None:
        self._buffer = ParagraphBuffer()
        self.saw_content = False

    def on_delta(self, chunk: str) -> None:
        """Render one streamed content chunk when a paragraph is complete."""
        self.saw_content = True
        for paragraph in self._buffer.push(chunk):
            self._print_markdown(paragraph)

    def finish(self, *, trailing_blank_line: bool = True) -> None:
        """Flush any remaining buffered content at the end of a turn."""
        had_content = self.saw_content
        if flushed := self._buffer.flush():
            self._print_markdown(flushed)
        if had_content and trailing_blank_line:
            rich_print()
        self.saw_content = False

    def _print_markdown(self, content: str) -> None:
        rich_print()
        rich_print(Markdown(content))


def _parse_tool_arguments_for_display(arguments: str) -> dict[str, Any] | None:
    """Best-effort JSON decoding for tool-call display."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _print_tool_call(tool_call: ToolCall) -> None:
    """Print one tool call with the old dense-progress layout."""
    header, body_sections = format_tool_call_display(tool_call)
    left_padding = (0, 0, 0, 2)

    rich_print()
    rich_print(f"[bold yellow]▶[/bold yellow] {header}")

    for key, value, language in body_sections:
        rich_print()
        rich_print(Padding(f"[dim]{key}:[/dim]", left_padding))
        if language == "markdown":
            rich_print(Padding(Markdown(value), left_padding))
        else:
            fence = f"````{language}\n{value}\n````" if language else f"````\n{value}\n````"
            rich_print(Padding(Markdown(fence), left_padding))

    if body_sections:
        rich_print()


def _get_tool_language_override(tool_name: str, arguments: dict[str, Any]) -> str | None:
    """Infer a code-fence language from file extensions when helpful."""
    if tool_name not in {"filesystem_write_file", "filesystem_edit_file"}:
        return None

    path = arguments.get("path")
    if not isinstance(path, str):
        return None

    basename = os.path.basename(path)
    _, extension = os.path.splitext(basename)
    if not extension:
        return None
    return extension[1:]


def print_system_message(message: SystemMessage) -> None:
    """Render the active system prompt at startup."""
    assert isinstance(message.content, str)
    rich_print(Panel(Markdown(message.content), title="System"))


def print_tool_calls(message: AssistantMessage) -> None:
    """Render tool calls in the old dense-progress style."""
    for tool_call in message.tool_calls:
        _print_tool_call(tool_call)


def print_active_prompt(content: str | list[dict[str, Any]]) -> None:
    """Render one prompt when it becomes the active run - simple and clean."""
    if isinstance(content, str):
        rendered_content = content
    else:
        rendered_content = json.dumps(content, indent=2)

    lines = rendered_content.split("\n")
    rich_print()
    if len(lines) == 1:
        rich_print(f"  [bold cyan]▌[/bold cyan] [white]{lines[0]}[/white]")
    else:
        rich_print(f"  [bold cyan]▌[/bold cyan] [white]{lines[0]}[/white]")
        for line in lines[1:]:
            rich_print(f"    [dim]│[/dim] [white]{line}[/white]")
    rich_print()


def format_prompt_preview(content: str | list[dict[str, Any]]) -> str:
    """Return one compact prompt preview suitable for footer/status display."""
    if not isinstance(content, str):
        return "structured prompt"

    return " ".join(content.split())


def format_session_status(state: SessionState) -> str:
    """Return one compact session status line for the prompt footer or worker output."""
    if state.running:
        status = "running"
    elif state.paused:
        status = "paused"
    else:
        status = "idle"
    if state.pending_prompts:
        # When pending prompts exist, they're shown in the queued prompts widget above the input.
        # Only show the status in the footer to avoid redundancy.
        return status
    parts = [status, f"queued: {state.queued_prompt_count}"]
    return " | ".join(parts)


def print_session_status(state: SessionState) -> None:
    """Render one compact status line."""
    rich_print(f"[dim]{format_session_status(state)}[/dim]")


def print_info_message(message: str) -> None:
    """Render one informational status line."""
    rich_print(f"[bold blue]i[/bold blue] {message}")


def format_tool_call_display(tool_call: ToolCall) -> tuple[str, list[tuple[str, str, str]]]:
    """Return the tool-call header and multiline sections for display."""
    tool_name = tool_call.function.name or "<missing>"
    parsed_arguments = _parse_tool_arguments_for_display(tool_call.function.arguments)
    if parsed_arguments is None:
        return f"{tool_name}(arguments)", [("arguments", tool_call.function.arguments, "")]

    config = SPECIAL_TOOL_FORMATS.get(tool_name, {})
    hide_value_keys = set(config.get("hide_value", []))
    language_hints: dict[str, str] = dict(config.get("languages", {}))
    header_params: list[str] = []
    body_sections: list[tuple[str, str, str]] = []
    language_override = _get_tool_language_override(tool_name, parsed_arguments)

    for key, value in parsed_arguments.items():
        if key in hide_value_keys:
            header_params.append(key)
            continue

        if key in language_hints:
            formatted_value = value if isinstance(value, str) else json.dumps(value, indent=2)
            if "\n" in formatted_value:
                header_params.append(key)
                body_sections.append((key, formatted_value, language_override or language_hints[key]))
                continue

        header_params.append(f"{key}={json.dumps(value)}")

    args_suffix = f"({', '.join(header_params)})"
    return f"{tool_name}{args_suffix}", body_sections
