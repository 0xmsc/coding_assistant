from __future__ import annotations

import json
from typing import Any

from rich import print as rich_print
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.core.agent_session import SessionState
from coding_assistant.llm.types import AssistantMessage, SystemMessage, ToolCall


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

    def on_delta(self, chunk: str) -> None:
        """Render one streamed content chunk when a paragraph is complete."""
        for paragraph in self._buffer.push(chunk):
            self._print_markdown(paragraph)

    def finish(self) -> None:
        """Flush any remaining buffered content at the end of a turn."""
        if flushed := self._buffer.flush():
            self._print_markdown(flushed)

    def _print_markdown(self, content: str) -> None:
        rich_print()
        rich_print(Markdown(content))


def _format_tool_call(tool_call: ToolCall) -> str:
    """Format a tool call as a simple one-liner."""
    tool_name = tool_call.function.name or "<missing>"
    try:
        args = json.loads(tool_call.function.arguments)
        if isinstance(args, dict):
            params = ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items())
            return f"{tool_name}({params})"
    except json.JSONDecodeError:
        pass
    return f"{tool_name}({tool_call.function.arguments})"


def print_tool_calls(message: AssistantMessage) -> None:
    """Print tool calls in simple format."""
    rich_print()  # leading blank line
    for tool_call in message.tool_calls:
        rich_print(f"[bold yellow]▶[/bold yellow] {_format_tool_call(tool_call)}")


def print_system_message(message: SystemMessage) -> None:
    """Render the active system prompt at startup."""
    assert isinstance(message.content, str)
    rich_print(Panel(Markdown(message.content), title="System"))


def print_active_prompt(content: str | list[dict[str, Any]]) -> None:
    """Render one prompt when it becomes the active run."""
    if isinstance(content, str):
        rendered_content = content
    else:
        rendered_content = json.dumps(content, indent=2)

    lines = rendered_content.split("\n")
    rich_print()  # leading blank
    for line in lines:
        rich_print(f"  [bold cyan]▌[/bold cyan] [white]{line}[/white]")


def format_prompt_preview(content: str | list[dict[str, Any]]) -> str:
    """Return one compact prompt preview suitable for footer/status display."""
    if not isinstance(content, str):
        return "structured prompt"

    return " ".join(content.split())


def format_session_status(state: SessionState) -> str:
    """Return one compact session status line for the prompt footer or worker output."""
    if state.running:
        return "running"
    if state.paused:
        return "paused"
    return "idle"


def print_session_status(state: SessionState) -> None:
    """Render one compact status line."""
    rich_print(f"[dim]{format_session_status(state)}[/dim]")


def print_info_message(message: str) -> None:
    """Render one informational status line."""
    rich_print(f"[bold blue]i[/bold blue] {message}")
