from __future__ import annotations

import json
from typing import Any, Literal

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

    def __init__(self, *, style: str | None = None) -> None:
        self._buffer = ParagraphBuffer()
        self._style = style

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
        rich_print(Markdown(content, style=self._style or "none"))


class StreamRenderer:
    """Render assistant content and reasoning streams without interleaving buffers."""

    def __init__(self) -> None:
        self._content_renderer = DeltaRenderer()
        self._reasoning_renderer = DeltaRenderer(style="dim")
        self._active_stream: Literal["content", "reasoning"] | None = None

    def on_content_delta(self, chunk: str) -> None:
        """Render one assistant content chunk."""
        self._switch_stream("content")
        self._content_renderer.on_delta(chunk)

    def on_reasoning_delta(self, chunk: str) -> None:
        """Render one assistant reasoning chunk."""
        self._switch_stream("reasoning")
        self._reasoning_renderer.on_delta(chunk)

    def finish(self) -> None:
        """Flush any buffered content or reasoning output."""
        self._content_renderer.finish()
        self._reasoning_renderer.finish()
        self._active_stream = None

    def _switch_stream(self, stream: Literal["content", "reasoning"]) -> None:
        if self._active_stream == stream:
            return
        self._finish_active_stream()
        self._active_stream = stream

    def _finish_active_stream(self) -> None:
        if self._active_stream == "content":
            self._content_renderer.finish()
        elif self._active_stream == "reasoning":
            self._reasoning_renderer.finish()


# Maximum length for a single argument value before truncation
_MAX_ARG_VALUE_LENGTH = 50


def _truncate_value(value: str) -> str:
    """Truncate a value string if it exceeds the maximum length."""
    if len(value) <= _MAX_ARG_VALUE_LENGTH:
        return value
    return value[:_MAX_ARG_VALUE_LENGTH] + "[...]"


def _format_tool_call(tool_call: ToolCall) -> str:
    """Format a tool call as a simple one-liner with truncated long arguments."""
    tool_name = tool_call.function.name or "<missing>"
    try:
        args = json.loads(tool_call.function.arguments)
        if isinstance(args, dict):
            params = ", ".join(f"{k}={json.dumps(_truncate_value(str(v)))}" for k, v in args.items())
            return f"{tool_name}({params})"
    except json.JSONDecodeError:
        pass
    return f"{tool_name}({tool_call.function.arguments})"


def print_tool_calls(message: AssistantMessage) -> None:
    """Print tool calls in simple format."""
    for tool_call in message.tool_calls:
        rich_print()
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


def print_info_message(message: str) -> None:
    """Render one informational status line."""
    rich_print(f"[bold blue]ℹ[/bold blue] {message}")
