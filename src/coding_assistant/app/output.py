from __future__ import annotations

import json
import os
from typing import Any

from rich import print as rich_print
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.styled import Styled

from coding_assistant.core.agent_session import (
    AgentSession,
    PromptAcceptedEvent,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import AssistantMessage, SystemMessage, ToolCall
from coding_assistant.llm.types import CompletionEvent, ContentDeltaEvent, ReasoningDeltaEvent, StatusEvent


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
    """Render one prompt when it becomes the active run."""
    renderable: Markdown
    if isinstance(content, str):
        renderable = Markdown(content)
    else:
        renderable = Markdown("```json\n" + json.dumps(content, indent=2) + "\n```")

    rich_print(Styled(renderable, "on grey11"))


def format_prompt_preview(content: str | list[dict[str, Any]], *, limit: int = 40) -> str:
    """Return one compact prompt preview suitable for footer/status display."""
    if not isinstance(content, str):
        return "structured prompt"

    preview = " ".join(content.split())
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 3]}..."


def format_session_status(state: SessionState) -> str:
    """Return one compact session status line for the prompt footer or worker output."""
    status = "running" if state.running else "idle"
    parts = [status, f"queued: {state.queued_prompt_count}"]
    if state.pending_prompts:
        previews = [format_prompt_preview(prompt) for prompt in state.pending_prompts[:2]]
        parts.append(f"next: {previews[0]}")
        if len(previews) > 1:
            parts.append(f"then: {previews[1]}")
        remaining_count = state.queued_prompt_count - len(previews)
        if remaining_count > 0:
            parts.append(f"+{remaining_count} more")
    return " | ".join(parts)


def print_session_status(state: SessionState) -> None:
    """Render one compact status line."""
    rich_print(f"[dim]{format_session_status(state)}[/dim]")


async def run_session_output(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    show_state_updates: bool = False,
    show_prompt_accepted: bool = False,
) -> None:
    """Render one session's streamed events to the local terminal."""
    renderer = DeltaRenderer()
    last_state_summary: str | None = None
    print_system_message(system_message)

    async with session.subscribe() as queue:
        try:
            while True:
                event = await queue.get()
                if isinstance(event, ContentDeltaEvent):
                    renderer.on_delta(event.content)
                    continue
                if isinstance(event, PromptAcceptedEvent):
                    if not show_prompt_accepted:
                        continue
                    renderer.finish()
                    print_active_prompt(event.content)
                    continue
                if isinstance(event, PromptStartedEvent):
                    renderer.finish()
                    print_active_prompt(event.content)
                    continue
                if isinstance(event, ToolCallsEvent):
                    renderer.finish(trailing_blank_line=False)
                    print_tool_calls(event.message)
                    continue
                if isinstance(event, RunFinishedEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunCancelledEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunFailedEvent):
                    renderer.finish()
                    rich_print(f"[bold red]Run failed:[/bold red] {event.error}")
                    continue
                if isinstance(event, StateChangedEvent):
                    if not show_state_updates:
                        continue
                    state_summary = format_session_status(event.state)
                    if state_summary == last_state_summary:
                        continue
                    last_state_summary = state_summary
                    renderer.finish(trailing_blank_line=False)
                    print_session_status(event.state)
                    continue
                if isinstance(event, (ReasoningDeltaEvent, StatusEvent, CompletionEvent)):
                    continue
        finally:
            renderer.finish()


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
