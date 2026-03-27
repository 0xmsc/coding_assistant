from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from rich import print as rich_print
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

from coding_assistant.app.image import get_image
from coding_assistant.app.instructions import get_instructions
from coding_assistant.app.sandbox import sandbox
from coding_assistant.app.ui import DefaultAnswerUI, PromptToolkitUI, UI
from coding_assistant.core.agent import AwaitingTools, execute_tool_calls, run_agent_until_boundary
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    SystemMessage,
    Tool,
    UserMessage,
)
from coding_assistant.mcp import __name__ as mcp_package_name
from coding_assistant.integrations.mcp_client import (
    MCPServer,
    MCPServerConfig,
    get_mcp_servers_from_config,
    get_mcp_wrapped_tools,
    print_mcp_tools,
)


@dataclass(slots=True)
class DefaultAgentConfig:
    """Configuration for the default CLI and embedding setup."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    mcp_env: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str
    mcp_servers: list[MCPServer]


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

    def finish(self) -> None:
        """Flush any remaining buffered content at the end of a turn."""
        if flushed := self._buffer.flush():
            self._print_markdown(flushed)
        if self.saw_content:
            rich_print()

    def _print_markdown(self, content: str) -> None:
        rich_print()
        rich_print(Markdown(content))


def build_default_agent_config(args: Namespace) -> DefaultAgentConfig:
    """Translate CLI arguments into the default agent configuration."""
    working_directory = Path(os.getcwd())
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return DefaultAgentConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        mcp_env=tuple(args.mcp_env),
        user_instructions=tuple(args.instructions),
    )


def _get_default_mcp_server_config(
    skills_directories: tuple[str, ...],
    env: tuple[str, ...] = (),
) -> MCPServerConfig:
    """Build the bundled MCP server config used by default runs."""
    args = ["-m", mcp_package_name]
    if skills_directories:
        args.append("--skills-directories")
        args.extend(skills_directories)

    return MCPServerConfig(
        name="coding_assistant.mcp",
        command=sys.executable,
        args=args,
        env=list(env),
    )


@asynccontextmanager
async def create_default_agent(
    *,
    config: DefaultAgentConfig,
) -> AsyncIterator[DefaultAgentBundle]:
    """Resolve instructions and tools for a default agent run."""
    server_configs = (
        *config.mcp_server_configs,
        _get_default_mcp_server_config(config.skills_directories, config.mcp_env),
    )

    async with get_mcp_servers_from_config(list(server_configs), working_directory=config.working_directory) as servers:
        tools = await get_mcp_wrapped_tools(servers)
        instructions = get_instructions(
            working_directory=config.working_directory,
            user_instructions=list(config.user_instructions),
            mcp_servers=servers,
        )
        yield DefaultAgentBundle(
            tools=tools,
            instructions=instructions,
            mcp_servers=servers,
        )


async def run_cli(args: Namespace) -> None:
    """Run the interactive or single-shot CLI entry point."""
    config = build_default_agent_config(args)

    if args.sandbox:
        _apply_sandbox(args=args, config=config)

    ui: UI
    if args.task is None or args.ask_user:
        ui = PromptToolkitUI()
    else:
        ui = DefaultAnswerUI()

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = _build_initial_system_message(instructions=bundle.instructions)
        _print_system_message(system_message)
        current_history: list[BaseMessage] = [system_message]
        if args.task is not None:
            current_history.append(UserMessage(content=args.task))

        await _drive_agent(
            history=current_history,
            model=args.model,
            tools=bundle.tools,
            ui=ui,
            interactive=args.task is None or args.ask_user,
        )


def _apply_sandbox(*, args: Namespace, config: DefaultAgentConfig) -> None:
    """Apply the CLI sandbox policy before starting the agent."""
    coding_assistant_root = Path(__file__).resolve().parent.parent
    readable = [
        *[Path(path).resolve() for path in args.readable_sandbox_directories],
        *[Path(path).resolve() for path in config.skills_directories],
        coding_assistant_root,
    ]
    writable = [*([Path(path).resolve() for path in args.writable_sandbox_directories]), config.working_directory]
    sandbox(readable_paths=readable, writable_paths=writable, include_defaults=True)


def _build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=build_system_prompt(instructions=instructions))


def _print_system_message(message: SystemMessage) -> None:
    """Render the active system prompt at startup."""
    assert isinstance(message.content, str)
    rich_print(Panel(Markdown(message.content), title="System"))


def _print_tool_calls(message: AssistantMessage) -> None:
    """Render tool calls in the old dense-progress style."""
    for tool_call in message.tool_calls:
        _print_tool_call(tool_call)


def _format_tool_call_display(tool_call: Any) -> tuple[str, list[tuple[str, str, str]]]:
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


def _parse_tool_arguments_for_display(arguments: str) -> dict[str, Any] | None:
    """Best-effort JSON decoding for tool-call display."""
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _print_tool_call(tool_call: Any) -> None:
    """Print one tool call with the old dense-progress layout."""
    header, body_sections = _format_tool_call_display(tool_call)
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


async def _drive_agent(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    ui: UI,
    interactive: bool,
) -> None:
    """Drive one or more `run_agent` turns until the CLI should exit."""
    command_names = ["/exit", "/help", "/compact", "/image"]

    current_history = list(history)
    while True:
        renderer = DeltaRenderer()
        boundary = await run_agent_until_boundary(
            history=current_history,
            model=model,
            tools=tools,
            on_content=renderer.on_delta,
        )

        renderer.finish()

        if isinstance(boundary, AwaitingTools):
            _print_tool_calls(boundary.message)
            current_history = await execute_tool_calls(
                boundary=boundary,
                tools=tools,
            )
            continue

        current_history = boundary.history
        if not interactive:
            return

        while True:
            answer = await ui.prompt(words=command_names)
            stripped = answer.strip()
            if stripped == "/exit":
                return
            if stripped == "/help":
                print("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")
                continue
            if stripped == "/compact":
                current_history.append(
                    UserMessage(
                        content="Immediately compact our conversation so far by using the `compact_conversation` tool."
                    )
                )
                break
            if stripped.startswith("/image"):
                parts = stripped.split(maxsplit=1)
                if len(parts) < 2:
                    print("/image requires a path or URL argument.")
                    continue
                data_url = await get_image(parts[1])
                image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                current_history.append(UserMessage(content=image_content))
                break

            current_history.append(UserMessage(content=answer))
            break
