from __future__ import annotations

import os
import re
from argparse import Namespace
from collections.abc import Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from coding_assistant.app.image import get_image
from coding_assistant.app.instructions import get_instructions
from coding_assistant.app.output import DeltaRenderer, print_system_message, print_tool_calls
from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.history import build_system_prompt
from coding_assistant.core.tool_calls import execute_tool_calls
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.integrations.mcp_client import (
    MCPServer,
    MCPServerConfig,
    get_mcp_servers_from_config,
    get_mcp_wrapped_tools,
    print_mcp_tools,
)
from coding_assistant.llm.types import (
    BaseMessage,
    ContentDeltaEvent,
    SystemMessage,
    Tool,
    UserMessage,
)
from coding_assistant.tools.local_bundle import create_local_tool_bundle
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich import print
from rich.console import Console
from rich.rule import Rule

CLI_COMMAND_NAMES = ["/exit", "/help", "/compact", "/image"]


@dataclass(slots=True)
class DefaultAgentConfig:
    """Configuration for the default CLI and embedding setup."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str
    mcp_servers: list[MCPServer]


class SlashCompleter(Completer):
    """Autocomplete only slash-prefixed commands."""

    def __init__(self, words: list[str]):
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield slash-command completions when the prompt starts with `/`."""
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


def build_default_agent_config(args: Namespace) -> DefaultAgentConfig:
    """Translate CLI arguments into the default agent configuration."""
    working_directory = Path(os.getcwd())
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return DefaultAgentConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        user_instructions=tuple(args.instructions),
    )


@asynccontextmanager
async def create_default_agent(
    *,
    config: DefaultAgentConfig,
    include_worker_tools: bool = True,
) -> AsyncIterator[DefaultAgentBundle]:
    """Resolve instructions and tools for a default agent run."""
    local_tool_bundle = create_local_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
        include_worker_tools=include_worker_tools,
    )

    try:
        async with get_mcp_servers_from_config(
            list(config.mcp_server_configs),
            working_directory=config.working_directory,
        ) as servers:
            external_tools = await get_mcp_wrapped_tools(servers)
            instructions = get_instructions(
                working_directory=config.working_directory,
                user_instructions=list(config.user_instructions),
                extra_sections=[local_tool_bundle.instructions],
                mcp_servers=servers,
            )
            yield DefaultAgentBundle(
                tools=[*local_tool_bundle.tools, *external_tools],
                instructions=instructions,
                mcp_servers=servers,
            )
    finally:
        await local_tool_bundle.close()


async def run_cli(args: Namespace) -> None:
    """Run the interactive CLI entry point."""
    config = build_default_agent_config(args)
    prompt_session = _create_prompt_session()

    async def prompt_user(words: list[str] | None = None) -> str:
        return await _prompt_with_session(prompt_session, words=words)

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = _build_initial_system_message(instructions=bundle.instructions)
        print_system_message(system_message)
        current_history: list[BaseMessage] = [system_message]

        await _drive_agent(
            history=current_history,
            model=args.model,
            tools=bundle.tools,
            prompt_user=prompt_user,
        )


def _build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=build_system_prompt(instructions=instructions))


async def _drive_agent(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    prompt_user: Callable[[list[str] | None], Awaitable[str]],
) -> None:
    """Drive the interactive agent loop until the CLI should exit."""
    current_history = list(history)
    while True:
        renderer = DeltaRenderer()
        boundary: AwaitingUser | AwaitingToolCalls | None = None
        async for event in run_agent_event_stream(
            history=current_history,
            model=model,
            tools=tools,
        ):
            if isinstance(event, ContentDeltaEvent):
                renderer.on_delta(event.content)
                continue
            if isinstance(event, (AwaitingUser, AwaitingToolCalls)):
                boundary = event

        if boundary is None:
            renderer.finish()
            raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

        if isinstance(boundary, AwaitingToolCalls):
            renderer.finish(trailing_blank_line=False)
            print_tool_calls(boundary.message)
            current_history = await execute_tool_calls(
                boundary=boundary,
                tools=tools,
            )
            continue

        renderer.finish()
        current_history = boundary.history
        while True:
            answer = await prompt_user(CLI_COMMAND_NAMES)
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


def _create_prompt_session() -> PromptSession[str]:
    """Create the prompt-toolkit session used by the interactive CLI."""
    history_dir = get_app_cache_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=True,
    )


async def _prompt_with_session(prompt_session: PromptSession[str], *, words: list[str] | None = None) -> str:
    """Prompt for input with optional slash-command completion."""
    Console().bell()
    print(Rule(style="dim"))
    completer = SlashCompleter(words) if words else None
    return await prompt_session.prompt_async("> ", completer=completer, complete_while_typing=True)
