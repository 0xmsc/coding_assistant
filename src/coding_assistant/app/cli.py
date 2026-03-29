from __future__ import annotations

import os
import re
from argparse import Namespace
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from coding_assistant.app.image import get_image
from coding_assistant.app.instructions import get_instructions
from coding_assistant.app.output import DeltaRenderer
from coding_assistant.core.history import build_system_prompt
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.llm.types import SystemMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich import print
from rich.console import Console
from rich.rule import Rule

from coding_assistant.tools.local_bundle import create_local_tool_bundle
from coding_assistant.integrations.mcp_client import (
    MCPServer,
    MCPServerConfig,
    get_mcp_servers_from_config,
    get_mcp_wrapped_tools,
)
from coding_assistant.llm.types import Tool

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
    set_local_worker_endpoint: Callable[[str], None]
    close_tools: Callable[[], Awaitable[None]]


class SlashCompleter(Completer):
    """Autocomplete only slash-prefixed commands."""

    def __init__(self, words: list[str]):
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield slash-command completions when the prompt starts with `/`."""
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


def _build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=build_system_prompt(instructions=instructions))


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


async def handle_cli_input(
    *,
    answer: str,
    renderer: DeltaRenderer,
    submit_prompt_or_warn: Callable[[str | list[dict[str, Any]]], Awaitable[bool]],
) -> bool:
    """Handle CLI-specific commands and convert them into semantic session actions."""
    stripped = answer.strip()
    if stripped == "/exit":
        renderer.finish()
        return True
    if stripped == "/help":
        print("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")
        return False
    if stripped == "/compact":
        await submit_prompt_or_warn(
            "Immediately compact our conversation so far by using the `compact_conversation` tool."
        )
        return False
    if stripped.startswith("/image"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            print("/image requires a path or URL argument.")
            return False
        data_url = await get_image(parts[1])
        image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
        await submit_prompt_or_warn(image_content)
        return False

    await submit_prompt_or_warn(answer)
    return False


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
) -> AsyncIterator[DefaultAgentBundle]:
    """Resolve instructions and tools for a default agent run."""
    local_tool_bundle = create_local_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories]
    )

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
            set_local_worker_endpoint=local_tool_bundle.set_local_worker_endpoint,
            close_tools=local_tool_bundle.close,
        )
