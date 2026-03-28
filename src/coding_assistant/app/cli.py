from __future__ import annotations

import os
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from coding_assistant.app.image import get_image
from coding_assistant.app.instructions import get_instructions
from coding_assistant.app.output import DeltaRenderer, print_system_message, print_tool_calls
from coding_assistant.app.ui import DefaultAnswerUI, PromptToolkitUI, UI
from coding_assistant.core.agent import (
    AwaitingTools,
    AwaitingUser,
    execute_tool_calls,
    run_agent_event_stream,
)
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import (
    BaseMessage,
    ContentDeltaEvent,
    SystemMessage,
    Tool,
    UserMessage,
)
from coding_assistant.tools import create_local_tool_bundle
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
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str
    mcp_servers: list[MCPServer]


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
        )


async def run_cli(args: Namespace) -> None:
    """Run the interactive or single-shot CLI entry point."""
    config = build_default_agent_config(args)

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
        print_system_message(system_message)
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


def _build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=build_system_prompt(instructions=instructions))


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
        boundary: AwaitingUser | AwaitingTools | None = None
        async for event in run_agent_event_stream(
            history=current_history,
            model=model,
            tools=tools,
        ):
            if isinstance(event, ContentDeltaEvent):
                renderer.on_delta(event.content)
                continue
            if isinstance(event, (AwaitingUser, AwaitingTools)):
                boundary = event

        renderer.finish()
        if boundary is None:
            raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

        if isinstance(boundary, AwaitingTools):
            print_tool_calls(boundary.message)
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
