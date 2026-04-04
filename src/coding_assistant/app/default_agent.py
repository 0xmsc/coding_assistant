from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from coding_assistant.app.instructions import get_instructions
from coding_assistant.llm.types import SystemMessage, Tool
from coding_assistant.tools.local_bundle import create_local_tool_bundle
from coding_assistant.tools.mcp_manager import MCPServerConfig


@dataclass(slots=True)
class DefaultAgentConfig:
    """Configuration for the default CLI and worker setup."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str


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
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
        mcp_server_configs=config.mcp_server_configs,
        working_directory=config.working_directory,
    )

    instructions = get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[local_tool_bundle.instructions],
    )

    try:
        yield DefaultAgentBundle(
            tools=local_tool_bundle.tools,
            instructions=instructions,
        )
    finally:
        await local_tool_bundle.close()


def build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=instructions)
