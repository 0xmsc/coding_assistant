from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from coding_assistant.cli.tool_bundle import create_cli_tool_bundle
from coding_assistant.core.instructions import get_instructions
from coding_assistant.llm.types import Tool
from coding_assistant.tools.mcp_manager import MCPServerConfig


@dataclass(slots=True)
class CliAgentConfig:
    """Configuration for the interactive CLI agent."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()


@dataclass(slots=True)
class CliAgentBundle:
    """Resolved instructions and tools needed to run a CLI agent."""

    tools: list[Tool]
    instructions: str


def build_cli_agent_config(args: Namespace) -> CliAgentConfig:
    """Translate CLI arguments into the CLI agent configuration."""
    working_directory = Path(os.getcwd())
    mcp_server_configs = tuple(MCPServerConfig.model_validate_json(item) for item in args.mcp_servers)
    return CliAgentConfig(
        working_directory=working_directory,
        mcp_server_configs=mcp_server_configs,
        skills_directories=tuple(args.skills_directories),
        user_instructions=tuple(args.instructions),
    )


@asynccontextmanager
async def create_cli_agent(*, config: CliAgentConfig) -> AsyncIterator[CliAgentBundle]:
    """Resolve instructions and tools for an interactive CLI run."""
    tool_bundle = create_cli_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
        mcp_server_configs=config.mcp_server_configs,
        working_directory=config.working_directory,
    )

    instructions = get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[tool_bundle.instructions],
    )

    try:
        yield CliAgentBundle(
            tools=tool_bundle.tools,
            instructions=instructions,
        )
    finally:
        await tool_bundle.close()
