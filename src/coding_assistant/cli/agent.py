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


@dataclass(slots=True)
class CliAgentConfig:
    """Configuration for the interactive CLI agent."""

    working_directory: Path
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
    return CliAgentConfig(
        working_directory=working_directory,
        skills_directories=tuple(args.skills_directories),
        user_instructions=tuple(args.instructions),
    )


@asynccontextmanager
async def create_cli_agent(*, config: CliAgentConfig) -> AsyncIterator[CliAgentBundle]:
    """Resolve instructions and tools for an interactive CLI run."""
    tool_bundle = create_cli_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
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
