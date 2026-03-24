from __future__ import annotations

import importlib.resources
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from coding_assistant.config import MCPServerConfig
from coding_assistant.history_store import FileHistoryStore
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.types import Tool
from coding_assistant.mcp import __name__ as mcp_package_name
from coding_assistant.tools.mcp import MCPServer, get_mcp_servers_from_config, get_mcp_wrapped_tools


@dataclass(slots=True)
class DefaultAgentConfig:
    """Configuration for the default CLI and embedding setup."""

    working_directory: Path
    mcp_server_configs: tuple[MCPServerConfig, ...] = ()
    skills_directories: tuple[str, ...] = ()
    mcp_env: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()
    coding_assistant_root: Path | None = None


@dataclass(slots=True)
class DefaultAgentBundle:
    """Resolved defaults needed to run an agent in one place."""

    tools: list[Tool]
    instructions: str
    mcp_servers: list[MCPServer]
    history_store: FileHistoryStore


def get_default_mcp_server_config(
    root_directory: Path,
    skills_directories: tuple[str, ...],
    env: tuple[str, ...] = (),
) -> MCPServerConfig:
    """Build the bundled MCP server config used by default runs."""
    import sys

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
    """Resolve instructions, tools, and history storage for a default agent run."""
    root = config.coding_assistant_root or Path(str(importlib.resources.files("coding_assistant"))).parent.resolve()
    history_store = FileHistoryStore(config.working_directory)
    server_configs = (
        *config.mcp_server_configs,
        get_default_mcp_server_config(root, config.skills_directories, config.mcp_env),
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
            history_store=history_store,
        )
