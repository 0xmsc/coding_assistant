import logging
import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from coding_assistant.config import Config, MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    name: str
    session: ClientSession


class Tool(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters(self) -> dict: ...

    @abstractmethod
    async def execute(self, parameters) -> str: ...


def get_default_env():
    default_env = dict()
    if "HTTPS_PROXY" in os.environ:
        default_env["HTTPS_PROXY"] = os.environ["HTTPS_PROXY"]
    return default_env


@dataclass
class Tools:
    mcp_servers: list = field(default_factory=list)
    tools: list = field(default_factory=list)


@asynccontextmanager
async def _get_mcp_server(
    name: str, command: str, args: List[str], env: dict[str, str] | None = None
) -> AsyncGenerator[MCPServer, None]:
    logger.info(f"Starting MCP server '{name}' with command '{command}', args '{' '.join(args)}' and env: '{env}'")
    params = StdioServerParameters(command=command, args=args, env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPServer(name=name, session=session)


@asynccontextmanager
async def get_mcp_servers_from_config(
    config_servers: List[MCPServerConfig], working_directory: Path
) -> AsyncGenerator[List[MCPServer], None]:
    """Create MCP servers from configuration objects."""
    if not working_directory.exists():
        raise ValueError(f"Working directory {working_directory} does not exist.")

    async with AsyncExitStack() as stack:
        servers: List[MCPServer] = []

        for server_config in config_servers:
            # Format all arguments with available variables
            format_vars = {"working_directory": str(working_directory)}
            args = [arg.format(**format_vars) for arg in server_config.args]

            # Merge environment variables with current environment and server-specific env
            env = {**get_default_env()}

            # Add environment variables specified in server config
            for env_var in server_config.env:
                if env_var in os.environ:
                    env[env_var] = os.environ[env_var]

            server = await stack.enter_async_context(
                _get_mcp_server(name=server_config.name, command=server_config.command, args=args, env=env)
            )
            servers.append(server)

        yield servers
