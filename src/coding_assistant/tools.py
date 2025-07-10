import logging
import os
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from coding_assistant.config import Config

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
async def get_filesystem_server(config: Config) -> AsyncGenerator[MCPServer, None]:
    async with _get_mcp_server(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(config.working_directory),
        ],
        env=get_default_env(),
    ) as server:
        yield server


@asynccontextmanager
async def get_fetch_server() -> AsyncGenerator[MCPServer, None]:
    async with _get_mcp_server(
        name="fetch",
        command="uvx",
        args=[
            "mcp-server-fetch",
        ],
        env=get_default_env(),
    ) as server:
        yield server


@asynccontextmanager
async def get_tavily_server() -> AsyncGenerator[MCPServer, None]:
    async with _get_mcp_server(
        name="tavily",
        command="npx",
        args=[
            "-y",
            "tavily-mcp@0.2.1",
        ],
        env={
            **get_default_env(),
            "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        },
    ) as server:
        yield server


@asynccontextmanager
async def get_all_mcp_servers(config: Config) -> AsyncGenerator[List[MCPServer], None]:
    if not config.working_directory.exists():
        raise ValueError(f"Working directory {config.working_directory} does not exist.")

    async with AsyncExitStack() as stack:
        servers: List[MCPServer] = []

        servers.append(await stack.enter_async_context(get_filesystem_server(config)))
        servers.append(await stack.enter_async_context(get_fetch_server()))

        if os.environ.get("TAVILY_API_KEY"):
            servers.append(await stack.enter_async_context(get_tavily_server()))

        yield servers
