from abc import ABC, abstractmethod
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from coding_assistant.config import Config


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


@dataclass
class Tools:
    mcp_servers: list = field(default_factory=list)
    tools: list = field(default_factory=list)


@asynccontextmanager
async def _get_mcp_server(
    name: str, command: str, args: List[str], env: dict[str, str] | None = None
) -> AsyncGenerator[MCPServer, None]:
    params = StdioServerParameters(command=command, args=args, env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPServer(name=name, session=session)


@asynccontextmanager
async def get_filesystem_server(config: Config) -> AsyncGenerator[MCPServer, None]:
    if not config.working_directory.exists():
        raise ValueError(
            f"Working directory {config.working_directory} does not exist."
        )

    async with _get_mcp_server(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(config.working_directory),
        ],
    ) as server:
        yield server


@asynccontextmanager
async def get_git_server(config: Config) -> AsyncGenerator[MCPServer, None]:
    assert config.working_directory.exists()

    async with _get_mcp_server(
        name="git",
        command="uvx",
        args=[
            "mcp-server-git",
            "--repository",
            str(config.working_directory),
        ],
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
    ) as server:
        yield server


@asynccontextmanager
async def get_tavily_server() -> AsyncGenerator[MCPServer, None]:
    async with _get_mcp_server(
        name="tavily",
        command="npx",
        args=[
            "-y",
            "tavily-mcp@0.1.4",
        ],
        env={
            "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"],
        },
    ) as server:
        yield server


@asynccontextmanager
async def get_all_mcp_servers(config: Config) -> AsyncGenerator[List[MCPServer], None]:
    servers: List[MCPServer] = []

    async with AsyncExitStack() as stack:
        servers.append(await stack.enter_async_context(get_filesystem_server(config)))
        servers.append(await stack.enter_async_context(get_git_server(config)))
        servers.append(await stack.enter_async_context(get_fetch_server()))

        if os.environ.get("TAVILY_API_KEY"):
            servers.append(await stack.enter_async_context(get_tavily_server()))

        yield servers
