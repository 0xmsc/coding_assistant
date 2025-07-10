from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
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
async def get_filesystem_server(config: Config) -> AsyncGenerator[MCPServer, None]:
    assert config.working_directory.exists()

    params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(config.working_directory),
        ],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPServer(
                name="filesystem",
                session=session,
            )


@asynccontextmanager
async def get_git_server(config: Config) -> AsyncGenerator[MCPServer, None]:
    assert config.working_directory.exists()

    params = StdioServerParameters(
        command="uvx",
        args=[
            "mcp-server-git",
            "--repository",
            str(config.working_directory),
        ],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield MCPServer(
                name="git",
                session=session,
            )
