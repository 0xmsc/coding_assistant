import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, cast

from fastmcp import Client
from fastmcp.mcp_config import RemoteMCPServer, StdioMCPServer
from pydantic import BaseModel, Field, model_validator
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from coding_assistant.llm.types import Tool

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """Configuration for connecting to one MCP server."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: list[str] = Field(default_factory=list)
    prefix: str | None = None

    @model_validator(mode="after")
    def check_command_or_url(self) -> "MCPServerConfig":
        """Require exactly one connection backend: command or URL."""
        if self.command and self.url:
            raise ValueError(f"MCP server '{self.name}' cannot have both a command and a url.")
        if not self.command and not self.url:
            raise ValueError(f"MCP server '{self.name}' must have either a command or a url.")
        return self


@dataclass
class MCPServer:
    """Live MCP server connection plus its surfaced metadata."""

    name: str
    client: Client[Any]
    instructions: str | None = None
    prefix: str | None = None


class MCPWrappedTool(Tool):
    """Expose one MCP tool through the local `Tool` interface."""

    def __init__(self, client: Client[Any], tool: Any, prefix: str | None = None) -> None:
        self._client = client
        self._tool = tool
        self._prefix = prefix

    def name(self) -> str:
        """Return the tool name, including any configured server prefix."""
        if self._prefix:
            return f"{self._prefix}{self._tool.name}"
        return str(self._tool.name)

    def description(self) -> str:
        """Return the MCP tool description as plain text."""
        return str(self._tool.description or "")

    def parameters(self) -> dict[str, Any]:
        """Return the MCP tool input schema."""
        return cast(dict[str, Any], self._tool.inputSchema)

    async def execute(self, parameters: dict[str, Any]) -> str:
        """Call the remote MCP tool and unwrap its single text response."""
        result = await self._client.call_tool(self._tool.name, parameters)

        if len(result.content) != 1:
            raise ValueError("Expected exactly one result from MCP tool call.")

        if not hasattr(result.content[0], "text"):
            raise ValueError("Expected result to have a 'text' attribute.")

        content = result.content[0].text
        return str(content)


DEFAULT_VARS = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TERM",
]


@asynccontextmanager
async def _get_mcp_server(
    name: str,
    config: StdioMCPServer | RemoteMCPServer,
    prefix: str | None = None,
) -> AsyncGenerator[MCPServer, None]:
    """Open and initialize one MCP server connection."""
    client = Client(config.to_transport(), name=name)
    async with client:
        result = await client.initialize()
        yield MCPServer(
            name=name,
            client=client,
            instructions=result.instructions,
            prefix=prefix,
        )


async def get_mcp_wrapped_tools(mcp_servers: list[MCPServer]) -> list[Tool]:
    """List and wrap all tools exposed by the active MCP servers."""
    wrapped: list[Tool] = []
    names: set[str] = set()
    for server in mcp_servers:
        tools = await server.client.list_tools()
        for tool in tools:
            wrapped_tool = MCPWrappedTool(
                client=server.client,
                tool=tool,
                prefix=server.prefix,
            )
            name = wrapped_tool.name()
            if name in names:
                raise ValueError(f"MCP tool name collision detected: {name}")
            names.add(name)
            wrapped.append(wrapped_tool)
    return wrapped


def get_default_env() -> dict[str, str]:
    """Return the safe default environment passed through to MCP subprocesses."""
    default_env: dict[str, str] = dict()
    for var in DEFAULT_VARS:
        if var in os.environ:
            default_env[var] = os.environ[var]
    return default_env


@asynccontextmanager
async def get_mcp_servers_from_config(
    config_servers: list[MCPServerConfig], *, working_directory: Path
) -> AsyncGenerator[list[MCPServer], None]:
    """Create MCP servers from configuration objects."""
    if not working_directory.exists():
        raise ValueError(f"Working directory {working_directory} does not exist.")

    async with AsyncExitStack() as stack:
        servers: list[MCPServer] = []

        for server_config in config_servers:
            format_vars = {
                "working_directory": str(working_directory),
                "home_directory": str(Path.home()),
            }
            args = [arg.format(**format_vars) for arg in server_config.args]

            env = {**get_default_env()}

            for env_var in server_config.env:
                if env_var not in os.environ:
                    raise ValueError(
                        f"Required environment variable '{env_var}' for MCP server '{server_config.name}' is not set"
                    )
                env[env_var] = os.environ[env_var]

            backend: StdioMCPServer | RemoteMCPServer
            if server_config.url:
                backend = RemoteMCPServer(url=server_config.url)
            elif server_config.command:
                backend = StdioMCPServer(
                    command=server_config.command,
                    args=args,
                    env=env,
                    cwd=str(working_directory),
                )
            else:
                raise ValueError(f"MCP server '{server_config.name}' must have either a command or a url.")

            server = await stack.enter_async_context(
                _get_mcp_server(
                    name=server_config.name,
                    config=backend,
                    prefix=server_config.prefix,
                )
            )
            servers.append(server)

        yield servers


async def print_mcp_tools(mcp_servers: list[MCPServer]) -> None:
    """Pretty-print the available MCP tools for inspection in the CLI."""
    console = Console()

    if not mcp_servers:
        console.print("[yellow]No MCP servers found.[/yellow]")
        return

    table = Table(show_header=True, show_lines=True)
    table.add_column("Server Name", style="magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")

    for server in mcp_servers:
        tools = await server.client.list_tools()

        if not tools:
            logger.info(f"No tools found for MCP server: {server.name}")
            continue

        for tool in tools:
            table.add_row(
                server.name,
                tool.name,
                tool.description or "",
                Pretty(tool.inputSchema, expand_all=True, indent_size=2),
            )

    console.print(table)
