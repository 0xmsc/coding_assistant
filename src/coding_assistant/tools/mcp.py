import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

from fastmcp import Client
from fastmcp.mcp_config import RemoteMCPServer, StdioMCPServer
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table

from coding_assistant.framework.types import TextResult, Tool
from coding_assistant.config import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    name: str
    client: Client
    instructions: str | None = None
    prefix: str | None = None


class MCPWrappedTool(Tool):
    def __init__(self, client: Client, server_name: str, tool, prefix: str | None = None):
        self._client = client
        self._server_name = server_name
        self._tool = tool
        self._prefix = prefix

    def name(self) -> str:
        if self._prefix:
            return f"{self._prefix}{self._tool.name}"
        return self._tool.name

    def description(self) -> str:
        return self._tool.description or ""

    def parameters(self) -> dict:
        return self._tool.inputSchema

    async def execute(self, parameters) -> TextResult:
        result = await self._client.call_tool(self._tool.name, parameters)

        if len(result.content) != 1:
            raise ValueError("Expected exactly one result from MCP tool call.")

        if not hasattr(result.content[0], "text"):
            raise ValueError("Expected result to have a 'text' attribute.")

        content = result.content[0].text
        return TextResult(content=content)


async def get_mcp_wrapped_tools(mcp_servers: list[MCPServer]) -> list[Tool]:
    wrapped: list[Tool] = []
    names: set[str] = set()
    for server in mcp_servers:
        tools = await server.client.list_tools()
        for tool in tools:
            wrapped_tool = MCPWrappedTool(
                client=server.client,
                server_name=server.name,
                tool=tool,
                prefix=server.prefix,
            )
            name = wrapped_tool.name()
            if name in names:
                raise ValueError(f"MCP tool name collision detected: {name}")
            names.add(name)
            wrapped.append(wrapped_tool)
    return wrapped


def get_default_env():
    default_env = dict()
    if "HTTPS_PROXY" in os.environ:
        default_env["HTTPS_PROXY"] = os.environ["HTTPS_PROXY"]
    return default_env


@asynccontextmanager
async def _get_mcp_server(
    name: str,
    config: StdioMCPServer | RemoteMCPServer,
    prefix: str | None = None,
) -> AsyncGenerator[MCPServer, None]:
    client = Client(config.to_transport(), name=name)
    async with client:
        result = await client.initialize()
        yield MCPServer(
            name=name,
            client=client,
            instructions=result.instructions,
            prefix=prefix,
        )


@asynccontextmanager
async def get_mcp_servers_from_config(
    config_servers: list[MCPServerConfig], working_directory: Path
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


async def print_mcp_tools(mcp_servers):
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
