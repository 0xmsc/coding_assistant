import asyncio
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
    instructions: str | None


class MCPWrappedTool(Tool):
    def __init__(self, client: Client, server_name: str, tool):
        self._client = client
        self._server_name = server_name
        self._tool = tool

    def name(self) -> str:
        return f"mcp_{self._server_name}_{self._tool.name}"

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
    for server in mcp_servers:
        tools = await server.client.list_tools()
        for tool in tools:
            wrapped.append(
                MCPWrappedTool(
                    client=server.client,
                    server_name=server.name,
                    tool=tool,
                )
            )
    return wrapped


def get_default_env():
    default_env = dict()
    if "HTTPS_PROXY" in os.environ:
        default_env["HTTPS_PROXY"] = os.environ["HTTPS_PROXY"]
    return default_env


@asynccontextmanager
async def launch_coding_assistant_mcp(root_directory: Path, working_directory: Path) -> AsyncGenerator[str, None]:
    """Launch the coding_assistant_mcp server as a network service and return its URL."""
    import socket

    def get_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    port = get_free_port()
    # FastMCP using 'streamable-http' serves on /mcp
    url = f"http://localhost:{port}/mcp"

    mcp_project_dir = root_directory / "packages" / "coding_assistant_mcp"

    args = [
        "uv",
        "--project",
        str(mcp_project_dir),
        "run",
        "coding-assistant-mcp",
        "--transport",
        "streamable-http",
        "--port",
        str(port),
    ]

    env = {**get_default_env(), **os.environ.copy()}

    logger.info(f"Launching coding_assistant_mcp on {url}")

    process = await asyncio.create_subprocess_exec(
        args[0],
        *args[1:],
        env=env,
        cwd=str(working_directory),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def log_stream(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.info(f"[{prefix}] {line.decode().strip()}")

    # Consume stdout and stderr in the background
    stdout_task = asyncio.create_task(log_stream(process.stdout, "mcp-server:stdout"))
    stderr_task = asyncio.create_task(log_stream(process.stderr, "mcp-server:stderr"))

    try:
        # Give it a moment to start up and bind to the port
        await asyncio.sleep(1)
        if process.returncode is not None:
            raise RuntimeError(f"MCP server died immediately with exit code {process.returncode}")
        yield url
    finally:
        logger.info("Terminating coding_assistant_mcp")
        process.terminate()
        await process.wait()
        stdout_task.cancel()
        stderr_task.cancel()


@asynccontextmanager
async def _get_mcp_server(
    name: str,
    config: StdioMCPServer | RemoteMCPServer,
) -> AsyncGenerator[MCPServer, None]:
    client = Client(config.to_transport(), name=name)
    async with client:
        yield MCPServer(
            name=name,
            client=client,
            instructions=None,  # FastMCP Client doesn't expose server instructions directly yet
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
