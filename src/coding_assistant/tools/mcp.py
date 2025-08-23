import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Set

from mcp import ClientSession, StdioServerParameters
import mcp
from mcp.client.stdio import stdio_client
from rich.console import Console
from rich.table import Table

from coding_assistant.agents.types import TextResult, Tool
from coding_assistant.config import MCPServerConfig

logger = logging.getLogger(__name__)


@dataclass
class MCPServer:
    name: str
    session: ClientSession
    instructions: str | None


def _fix_input_schema(input_schema: dict):
    """Normalize input schema (e.g. remove unsupported formats)."""
    if not input_schema:
        return
    for prop in input_schema.get("properties", {}).values():
        if prop.get("format") == "uri":
            prop.pop("format", None)


class MCPWrappedTool(Tool):
    def __init__(self, session: ClientSession, server_name: str, function_name: str, description: str, schema: dict):
        self._session = session
        self._server_name = server_name
        self._function_name = function_name
        self._description = description
        self._schema = schema
        _fix_input_schema(self._schema)

    def name(self) -> str:
        return f"mcp_{self._server_name}_{self._function_name}"

    def description(self) -> str:
        return self._description

    def parameters(self) -> dict:
        return self._schema

    async def execute(self, parameters) -> TextResult:
        result = await self._session.call_tool(self._function_name, parameters)

        if len(result.content) != 1:
            raise ValueError(f"Expected single result, got {len(result.content)}")

        if not isinstance(result.content[0], mcp.types.TextContent):
            raise ValueError(f"Expected TextContent, got {type(result.content[0])}")

        return TextResult(content=result.content[0].text)


async def get_mcp_wrapped_tools(mcp_servers: list[MCPServer]) -> list[Tool]:
    wrapped: List = []
    for server in mcp_servers:
        tools_response = await server.session.list_tools()
        for remote_tool in getattr(tools_response, "tools"):
            wrapped.append(
                MCPWrappedTool(
                    session=server.session,
                    server_name=server.name,
                    function_name=remote_tool.name,
                    description=remote_tool.description,
                    schema=remote_tool.inputSchema,
                )
            )
    return wrapped


def get_default_env():
    default_env = dict()
    if "HTTPS_PROXY" in os.environ:
        default_env["HTTPS_PROXY"] = os.environ["HTTPS_PROXY"]
    return default_env


@asynccontextmanager
async def _get_mcp_server(
    name: str, command: str, args: List[str], env: dict[str, str] | None = None
) -> AsyncGenerator[MCPServer, None]:
    logger.info(f"Starting MCP server '{name}' with command '{command}', args '{' '.join(args)}' and env: '{env}'")
    params = StdioServerParameters(command=command, args=args, env=env)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialize_result = await session.initialize()
            yield MCPServer(
                name=name,
                session=session,
                instructions=initialize_result.instructions,
            )


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
                if env_var not in os.environ:
                    raise ValueError(
                        f"Required environment variable '{env_var}' for MCP server '{server_config.name}' is not set"
                    )
                env[env_var] = os.environ[env_var]

            server = await stack.enter_async_context(
                _get_mcp_server(name=server_config.name, command=server_config.command, args=args, env=env)
            )
            servers.append(server)

        yield servers


async def handle_mcp_tool_call(function_name, arguments, mcp_servers):
    parts = function_name.split("_")
    assert parts[0] == "mcp"

    server_name = parts[1]
    tool_name = "_".join(parts[2:])

    for server in mcp_servers:
        if server.name == server_name:
            result = await server.session.call_tool(tool_name, arguments)
            if not result.content:
                return "MCP server did not return any content."
            return result.content[0].text

    raise RuntimeError(f"Server {server_name} not found in MCP servers.")


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
        tools_response = await server.session.list_tools()
        server_tools = tools_response.tools

        if not server_tools:
            logger.info(f"No tools found for MCP server: {server.name}")
            continue

        for tool in server_tools:
            name = tool.name
            description = tool.description
            parameters = tool.inputSchema
            parameters_str = ", ".join(parameters.get("properties", {}).keys()) if parameters else "None"
            table.add_row(server.name, name, description, parameters_str)

    console.print(table)
