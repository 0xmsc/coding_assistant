"""Lazy-loading MCP server manager."""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastmcp import Client
from pydantic import BaseModel, Field, model_validator


class MCPServerConfig(BaseModel):
    """Configuration for one MCP server."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_command_or_url(self) -> "MCPServerConfig":
        """Require exactly one connection backend: command or URL."""
        if self.command and self.url:
            raise ValueError(f"MCP server '{self.name}' cannot have both a command and a url.")
        if not self.command and not self.url:
            raise ValueError(f"MCP server '{self.name}' must have either a command or a url.")
        return self


DEFAULT_ENV_VARS = [
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


@dataclass
class RunningMCPServer:
    """A running MCP server connection."""

    name: str
    client: Client[Any]
    tools: list[Any]  # MCP tool objects


def _get_default_env() -> dict[str, str]:
    """Return safe env vars to pass to MCP subprocesses."""
    return {k: v for k, v in os.environ.items() if k in DEFAULT_ENV_VARS}


class MCPServerManager:
    """Manage MCP server lifecycle for lazy loading."""

    def __init__(self, configs: list[MCPServerConfig], working_directory: Path) -> None:
        self._configs = {c.name: c for c in configs}
        self._running: dict[str, RunningMCPServer] = {}
        self._working_directory = working_directory
        self._exit_stack: AsyncExitStack | None = None

    @property
    def available_servers(self) -> list[str]:
        """List configured server names."""
        return list(self._configs.keys())

    @property
    def running_servers(self) -> list[str]:
        """List currently running server names."""
        return list(self._running.keys())

    async def start(self, name: str) -> str:
        """Start an MCP server by name."""
        if name in self._running:
            return f"Server '{name}' is already running."

        config = self._configs.get(name)
        if not config:
            return f"Unknown server '{name}'. Available: {', '.join(self.available_servers)}"

        # Build environment
        env = _get_default_env()
        for env_var in config.env:
            if env_var not in os.environ:
                return f"Required env var '{env_var}' not set."
            env[env_var] = os.environ[env_var]

        # Build args with variable substitution
        format_vars = {
            "working_directory": str(self._working_directory),
            "home_directory": str(Path.home()),
        }
        args = [arg.format(**format_vars) for arg in config.args]

        # Create MCP client
        backend: StdioMCPServer | RemoteMCPServer
        if config.url:
            from fastmcp.mcp_config import RemoteMCPServer

            backend = RemoteMCPServer(url=config.url)
        else:
            from fastmcp.mcp_config import StdioMCPServer

            backend = StdioMCPServer(
                command=config.command or "",
                args=args,
                env=env,
                cwd=str(self._working_directory),
            )

        client = Client(backend.to_transport(), name=name)

        try:
            async with client:
                await client.initialize()
                tools = await client.list_tools()

                self._running[name] = RunningMCPServer(
                    name=name,
                    client=client,
                    tools=list(tools),
                )

                tool_names = [t.name for t in tools]
                return f"Started '{name}' with {len(tools)} tools: {', '.join(tool_names)}"
        except Exception as exc:
            return f"Failed to start '{name}': {exc}"

    async def stop(self, name: str) -> str:
        """Stop a running MCP server."""
        server = self._running.pop(name, None)
        if not server:
            return f"Server '{name}' is not running."

        # Client is closed when exiting async context
        return f"Stopped '{name}'."

    async def list_tools(self, name: str) -> str:
        """List available tools on a running server."""
        server = self._running.get(name)
        if not server:
            return f"Server '{name}' is not running. Use mcp_start first."

        lines = [f"Tools on '{name}':"]
        for tool in server.tools:
            desc = tool.description or "(no description)"
            lines.append(f"  - {tool.name}: {desc}")

        return "\n".join(lines)

    async def call(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on a running MCP server."""
        server = self._running.get(server_name)
        if not server:
            return f"Server '{server_name}' is not running. Use mcp_start first."

        try:
            result = await server.client.call_tool(tool_name, arguments)

            # Unwrap single text result
            if len(result.content) == 1 and hasattr(result.content[0], "text"):
                return str(result.content[0].text)

            # Otherwise return repr
            return str(result.content)
        except Exception as exc:
            return f"Error calling '{tool_name}' on '{server_name}': {exc}"

    async def close(self) -> None:
        """Stop all running servers."""
        for name in list(self._running.keys()):
            await self.stop(name)
