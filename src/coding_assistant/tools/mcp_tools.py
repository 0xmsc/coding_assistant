"""MCP management tools - start/stop/call MCP servers without dynamic registration."""

from __future__ import annotations

from typing import Any

from coding_assistant.llm.types import Tool
from coding_assistant.tools.mcp_manager import MCPServerManager


class MCPStartTool(Tool):
    """Start an MCP server and connect to it."""

    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "mcp_start"

    def description(self) -> str:
        available = ", ".join(self._manager.available_servers) if self._manager.available_servers else "none"
        return f"Start an MCP server by name. Available servers: {available}"

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "enum": list(self._manager.available_servers),
                    "description": "The MCP server to start",
                },
            },
            "required": ["server"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return await self._manager.start(parameters["server"])


class MCPStopTool(Tool):
    """Stop a running MCP server."""

    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "mcp_stop"

    def description(self) -> str:
        return "Stop a running MCP server by name."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "The MCP server to stop",
                },
            },
            "required": ["server"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return await self._manager.stop(parameters["server"])


class MCPCallTool(Tool):
    """Call a tool from a running MCP server."""

    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "mcp_call"

    def description(self) -> str:
        return (
            "Call a tool from a running MCP server. "
            "The server must be started first with mcp_start. "
            "Use mcp_list_tools(server='...') to see available tools."
        )

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "The MCP server name",
                },
                "tool": {
                    "type": "string",
                    "description": "The tool name to call",
                },
                "arguments": {
                    "type": "object",
                    "description": "Tool arguments as key-value pairs",
                },
            },
            "required": ["server", "tool"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return await self._manager.call(
            parameters["server"],
            parameters["tool"],
            parameters.get("arguments", {}),
        )


class MCPListToolsTool(Tool):
    """List available tools on a running MCP server."""

    def __init__(self, manager: MCPServerManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "mcp_list_tools"

    def description(self) -> str:
        return "List available tools on a running MCP server."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {
                    "type": "string",
                    "description": "The MCP server name",
                },
            },
            "required": ["server"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return await self._manager.list_tools(parameters["server"])


def create_mcp_tools(manager: MCPServerManager) -> list[Tool]:
    """Create MCP management tools."""
    return [
        MCPStartTool(manager),
        MCPStopTool(manager),
        MCPCallTool(manager),
        MCPListToolsTool(manager),
    ]
