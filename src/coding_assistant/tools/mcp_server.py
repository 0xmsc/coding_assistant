import asyncio
import logging
import socket
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.tool import Tool as FastMCPTool, ToolResult as FastMCPToolResult
from pydantic import PrivateAttr

from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import Tool

logger = logging.getLogger(__name__)


def get_free_port() -> int:
    """Get a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class AggregatedTool(FastMCPTool):
    _wrapped_tool: Tool = PrivateAttr()

    def __init__(self, tool: Tool, **kwargs: Any):
        super().__init__(
            name=tool.name(),
            description=tool.description(),
            parameters=tool.parameters(),
            **kwargs,
        )

        self._wrapped_tool = tool

    async def run(self, arguments: dict[str, Any]) -> FastMCPToolResult:
        result = await self._wrapped_tool.execute(arguments)
        if not isinstance(result, TextResult):
            raise ValueError("Expected TextResult from wrapped tool execution.")
        return FastMCPToolResult(content=result.content)


async def start_mcp_server(tools: list[Tool], port: int) -> asyncio.Task:
    mcp = FastMCP("Coding Assistant", instructions="Exposes Coding Assistant tools via MCP")

    for tool in tools:
        agg_tool = AggregatedTool(tool=tool)
        mcp.add_tool(agg_tool)

    logger.info(f"Starting background MCP server on port {port}")

    task = asyncio.create_task(
        mcp.run_async(
            transport="streamable-http",
            port=port,
            show_banner=False,
            log_level="error",
        )
    )

    return task
