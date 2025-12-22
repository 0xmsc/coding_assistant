import asyncio
import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.tool import Tool as FastMCPTool
from pydantic import PrivateAttr

from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import Tool

logger = logging.getLogger(__name__)


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

    async def run(self, arguments: dict[str, Any]) -> str:
        result = await self._wrapped_tool.execute(arguments)
        if not isinstance(result, TextResult):
            raise ValueError("Expected TextResult from wrapped tool execution.")
        return result.content


async def start_mcp_server(tools: list[Tool], port: int) -> asyncio.Task:
    mcp = FastMCP(
        "Coding Assistant", instructions="Exposes Coding Assistant tools via MCP"
    )

    for tool in tools:
        # Create an instance of our custom Tool subclass and register it.
        agg_tool = AggregatedTool(tool=tool)
        mcp.add_tool(agg_tool)

    logger.info(f"Starting background MCP server on port {port}")

    # Start the server as a background task.
    task = asyncio.create_task(
        mcp.run_async(transport="streamable-http", port=port, show_banner=False)
    )

    return task
