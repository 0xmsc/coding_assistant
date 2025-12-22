import asyncio
import logging
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool as FastMCPTool

from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import Tool

logger = logging.getLogger(__name__)


class AggregatedTool(FastMCPTool):
    """
    An MCP tool that wraps a coding_assistant Tool object.

    This implementation inherits from FastMCP's base Tool class and
    overwrites the run() method to delegate to the wrapped tool.
    """

    # Note: FastMCPTool is a Pydantic model. We use this field to store our internal tool.
    # We provide a default factory to satisfy Pydantic if needed, but it's set in __init__.
    wrapped_tool: Any

    def __init__(self, tool: Tool, **kwargs: Any):
        # We pass metadata to the parent Tool model.
        # FastMCP uses these fields for the MCP list_tools response.
        super().__init__(
            name=tool.name(),
            description=tool.description(),
            parameters=tool.parameters(),
            wrapped_tool=tool,
            **kwargs,
        )

    async def run(self, arguments: dict[str, Any]) -> str:
        """
        Overwritten run method that FastMCP calls when the tool is executed.
        """
        result = await self.wrapped_tool.execute(arguments)
        assert isinstance(
            result, TextResult
        ), f"Expected TextResult from tool {self.name}, got {type(result).__name__}"
        return result.content


async def start_mcp_server(tools: list[Tool], port: int) -> asyncio.Task:
    """
    Create and start a FastMCP server in the background that provides access to the given tools.
    """
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
