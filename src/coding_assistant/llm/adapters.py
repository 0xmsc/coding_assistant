"""
Adapters for converting tool structures to LiteLLM format.

This module contains functions that adapt various tool representations
(MCP servers, Tool instances) to the format expected by LiteLLM.
"""

from typing import List

from coding_assistant.agents.types import Tool, TextResult, ToolResult
from coding_assistant.tools.mcp import MCPServer, handle_mcp_tool_call


def fix_input_schema(input_schema: dict):
    """
    Fixes the input schema to be compatible with Gemini API
    This is a workaround for the fact that Gemini API does not support certain values for the `format` field
    """

    for property in input_schema.get("properties", {}).values():
        if (format := property.get("format")) and format == "uri":
            # Gemini API does not support `format: uri`, so we remove it
            property.pop("format", None)


async def get_tools(tools: list[Tool], mcp_servers: list[MCPServer]) -> list[dict]:
    """
    Convert both Tool instances and MCP server tools to LiteLLM function calling format.

    Args:
        tools: List of Tool instances
        mcp_servers: List of MCPServer instances

    Returns:
        List of tool definitions in LiteLLM format
    """
    result = []
    
    # Add tools from the tools list
    for tool in tools:
        if tool.name().startswith("mcp_"):
            raise ValueError("Tools cannot start with mcp_")

        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                },
            }
        )
    
    # Add tools from MCP servers
    for server in mcp_servers:
        for _, tool_list in await server.session.list_tools():
            for mcp_tool in tool_list or []:
                tool_id = f"mcp_{server.name}_{mcp_tool.name}"

                fix_input_schema(mcp_tool.inputSchema)

                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_id,
                            "description": mcp_tool.description,
                            "parameters": mcp_tool.inputSchema,
                        },
                    }
                )
    
    return result


async def execute_tool_call(function_name: str, function_args: dict, tools: list[Tool], mcp_servers: list[MCPServer]) -> ToolResult:
    """
    Execute a tool call, handling both regular tools and MCP server tools.

    Args:
        function_name: Name of the function to call
        function_args: Arguments to pass to the function
        tools: List of available Tool instances
        mcp_servers: List of available MCP servers

    Returns:
        ToolResult from the executed tool

    Raises:
        RuntimeError: If the tool is not found
    """
    if function_name.startswith("mcp_"):
        content = await handle_mcp_tool_call(function_name, function_args, mcp_servers)
        return TextResult(content=content)
    else:
        for tool in tools:
            if tool.name() == function_name:
                return await tool.execute(function_args)
        raise ValueError(f"Tool {function_name} not found in agent tools.")
