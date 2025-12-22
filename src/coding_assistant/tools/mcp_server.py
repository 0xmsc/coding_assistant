import asyncio
import logging
from typing import Any, Optional, Type

from fastmcp import FastMCP
from pydantic import BaseModel, Field, create_model

from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import Tool

logger = logging.getLogger(__name__)


def _schema_to_pydantic(schema: dict[str, Any]) -> Type[BaseModel]:
    """
    Convert a JSON schema (as returned by Tool.parameters()) into a Pydantic model.
    This allows FastMCP to correctly expose the tool's input schema.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for name, prop in properties.items():
        # Map JSON schema types to Python types
        json_type = prop.get("type")
        description = prop.get("description")

        py_type: Any
        if json_type == "string":
            py_type = str
        elif json_type == "integer":
            py_type = int
        elif json_type == "number":
            py_type = float
        elif json_type == "boolean":
            py_type = bool
        elif json_type == "array":
            py_type = list[Any]
        elif json_type == "object":
            py_type = dict[str, Any]
        else:
            py_type = Any

        if name not in required:
            py_type = Optional[py_type]
            fields[name] = (py_type, Field(default=None, description=description))
        else:
            fields[name] = (py_type, Field(description=description))

    # If no properties are defined, we return an empty model
    if not fields:
        return create_model("Arguments", __base__=BaseModel)

    return create_model("Arguments", __base__=BaseModel, **fields)


async def start_mcp_server(tools: list[Tool], port: int) -> asyncio.Task:
    """
    Create and start a FastMCP server in the background that provides access to the given tools.
    """
    mcp = FastMCP(
        "Coding Assistant", instructions="Exposes Coding Assistant tools via MCP"
    )

    for tool in tools:
        # Create a Pydantic model from the tool's JSON schema
        try:
            ArgsModel = _schema_to_pydantic(tool.parameters())
        except Exception as e:
            logger.warning(
                f"Failed to create Pydantic model for tool {tool.name()}: {e}"
            )
            # Fallback to an empty model
            ArgsModel = create_model("Arguments", __base__=BaseModel)

        # Create the handler function
        # Using a closure and default argument to capture the specific tool instance
        def make_handler(t: Tool, model: Type[BaseModel]):
            async def handler(args: BaseModel) -> str:
                # Convert the Pydantic model back to a dict for the Tool.execute method
                params = args.model_dump()
                result = await t.execute(params)
                if isinstance(result, TextResult):
                    return result.content
                return str(result)

            # We need to set the type hint for the first argument so FastMCP can see it
            handler.__annotations__ = {"args": model, "return": str}
            handler.__name__ = t.name()
            handler.__doc__ = t.description()
            return handler

        mcp.tool(
            name=tool.name(),
            description=tool.description(),
        )(make_handler(tool, ArgsModel))

    logger.info(f"Starting background MCP server on port {port}")

    # Start the server as a background task
    task = asyncio.create_task(
        mcp.run_async(transport="streamable-http", port=port, show_banner=False)
    )

    return task
