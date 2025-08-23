from coding_assistant.agents.types import Tool, ToolResult


def fix_input_schema(input_schema: dict):
    """
    Fixes the input schema to be compatible with Gemini API
    This is a workaround for the fact that Gemini API does not support certain values for the `format` field
    """

    for prop in input_schema.get("properties", {}).values():
        fmt = prop.get("format")
        if fmt == "uri":
            # Gemini API does not support `format: uri`, so we remove it
            prop.pop("format", None)


async def get_tools(tools: list[Tool]) -> list[dict]:
    """Convert Tool instances to LiteLLM format."""
    result: list[dict] = []
    for tool in tools:
        params = tool.parameters()
        fix_input_schema(params)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": params,
                },
            }
        )
    return result


async def execute_tool_call(function_name: str, function_args: dict, tools: list[Tool]) -> ToolResult:
    for tool in tools:
        if tool.name() == function_name:
            return await tool.execute(function_args)
    raise ValueError(f"Tool {function_name} not found in agent tools.")
