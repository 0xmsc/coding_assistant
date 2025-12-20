from collections.abc import Sequence
from coding_assistant.llm.types import Tool


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


async def get_tools(tools: Sequence[Tool]) -> list[dict]:
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
