import pytest

from coding_assistant.agents.types import TextResult, Tool
from coding_assistant.llm import adapters
from coding_assistant.tools.mcp import MCPWrappedTool, MCPServer
from mcp import ClientSession  # type: ignore
from types import SimpleNamespace


class DummyTool(Tool):
    def __init__(self, name: str, result: str):
        self._name = name
        self._result = result

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "desc"

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content=self._result)


def test_fix_input_schema_removes_uri_format():
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "other": {"type": "string"},
        },
    }

    adapters.fix_input_schema(schema)

    assert "format" not in schema["properties"]["url"]
    assert "format" not in schema["properties"]["other"]  # unchanged


@pytest.mark.asyncio
async def test_execute_tool_call_regular_tool_and_not_found():
    tool = DummyTool("echo", "ok")

    res = await adapters.execute_tool_call("echo", {}, tools=[tool])
    assert isinstance(res, TextResult)
    assert res.content == "ok"

    with pytest.raises(ValueError, match="Tool missing not found"):
        await adapters.execute_tool_call("missing", {}, tools=[tool])
