import pytest
from typing import cast

from types import SimpleNamespace

from coding_assistant.agents.types import Tool
from coding_assistant.llm.adapters import get_tools
from coding_assistant.tools.mcp import MCPServer
from mcp import ClientSession


class _DummyTool(Tool):
    def name(self) -> str:
        return "echo"

    def description(self) -> str:
        return "desc"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, parameters: dict):
        raise NotImplementedError


class _McpTool:
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeMcpSession:
    def __init__(self, tools):
        self._tools = tools

    async def list_tools(self):
        # Mimic MCP response with attribute 'tools'
        return SimpleNamespace(tools=self._tools)


@pytest.mark.asyncio
async def test_get_tools_combines_regular_and_mcp_and_fixes_schema():
    regular = _DummyTool()

    mcp_tool = _McpTool(
        name="download",
        description="Download a URL",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "format": "uri"},
            },
            "required": ["url"],
        },
    )

    mcp_server = MCPServer(name="srv", session=cast(ClientSession, _FakeMcpSession([mcp_tool])), instructions=None)

    tools = await get_tools([regular], [mcp_server])

    # Expect 2 tools, first regular, second MCP
    names = [t["function"]["name"] for t in tools]
    assert names == ["echo", "mcp_srv_download"]

    # Check schema fix applied (format removed)
    mcp_params = tools[1]["function"]["parameters"]
    assert "format" not in mcp_params["properties"]["url"]


@pytest.mark.asyncio
async def test_get_tools_rejects_regular_tool_with_mcp_prefix():
    class BadTool(_DummyTool):
        def name(self) -> str:
            return "mcp_bad"

    with pytest.raises(ValueError, match="Tools cannot start with mcp_"):
        await get_tools([BadTool()], [])
