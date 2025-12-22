import asyncio
import pytest
from fastmcp import Client

from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import Tool
from coding_assistant.tools.mcp_server import start_mcp_server


class MockTool(Tool):
    def name(self) -> str:
        return "mock_tool"

    def description(self) -> str:
        return "mock description"

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"val": {"type": "string"}},
            "required": ["val"],
        }

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content=f"Mock result: {parameters.get('val')}")


@pytest.mark.asyncio
async def test_mcp_aggregator_integration():
    port = 58766
    tools = [MockTool()]

    task = await start_mcp_server(tools, port)

    try:
        url = f"http://127.0.0.1:{port}/mcp"

        success = False
        for _ in range(50):
            async with Client(url) as client:
                mcp_tools = await client.list_tools()

                if any(t.name == "mock_tool" for t in mcp_tools):
                    # Tool call test
                    result = await client.call_tool("mock_tool", {"val": "integration test"})
                    assert len(result.content) > 0
                    assert hasattr(result.content[0], "text")
                    assert result.content[0].text == "Mock result: integration test"
                    success = True
                    break

            await asyncio.sleep(0.1)

        if not success:
            pytest.fail("Background MCP server failed to respond correctly within timeout.")
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
