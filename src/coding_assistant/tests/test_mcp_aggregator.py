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
    # Use a unique port for the test
    port = 58766
    tools = [MockTool()]

    # Start the server in the background
    task = await start_mcp_server(tools, port)

    try:
        url = f"http://127.0.0.1:{port}/mcp"

        # Wait for the server to be ready by attempting to connect with the Client
        mcp_client = None
        for _ in range(50):  # Wait up to 5 seconds
            try:
                # We want to keep the client open only if it succeeds
                async with Client(url) as client:
                    # If we can list tools, the server is fully ready
                    mcp_tools = await client.list_tools()
                    if any(t.name == "mock_tool" for t in mcp_tools):
                        # Tool call test
                        result = await client.call_tool("mock_tool", {"val": "integration test"})
                        assert len(result.content) > 0
                        assert hasattr(result.content[0], "text")
                        assert result.content[0].text == "Mock result: integration test"
                        mcp_client = True  # Success flag
                        break
            except Exception:
                pass
            await asyncio.sleep(0.1)

        if not mcp_client:
            pytest.fail("Background MCP server failed to respond correctly within timeout.")

    finally:
        # Clean up the background task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
