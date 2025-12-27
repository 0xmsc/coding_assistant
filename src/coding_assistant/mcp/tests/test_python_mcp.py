import asyncio
import pytest
import httpx
from fastmcp import FastMCP
from coding_assistant.mcp.python import create_python_server
from coding_assistant.mcp.tasks import TaskManager


@pytest.fixture
def manager():
    return TaskManager()


async def run_mock_server(mcp: FastMCP, port: int):
    """Run the FastMCP server in the background."""
    await mcp.run_async(transport="streamable-http", port=port)


@pytest.mark.asyncio
async def test_python_execute_loopback_with_mock_server(manager):
    # 1. Create a Mock MCP server
    mock_mcp = FastMCP("MockServer")

    @mock_mcp.tool()
    async def identity(val: str) -> str:
        return f"Hello, {val}"

    port = 54321
    mcp_url = f"http://localhost:{port}/mcp"

    # Start mock server in background
    server_task = asyncio.create_task(run_mock_server(mock_mcp, port))

    try:
        # Wait for server to be ready
        started = False
        for _ in range(20):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(mcp_url, timeout=1.0)
                    if resp.status_code == 200:
                        started = True
                        break
                    if resp.status_code == 406:
                        started = True
                        break
            except httpx.ConnectError:
                pass
            await asyncio.sleep(0.5)

        if not started:
            pytest.fail("Mock MCP server failed to start")

        # 2. Setup the python_execute tool pointing to our mock server
        python_server = create_python_server(manager, mcp_url=mcp_url)
        execute_tool = await python_server.get_tool("execute")

        # 3. Code that uses Client to talk back to the mock server
        code = """
import asyncio
import os
from fastmcp import Client

async def run():
    url = os.environ.get("MCP_SERVER_URL")
    async with Client(url) as client:
        result = await client.call_tool("identity", {"val": "World"})
        # Extract text from result.content
        text = "".join([c.text for c in result.content if hasattr(c, "text")])
        print(text)

asyncio.run(run())
"""
        # 4. Execute and verify
        output = await execute_tool.fn(code=code, timeout=30)
        assert "Hello, World" in output

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
