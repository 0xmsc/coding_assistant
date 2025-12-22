import asyncio
import socket
import pytest
import pytest_asyncio
import httpx
import os
from coding_assistant_mcp.main import _main
from coding_assistant_mcp.python import create_python_server
from coding_assistant_mcp.tasks import TaskManager

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

@pytest.mark.asyncio
async def test_python_mcp_integration():
    """
    Test that an agent can manually use fastmcp.Client to interact with the server
    using the MCP_SERVER_URL environment variable.
    """
    port = get_free_port()
    host = "localhost"
    url = f"http://{host}:{port}/mcp"

    # Start the MCP server in the background
    manager = TaskManager()
    from fastmcp import FastMCP
    from coding_assistant_mcp.shell import create_shell_server
    
    mcp = FastMCP("Integration Test Server")
    await mcp.import_server(create_shell_server(manager), prefix="shell")
    await mcp.import_server(create_python_server(manager, url), prefix="python")

    server_task = asyncio.create_task(mcp.run_async(
        transport="streamable-http",
        host=host,
        port=port,
        show_banner=False,
    ))

    # Give the server a moment to start
    for _ in range(20):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url)
                if resp.status_code < 500:
                    break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    else:
        server_task.cancel()
        pytest.fail("Server failed to start")

    try:
        python_execute = await mcp.get_tool("python_execute")
        
        # Manually call a tool via fastmcp.Client
        test_script = """
import asyncio
import os
from fastmcp import Client

async def run_test():
    url = os.environ.get("MCP_SERVER_URL")
    async with Client(url) as client:
        # Call shell_execute tool
        result = await client.call_tool("shell_execute", {"command": "echo 'Manual Client Success'"})
        
        # Extract text from CallToolResult manually
        text = "\\n".join([c.text for c in result.content if hasattr(c, "text")])
        print(f"RESULT: {text.strip()}")

asyncio.run(run_test())
"""
        output = await python_execute.fn(code=test_script, timeout=60)
        print(f"DEBUG OUTPUT:\n{output}")
        assert "RESULT: Manual Client Success" in output

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
