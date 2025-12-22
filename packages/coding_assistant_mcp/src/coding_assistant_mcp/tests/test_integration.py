import asyncio
import socket
import pytest
import pytest_asyncio
import httpx
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
    Test that the injected functions (get_available_mcp_tools, call_mcp_tool_sync)
    can successfully interact with the MCP server.
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
        
        # Test 1: Using the sync helper directly in a script (Standard use case)
        test_script_sync = """
# Test calling a tool synchronously
res = call_mcp_tool_sync("shell_execute", {"command": "echo 'Sync Loopback Success'"})
print(f"RESULT: {res.strip()}")
"""
        output = await python_execute.fn(code=test_script_sync, timeout=60)
        print(f"DEBUG OUTPUT SYNC:\n{output}")
        assert "RESULT: Sync Loopback Success" in output

        # Test 2: Using the async helper in a custom loop
        test_script_async = """
import asyncio
async def run_test():
    tools = await get_available_mcp_tools()
    print(f"TOOLS: {[t['name'] for t in tools]}")
    res = await call_mcp_tool("shell_execute", {"command": "echo 'Async Loopback Success'"})
    print(f"ASYNC_RESULT: {res.strip()}")

asyncio.run(run_test())
"""
        output = await python_execute.fn(code=test_script_async, timeout=60)
        print(f"DEBUG OUTPUT ASYNC:\n{output}")
        assert "ASYNC_RESULT: Async Loopback Success" in output
        assert "shell_execute" in output

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
