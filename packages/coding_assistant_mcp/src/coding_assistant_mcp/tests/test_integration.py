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
    # We use a custom task manager and manually configure the python server with the URL
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
    # Polling with correct headers for streamable-http (GET expects certain headers or just works if server allows)
    # Actually, 406 means it didn't like the Accept header or similar. 
    # But for a health check, we just want to know it's UP.
    for _ in range(20):
        try:
            async with httpx.AsyncClient() as client:
                # We try a simple GET. Even if it returns 406, it means the server is responding.
                resp = await client.get(url)
                if resp.status_code < 500: # 406 is fine for "server is alive"
                    break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    else:
        server_task.cancel()
        pytest.fail("Server failed to start")

    try:
        # Now we use the python execute tool to run a script that calls back to the server
        python_execute = await mcp.get_tool("python_execute")
        
        test_script = """
import asyncio

async def run_test():
    # 1. Test listing tools
    tools = await get_available_mcp_tools()
    tool_names = [t['name'] for t in tools]
    print(f"TOOLS: {tool_names}")
    
    # 2. Test calling a tool (shell_execute)
    res = call_mcp_tool_sync("shell_execute", {"command": "echo 'Loopback Success'"})
    print(f"RESULT: {res.strip()}")

asyncio.run(run_test())
"""
        output = await python_execute.fn(code=test_script, timeout=60)
        print(f"DEBUG OUTPUT:\n{output}")
        
        assert "TOOLS:" in output
        assert "shell_execute" in output
        assert "python_execute" in output
        assert "RESULT: Loopback Success" in output

    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
