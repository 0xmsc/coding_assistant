import asyncio
from pathlib import Path

import httpx
import pytest
from fastmcp import Client

from coding_assistant_mcp.proc import start_process


@pytest.mark.asyncio
async def test_python_mcp_integration_real_server():
    """
    Test that an agent can interact with a REAL MCP server instance started via 'uv run'.
    """
    port = 53675
    host = "localhost"
    url = f"http://{host}:{port}/mcp"

    # Start the REAL MCP server using our internal proc utility
    project_root = Path(__file__).parents[2].absolute()
    cmd = [
        "uv",
        "run",
        "--project",
        str(project_root),
        "coding-assistant-mcp",
        "--transport",
        "streamable-http",
        "--host",
        host,
        "--port",
        str(port),
    ]

    handle = await start_process(cmd)

    try:
        # Give the server a moment to start and verify it's up
        started = False
        for _ in range(40):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, timeout=1.0)
                    if resp.status_code < 500:
                        started = True
                        break
            except Exception:
                pass
            await asyncio.sleep(0.5)
            if not handle.is_running:
                pytest.fail(f"Server exited prematurely with code {handle.exit_code}\nOutput: {handle.stdout}")

        if not started:
            pytest.fail(f"Real MCP server failed to start via 'uv run'\nOutput: {handle.stdout}")

        # Now we use a Client to call the python_execute tool on the real server
        async with Client(url) as client:
            # We want to test if the python_tool on the server can call the shell_tool on the same server
            test_script = """
import asyncio
import os
from fastmcp import Client

async def run_test():
    url = os.environ.get("MCP_SERVER_URL")
    async with Client(url) as client:
        # Call shell_execute
        result = await client.call_tool("shell_execute", {"command": "echo 'Real uv run Success'"})
        text = "\\n".join([c.text for c in result.content if hasattr(c, "text")])
        print(f"RESULT: {text.strip()}")

asyncio.run(run_test())
"""
            # Call tool (prefixed with 'python')
            result = await client.call_tool("python_execute", {"code": test_script})

            output = "\n".join([c.text for c in result.content if hasattr(c, "text")])
            assert "RESULT: Real uv run Success" in output

    finally:
        await handle.terminate()
