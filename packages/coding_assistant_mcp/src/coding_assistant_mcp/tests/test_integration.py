import asyncio
import pytest
import httpx
import subprocess
from pathlib import Path


@pytest.mark.asyncio
async def test_python_mcp_integration_real_server():
    """
    Test that an agent can interact with a REAL MCP server instance started via 'uv run'.
    """
    port = 53675
    host = "localhost"
    url = f"http://{host}:{port}/mcp"

    # Start the REAL MCP server using uv run
    # Use Path(__file__) and --project to start the right MCP
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

    server_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Give the server a moment to start and verify it's up
    try:
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
            if server_process.poll() is not None:
                print(f"Server exited prematurely with code {server_process.returncode}")
                stdout, _ = server_process.communicate()
                print(f"STDOUT/STDERR: {stdout}")
                break

        if not started:
            pytest.fail("Real MCP server failed to start via 'uv run'")

        # Now we use a Client to call the python_execute tool on the real server
        from fastmcp import Client

        async with Client(url) as client:
            # We want to test if the python_execute tool on the server can call the shell_execute tool on the same server
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
            # Call python_execute (prefixed with 'python' if main.py adds prefixes)
            # In current main.py: await mcp.import_server(..., prefix="python")
            # So the tool name is "python_execute"
            result = await client.call_tool("python_execute", {"code": test_script})

            output = "\\n".join([c.text for c in result.content if hasattr(c, "text")])
            print(f"DEBUG OUTPUT:\n{output}")
            assert "RESULT: Real uv run Success" in output

    finally:
        server_process.terminate()
        server_process.wait(timeout=5)
        if server_process.poll() is None:
            server_process.kill()
