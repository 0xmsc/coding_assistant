import pytest
import pytest_asyncio
import os
from coding_assistant_mcp.python import create_python_server
from coding_assistant_mcp.tasks import TaskManager

@pytest.fixture
def manager():
    return TaskManager()

@pytest.mark.asyncio
async def test_mcp_env_var_and_fastmcp_dep(manager):
    # Setup server with an MCP URL
    mcp_url = "http://localhost:8000/mcp"
    server = create_python_server(manager, mcp_url=mcp_url)
    execute = await server.get_tool("execute")
    
    # Run a script that checks if MCP_SERVER_URL is set and fastmcp can be imported
    code = """
import os
import sys

mcp_url = os.environ.get("MCP_SERVER_URL")
assert mcp_url == "http://localhost:8000/mcp"

try:
    from fastmcp import Client
    print("fastmcp imported")
except ImportError:
    print("fastmcp NOT imported")
"""
    out = await execute.fn(code=code)
    assert "fastmcp imported" in out

@pytest.mark.asyncio
async def test_no_mcp_env_var(manager):
    # Setup server WITHOUT an MCP URL
    server = create_python_server(manager, mcp_url=None)
    execute = await server.get_tool("execute")
    
    code = """
import os
print(f"MCP_SERVER_URL: {os.environ.get('MCP_SERVER_URL')}")
"""
    out = await execute.fn(code=code)
    assert "MCP_SERVER_URL: None" in out

@pytest.mark.asyncio
async def test_python_run_happy_path_stdout(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="print('hello', end='')", timeout=5)
    assert out == "hello"

@pytest.mark.asyncio
async def test_python_run_timeout(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="import time; time.sleep(2)", timeout=1)
    assert "taking longer than 1s" in out
    assert "Task ID" in out

@pytest.mark.asyncio
async def test_python_run_exception_includes_traceback(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="import sys; sys.exit(7)")
    assert out.startswith("Exception (exit code 7):\n\n")

@pytest.mark.asyncio
async def test_python_run_truncates_output(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="print('x'*1000)", truncate_at=200)
    assert "[truncated output at: " in out
    assert "Full output available" in out

@pytest.mark.asyncio
async def test_python_run_stderr_captured_with_zero_exit(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="import sys; sys.stderr.write('oops\\n')")
    assert out == "oops\n"

@pytest.mark.asyncio
async def test_python_run_with_dependencies(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    code = """
# /// script
# dependencies = ["cowsay"]
# ///
import cowsay
cowsay.cow("moo")
"""
    out = await execute.fn(code=code, timeout=60)
    assert "moo" in out
    assert "^__^" in out

@pytest.mark.asyncio
async def test_python_run_exception_with_stderr_content(manager):
    server = create_python_server(manager)
    execute = await server.get_tool("execute")
    out = await execute.fn(code="import sys; sys.stderr.write('bad\\n'); sys.exit(4)")
    assert out.startswith("Exception (exit code 4):\n\n")
    assert "bad\n" in out
