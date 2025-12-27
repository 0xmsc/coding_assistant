import pytest
import pytest_asyncio
from coding_assistant.mcp.python import create_python_server
from coding_assistant.mcp.tasks import TaskManager


@pytest.fixture
def manager():
    return TaskManager()


@pytest_asyncio.fixture
async def execute(manager):
    server = create_python_server(manager)
    return await server.get_tool("execute")


@pytest.mark.asyncio
async def test_python_run_timeout(execute):
    out = await execute.fn(code="import time; time.sleep(2)", timeout=1)
    assert "taking longer than 1s" in out
    assert "Task ID: 1" in out


@pytest.mark.asyncio
async def test_python_run_exception_includes_traceback(execute):
    out = await execute.fn(code="import sys; sys.exit(7)")
    assert out.startswith("Exception (exit code 7):\n\n")


@pytest.mark.asyncio
async def test_python_run_truncates_output(execute):
    out = await execute.fn(code="print('x'*1000)", truncate_at=200)
    assert "[truncated output at: " in out
    assert "Full output available" in out


@pytest.mark.asyncio
async def test_python_run_happy_path_stdout(execute):
    out = await execute.fn(code="print('hello', end='')", timeout=5)
    assert out == "hello"


@pytest.mark.asyncio
async def test_python_run_stderr_captured_with_zero_exit(execute):
    out = await execute.fn(code="import sys; sys.stderr.write('oops\\n')")
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_python_run_with_dependencies(execute):
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
async def test_python_run_exception_with_stderr_content(execute):
    out = await execute.fn(code="import sys; sys.stderr.write('bad\\n'); sys.exit(4)")
    assert out.startswith("Exception (exit code 4):\n\n")
    assert "bad\n" in out
