import pytest
import pytest_asyncio
from coding_assistant.mcp.shell import create_shell_server
from coding_assistant.mcp.tasks import TaskManager


@pytest.fixture
def manager():
    return TaskManager()


@pytest_asyncio.fixture
async def execute(manager):
    server = create_shell_server(manager)
    return await server.get_tool("execute")


@pytest.mark.asyncio
async def test_shell_execute_timeout(execute):
    out = await execute.fn(command="echo 'start'; sleep 2; echo 'end'", timeout=1)
    assert "taking longer than 1s" in out
    assert "Task ID: 1" in out


@pytest.mark.asyncio
async def test_shell_execute_nonzero_exit_code(execute):
    out = await execute.fn(command="bash -lc 'exit 7'")
    assert out.startswith("Exit code: 7.\n\n")


@pytest.mark.asyncio
async def test_shell_execute_truncates_output(execute):
    out = await execute.fn(command="yes 1 | head -c 1000", truncate_at=200)
    assert "[truncated output at: " in out
    assert len(out) > 10


@pytest.mark.asyncio
async def test_shell_execute_happy_path_stdout(execute):
    out = await execute.fn(command="printf 'hello'", timeout=5)
    assert out == "hello"


@pytest.mark.asyncio
async def test_shell_execute_stderr_captured_with_zero_exit(execute):
    out = await execute.fn(command="echo 'oops' >&2; true", timeout=5)
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_shell_execute_nonzero_with_stderr_content(execute):
    out = await execute.fn(command="echo 'bad' >&2; exit 4", timeout=5)
    assert out.startswith("Exit code: 4.\n\n")
    assert "bad\n" in out


@pytest.mark.asyncio
async def test_shell_execute_echo(execute):
    out = await execute.fn(command="echo bar")
    assert out == "bar\n"
