import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from coding_assistant.llm.types import TextToolResult, Tool, ToolResult
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.tasks import TaskManager


@pytest_asyncio.fixture
async def manager() -> AsyncIterator[TaskManager]:
    task_manager = TaskManager()
    yield task_manager
    for task in task_manager.list_tasks():
        await task.handle.terminate()


@pytest.fixture
def execute(manager: TaskManager) -> Tool:
    return create_shell_tools(manager=manager)[0]


def _text(result: ToolResult) -> str:
    assert isinstance(result, TextToolResult)
    return result.content


@pytest.mark.asyncio
async def test_shell_execute_timeout(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "echo 'start'; sleep 2; echo 'end'", "timeout": 1}))
    assert "taking longer than 1s" in out
    assert "Task ID: 1" in out


@pytest.mark.asyncio
async def test_shell_execute_nonzero_exit_code(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "bash -lc 'exit 7'"}))
    assert out.startswith("Exit code: 7.\n\n")


@pytest.mark.asyncio
async def test_shell_execute_truncates_output(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "yes 1 | head -c 1000", "truncate_at": 200}))
    assert "[truncated output at: " in out
    assert len(out) > 10


@pytest.mark.asyncio
async def test_shell_execute_happy_path_stdout(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "printf 'hello'", "timeout": 5}))
    assert out == "hello"


@pytest.mark.asyncio
async def test_shell_execute_stderr_captured_with_zero_exit(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "echo 'oops' >&2; true", "timeout": 5}))
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_shell_execute_nonzero_with_stderr_content(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "echo 'bad' >&2; exit 4", "timeout": 5}))
    assert out.startswith("Exit code: 4.\n\n")
    assert "bad\n" in out


@pytest.mark.asyncio
async def test_shell_execute_echo(execute: Tool) -> None:
    out = _text(await execute.execute({"command": "echo bar"}))
    assert out == "bar\n"


@pytest.mark.asyncio
async def test_shell_execute_cancellation_terminates_foreground_process(execute: Tool, manager: TaskManager) -> None:
    task = asyncio.create_task(execute.execute({"command": "sleep 10"}))

    tracked_task = None
    for _ in range(20):
        tracked_task = manager.get_task(1)
        if tracked_task is not None:
            break
        await asyncio.sleep(0.05)

    assert tracked_task is not None
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert await tracked_task.handle.wait(timeout=1.0) is True
    assert tracked_task.handle.is_running is False
