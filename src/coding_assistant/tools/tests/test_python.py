import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from coding_assistant.llm.types import Tool
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.tasks import TaskManager


@pytest_asyncio.fixture
async def manager() -> AsyncIterator[TaskManager]:
    task_manager = TaskManager()
    yield task_manager
    for task in task_manager.list_tasks():
        await task.handle.terminate()


@pytest.fixture
def execute(manager: TaskManager) -> Tool:
    return create_python_tools(manager=manager)[0]


@pytest.mark.asyncio
async def test_python_run_timeout(execute: Tool) -> None:
    out = await execute.execute({"code": "import time; time.sleep(2)", "timeout": 1})
    assert "taking longer than 1s" in out
    assert "Task ID: 1" in out


@pytest.mark.asyncio
async def test_python_run_exception_includes_traceback(execute: Tool) -> None:
    out = await execute.execute({"code": "import sys; sys.exit(7)"})
    assert out.startswith("Exception (exit code 7):\n\n")


@pytest.mark.asyncio
async def test_python_run_truncates_output(execute: Tool) -> None:
    out = await execute.execute({"code": "print('x'*1000)", "truncate_at": 200})
    assert "[truncated output at: " in out
    assert "Full output available" in out


@pytest.mark.asyncio
async def test_python_run_happy_path_stdout(execute: Tool) -> None:
    out = await execute.execute({"code": "print('hello', end='')", "timeout": 5})
    assert out == "hello"


@pytest.mark.asyncio
async def test_python_run_stderr_captured_with_zero_exit(execute: Tool) -> None:
    out = await execute.execute({"code": "import sys; sys.stderr.write('oops\\n')"})
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_python_run_with_dependencies(execute: Tool) -> None:
    code = """
# /// script
# dependencies = ["cowsay"]
# ///
import cowsay
cowsay.cow("moo")
"""
    out = await execute.execute({"code": code, "timeout": 60})
    assert "moo" in out
    assert "^__^" in out


@pytest.mark.asyncio
async def test_python_run_exception_with_stderr_content(execute: Tool) -> None:
    out = await execute.execute({"code": "import sys; sys.stderr.write('bad\\n'); sys.exit(4)"})
    assert out.startswith("Exception (exit code 4):\n\n")
    assert "bad\n" in out


@pytest.mark.asyncio
async def test_python_execute_cancellation_terminates_foreground_process(execute: Tool, manager: TaskManager) -> None:
    task = asyncio.create_task(execute.execute({"code": "import time; time.sleep(10)"}))

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
