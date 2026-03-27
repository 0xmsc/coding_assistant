import asyncio
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from coding_assistant.llm.types import Tool
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.tasks import TaskManager, create_task_tools


@pytest_asyncio.fixture
async def manager() -> AsyncIterator[TaskManager]:
    task_manager = TaskManager()
    yield task_manager
    for task in task_manager.list_tasks():
        await task.handle.terminate()


def _get_tool(tools: list[Tool], name: str) -> Tool:
    return next(tool for tool in tools if tool.name() == name)


@pytest.fixture
def shell_execute(manager: TaskManager) -> Tool:
    return create_shell_tools(manager=manager)[0]


@pytest.fixture
def tasks_list_tasks(manager: TaskManager) -> Tool:
    return _get_tool(create_task_tools(manager=manager), "tasks_list_tasks")


@pytest.fixture
def tasks_get_output(manager: TaskManager) -> Tool:
    return _get_tool(create_task_tools(manager=manager), "tasks_get_output")


@pytest.fixture
def tasks_kill_task(manager: TaskManager) -> Tool:
    return _get_tool(create_task_tools(manager=manager), "tasks_kill_task")


@pytest.mark.asyncio
async def test_background_explicit(shell_execute: Tool, tasks_get_output: Tool, tasks_list_tasks: Tool) -> None:
    res = await shell_execute.execute({"command": "sleep 0.1; echo 'done'", "background": True})
    assert "Task started in background with ID: 1" in res

    tasks = await tasks_list_tasks.execute({})
    assert "ID: 1" in tasks

    out = await tasks_get_output.execute({"task_id": 1, "wait": True, "timeout": 5})
    assert "done" in out


@pytest.mark.asyncio
async def test_all_tasks_registered(shell_execute: Tool, tasks_list_tasks: Tool, tasks_get_output: Tool) -> None:
    res = await shell_execute.execute({"command": "echo 'sync task'"})
    assert res.strip() == "sync task"

    tasks = await tasks_list_tasks.execute({})
    assert "ID: 1" in tasks

    out = await tasks_get_output.execute({"task_id": 1})
    assert "sync task" in out


@pytest.mark.asyncio
async def test_truncation_note_with_id(shell_execute: Tool, tasks_get_output: Tool) -> None:
    res = await shell_execute.execute({"command": "echo '1234567890'", "truncate_at": 5})
    assert "truncated" in res
    assert "Full output available via `tasks_get_output(task_id=1)`" in res

    full = await tasks_get_output.execute({"task_id": 1})
    assert "1234567890" in full


@pytest.mark.asyncio
async def test_auto_cleanup(manager: TaskManager) -> None:
    manager._max_finished_tasks = 1
    shell_execute_tool = create_shell_tools(manager=manager)[0]
    tasks_list_tasks_tool = _get_tool(create_task_tools(manager=manager), "tasks_list_tasks")

    await shell_execute_tool.execute({"command": "sleep 0.1"})
    await shell_execute_tool.execute({"command": "sleep 0.1"})
    await shell_execute_tool.execute({"command": "sleep 0.1"})  # This will remove task 1

    tasks = await tasks_list_tasks_tool.execute({})
    print(tasks)

    assert "ID: 1" not in tasks
    assert "ID: 2" in tasks
    assert "ID: 3" in tasks


@pytest.mark.asyncio
async def test_auto_cleanup_keeps_running(manager: TaskManager) -> None:
    manager._max_finished_tasks = 1
    shell_execute_tool = create_shell_tools(manager=manager)[0]
    tasks_list_tasks_tool = _get_tool(create_task_tools(manager=manager), "tasks_list_tasks")

    await shell_execute_tool.execute({"command": "sleep 2", "background": True})
    await shell_execute_tool.execute({"command": "sleep 0.2"})
    await shell_execute_tool.execute({"command": "sleep 0.2"})
    await shell_execute_tool.execute({"command": "sleep 0.2"})

    tasks = await tasks_list_tasks_tool.execute({})
    print(tasks)

    assert "ID: 1" in tasks
    assert "ID: 2" not in tasks
    assert "ID: 3" in tasks
    assert "ID: 4" in tasks


@pytest.mark.asyncio
async def test_cleanup_exactly_max_finished(manager: TaskManager) -> None:
    manager._max_finished_tasks = 5
    shell_execute_tool = create_shell_tools(manager=manager)[0]

    for i in range(10):
        await shell_execute_tool.execute({"command": f"sleep 0.05; echo 'task {i + 1}'"})

    await asyncio.sleep(0.1)
    await shell_execute_tool.execute({"command": "echo 'task 11'"})  # Trigger cleanup

    tasks = manager.list_tasks()
    finished_tasks = [t for t in tasks if not t.handle.is_running]

    assert len(finished_tasks) == 6

    task_ids = [t.id for t in finished_tasks]
    assert task_ids == [6, 7, 8, 9, 10, 11]


@pytest.mark.asyncio
async def test_kill_task(shell_execute: Tool, tasks_kill_task: Tool, tasks_get_output: Tool) -> None:
    await shell_execute.execute({"command": "sleep 10", "background": True})
    kill_res = await tasks_kill_task.execute({"task_id": 1})
    assert "Task 1 has been terminated" in kill_res

    status = await tasks_get_output.execute({"task_id": 1})
    assert "finished" in status


@pytest.fixture
def tasks_get_status(manager: TaskManager) -> Tool:
    return _get_tool(create_task_tools(manager=manager), "tasks_get_status")


@pytest.mark.asyncio
async def test_get_status(shell_execute: Tool, tasks_get_status: Tool) -> None:
    # Test running status
    await shell_execute.execute({"command": "sleep 0.5", "background": True})
    status_running = await tasks_get_status.execute({"task_id": 1})
    assert "Status: running" in status_running

    # Test finished status
    await asyncio.sleep(0.6)
    status_finished = await tasks_get_status.execute({"task_id": 1})
    assert "Status: finished (Exit code: 0)" in status_finished

    # Test non-existent task
    status_missing = await tasks_get_status.execute({"task_id": 999})
    assert "Error: Task 999 not found" in status_missing


@pytest.mark.asyncio
async def test_incremental_output(shell_execute: Tool, tasks_get_output: Tool) -> None:
    # Run a command that produces output over time
    cmd = "echo 'line 1'; sleep 0.2; echo 'line 2'; sleep 0.2; echo 'line 3'"
    await shell_execute.execute({"command": cmd, "background": True})

    # Wait for the first line
    await asyncio.sleep(0.1)
    out1 = await tasks_get_output.execute({"task_id": 1})
    assert "line 1" in out1
    assert "line 2" not in out1
    assert "line 3" not in out1

    # Wait for the second line
    await asyncio.sleep(0.2)
    out2 = await tasks_get_output.execute({"task_id": 1})
    # The header still contains the task name (which might have 'line 1'),
    # but the actual output part should only have 'line 2'
    assert "line 1" not in out2[out2.find("\n\n") + 2 :]
    assert "line 2" in out2
    assert "line 3" not in out2

    # Wait for the third line
    await asyncio.sleep(0.2)
    out3 = await tasks_get_output.execute({"task_id": 1})
    assert "line 1" not in out3[out3.find("\n\n") + 2 :]
    assert "line 2" not in out3[out3.find("\n\n") + 2 :]
    assert "line 3" in out3

    # Another call should return nothing new
    out4 = await tasks_get_output.execute({"task_id": 1})
    actual_output = out4[out4.find("\n\n") + 2 :]
    assert actual_output.strip() == ""
