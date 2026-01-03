from typing import Any, cast
import asyncio
import pytest
import pytest_asyncio
from coding_assistant.mcp.shell import create_shell_server
from coding_assistant.mcp.tasks import create_task_server, TaskManager


@pytest.fixture
def manager() -> Any:
    return TaskManager()


@pytest_asyncio.fixture
async def shell_execute(manager: Any) -> Any:
    server = create_shell_server(manager)
    return await server.get_tool("execute")


@pytest_asyncio.fixture
async def tasks_list_tasks(manager: Any) -> Any:
    server = create_task_server(manager)
    return await server.get_tool("list_tasks")


@pytest_asyncio.fixture
async def tasks_get_output(manager: Any) -> Any:
    server = create_task_server(manager)
    return await server.get_tool("get_output")


@pytest_asyncio.fixture
async def tasks_kill_task(manager: Any) -> Any:
    server = create_task_server(manager)
    return await server.get_tool("kill_task")


@pytest.mark.asyncio
async def test_background_explicit(shell_execute: Any, tasks_get_output: Any, tasks_list_tasks: Any) -> None:
    res = await cast(Any, shell_execute).fn(command="sleep 0.1; echo 'done'", background=True)
    assert "Task started in background with ID: 1" in res

    tasks = await cast(Any, tasks_list_tasks).fn()
    assert "ID: 1" in tasks

    out = await cast(Any, tasks_get_output).fn(task_id=1, wait=True, timeout=5)
    assert "done" in out


@pytest.mark.asyncio
async def test_all_tasks_registered(shell_execute: Any, tasks_list_tasks: Any, tasks_get_output: Any) -> None:
    res = await cast(Any, shell_execute).fn(command="echo 'sync task'")
    assert res.strip() == "sync task"

    tasks = await cast(Any, tasks_list_tasks).fn()
    assert "ID: 1" in tasks

    out = await cast(Any, tasks_get_output).fn(task_id=1)
    assert "sync task" in out


@pytest.mark.asyncio
async def test_truncation_note_with_id(shell_execute: Any, tasks_get_output: Any) -> None:
    res = await cast(Any, shell_execute).fn(command="echo '1234567890'", truncate_at=5)
    assert "truncated" in res
    assert "Full output available via `tasks_get_output(task_id=1)`" in res

    full = await cast(Any, tasks_get_output).fn(task_id=1)
    assert "1234567890" in full


@pytest.mark.asyncio
async def test_auto_cleanup(manager: Any) -> None:
    manager._max_finished_tasks = 1
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")
    task_server = create_task_server(manager)
    tasks_list_tasks_tool = await task_server.get_tool("list_tasks")

    await cast(Any, shell_execute_tool).fn(command="sleep 0.1")
    await cast(Any, shell_execute_tool).fn(command="sleep 0.1")
    await cast(Any, shell_execute_tool).fn(command="sleep 0.1")  # This will remove task 1

    tasks = await cast(Any, tasks_list_tasks_tool).fn()
    print(tasks)

    assert "ID: 1" not in tasks
    assert "ID: 2" in tasks
    assert "ID: 3" in tasks


@pytest.mark.asyncio
async def test_auto_cleanup_keeps_running(manager: Any) -> None:
    manager._max_finished_tasks = 1
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")
    task_server = create_task_server(manager)
    tasks_list_tasks_tool = await task_server.get_tool("list_tasks")

    await cast(Any, shell_execute_tool).fn(command="sleep 2", background=True)
    await cast(Any, shell_execute_tool).fn(command="sleep 0.2")
    await cast(Any, shell_execute_tool).fn(command="sleep 0.2")
    await cast(Any, shell_execute_tool).fn(command="sleep 0.2")

    tasks = await cast(Any, tasks_list_tasks_tool).fn()
    print(tasks)

    assert "ID: 1" in tasks
    assert "ID: 2" not in tasks
    assert "ID: 3" in tasks
    assert "ID: 4" in tasks


@pytest.mark.asyncio
async def test_cleanup_exactly_max_finished(manager: Any) -> None:
    manager._max_finished_tasks = 5
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")

    for i in range(10):
        await cast(Any, shell_execute_tool).fn(command=f"sleep 0.05; echo 'task {i + 1}'")

    await asyncio.sleep(0.1)
    await cast(Any, shell_execute_tool).fn(command="echo 'task 11'")  # Trigger cleanup

    tasks = manager.list_tasks()
    finished_tasks = [t for t in tasks if not t.handle.is_running]

    assert len(finished_tasks) == 6

    task_ids = [t.id for t in finished_tasks]
    assert task_ids == [6, 7, 8, 9, 10, 11]


@pytest.mark.asyncio
async def test_kill_task(shell_execute: Any, tasks_kill_task: Any, tasks_get_output: Any) -> None:
    await cast(Any, shell_execute).fn(command="sleep 10", background=True)
    kill_res = await cast(Any, tasks_kill_task).fn(task_id=1)
    assert "Task 1 has been terminated" in kill_res

    status = await cast(Any, tasks_get_output).fn(task_id=1)
    assert "finished" in status


@pytest_asyncio.fixture
async def tasks_get_status(manager: Any) -> Any:
    server = create_task_server(manager)
    return await server.get_tool("get_status")


@pytest.mark.asyncio
async def test_get_status(shell_execute: Any, tasks_get_status: Any) -> None:
    # Test running status
    await cast(Any, shell_execute).fn(command="sleep 0.5", background=True)
    status_running = await cast(Any, tasks_get_status).fn(task_id=1)
    assert "Status: running" in status_running

    # Test finished status
    await asyncio.sleep(0.6)
    status_finished = await cast(Any, tasks_get_status).fn(task_id=1)
    assert "Status: finished (Exit code: 0)" in status_finished

    # Test non-existent task
    status_missing = await cast(Any, tasks_get_status).fn(task_id=999)
    assert "Error: Task 999 not found" in status_missing


@pytest.mark.asyncio
async def test_incremental_output(shell_execute: Any, tasks_get_output: Any) -> None:
    # Run a command that produces output over time
    cmd = "echo 'line 1'; sleep 0.2; echo 'line 2'; sleep 0.2; echo 'line 3'"
    await cast(Any, shell_execute).fn(command=cmd, background=True)

    # Wait for the first line
    await asyncio.sleep(0.1)
    out1 = await cast(Any, tasks_get_output).fn(task_id=1)
    assert "line 1" in out1
    assert "line 2" not in out1
    assert "line 3" not in out1

    # Wait for the second line
    await asyncio.sleep(0.2)
    out2 = await cast(Any, tasks_get_output).fn(task_id=1)
    # The header still contains the task name (which might have 'line 1'),
    # but the actual output part should only have 'line 2'
    assert "line 1" not in out2[out2.find("\n\n") + 2 :]
    assert "line 2" in out2
    assert "line 3" not in out2

    # Wait for the third line
    await asyncio.sleep(0.2)
    out3 = await cast(Any, tasks_get_output).fn(task_id=1)
    assert "line 1" not in out3[out3.find("\n\n") + 2 :]
    assert "line 2" not in out3[out3.find("\n\n") + 2 :]
    assert "line 3" in out3

    # Another call should return nothing new
    out4 = await cast(Any, tasks_get_output).fn(task_id=1)
    actual_output = out4[out4.find("\n\n") + 2 :]
    assert actual_output.strip() == ""
