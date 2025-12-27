import asyncio
import pytest
import pytest_asyncio
from coding_assistant.mcp.shell import create_shell_server
from coding_assistant.mcp.tasks import create_task_server, TaskManager


@pytest.fixture
def manager():
    return TaskManager()


@pytest_asyncio.fixture
async def shell_execute(manager):
    server = create_shell_server(manager)
    return await server.get_tool("execute")


@pytest_asyncio.fixture
async def tasks_list_tasks(manager):
    server = create_task_server(manager)
    return await server.get_tool("list_tasks")


@pytest_asyncio.fixture
async def tasks_get_output(manager):
    server = create_task_server(manager)
    return await server.get_tool("get_output")


@pytest_asyncio.fixture
async def tasks_kill_task(manager):
    server = create_task_server(manager)
    return await server.get_tool("kill_task")


@pytest.mark.asyncio
async def test_background_explicit(shell_execute, tasks_get_output, tasks_list_tasks):
    res = await shell_execute.fn(command="sleep 0.1; echo 'done'", background=True)
    assert "Task started in background with ID: 1" in res

    tasks = await tasks_list_tasks.fn()
    assert "ID: 1" in tasks

    out = await tasks_get_output.fn(task_id=1, wait=True, timeout=5)
    assert "done" in out


@pytest.mark.asyncio
async def test_all_tasks_registered(shell_execute, tasks_list_tasks, tasks_get_output):
    res = await shell_execute.fn(command="echo 'sync task'")
    assert res.strip() == "sync task"

    tasks = await tasks_list_tasks.fn()
    assert "ID: 1" in tasks

    out = await tasks_get_output.fn(task_id=1)
    assert "sync task" in out


@pytest.mark.asyncio
async def test_truncation_note_with_id(shell_execute, tasks_get_output):
    res = await shell_execute.fn(command="echo '1234567890'", truncate_at=5)
    assert "truncated" in res
    assert "Full output available via `tasks_get_output(task_id=1)`" in res

    full = await tasks_get_output.fn(task_id=1)
    assert "1234567890" in full


@pytest.mark.asyncio
async def test_auto_cleanup(manager):
    manager._max_finished_tasks = 1
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")
    task_server = create_task_server(manager)
    tasks_list_tasks_tool = await task_server.get_tool("list_tasks")

    await shell_execute_tool.fn(command="sleep 0.1")
    await shell_execute_tool.fn(command="sleep 0.1")
    await shell_execute_tool.fn(command="sleep 0.1")  # This will remove task 1

    tasks = await tasks_list_tasks_tool.fn()
    print(tasks)

    assert "ID: 1" not in tasks
    assert "ID: 2" in tasks
    assert "ID: 3" in tasks


@pytest.mark.asyncio
async def test_auto_cleanup_keeps_running(manager):
    manager._max_finished_tasks = 1
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")
    task_server = create_task_server(manager)
    tasks_list_tasks_tool = await task_server.get_tool("list_tasks")

    await shell_execute_tool.fn(command="sleep 2", background=True)
    await shell_execute_tool.fn(command="sleep 0.2")
    await shell_execute_tool.fn(command="sleep 0.2")
    await shell_execute_tool.fn(command="sleep 0.2")

    tasks = await tasks_list_tasks_tool.fn()
    print(tasks)

    assert "ID: 1" in tasks
    assert "ID: 2" not in tasks
    assert "ID: 3" in tasks
    assert "ID: 4" in tasks


@pytest.mark.asyncio
async def test_cleanup_exactly_max_finished(manager):
    manager._max_finished_tasks = 5
    shell_server = create_shell_server(manager)
    shell_execute_tool = await shell_server.get_tool("execute")

    for i in range(10):
        await shell_execute_tool.fn(command=f"sleep 0.5; echo 'task {i + 1}'")

    await asyncio.sleep(1)
    await shell_execute_tool.fn(command="echo 'task 11'")  # Trigger cleanup

    tasks = manager.list_tasks()
    finished_tasks = [t for t in tasks if not t.handle.is_running]

    assert len(finished_tasks) == 6

    task_ids = [t.id for t in finished_tasks]
    assert task_ids == [6, 7, 8, 9, 10, 11]


@pytest.mark.asyncio
async def test_kill_task(shell_execute, tasks_kill_task, tasks_get_output):
    await shell_execute.fn(command="sleep 10", background=True)
    kill_res = await tasks_kill_task.fn(task_id=1)
    assert "Task 1 has been terminated" in kill_res

    status = await tasks_get_output.fn(task_id=1)
    assert "finished" in status
