import pytest
import asyncio
from coding_assistant_mcp.shell import execute as shell_execute
from coding_assistant_mcp.python import execute as python_execute
from coding_assistant_mcp.tasks import (
    list_tasks as list_tasks_tool,
    get_output as get_output_tool,
    kill_task as kill_task_tool,
    remove_task as remove_task_tool,
    manager
)

@pytest.fixture(autouse=True)
def clean_manager():
    # Clear tasks between tests
    for task_id in list(manager._tasks.keys()):
        manager.remove_task(task_id)
    manager._next_id = 1
    manager._task_order = []
    yield

@pytest.mark.asyncio
async def test_background_explicit():
    res = await shell_execute(command="sleep 0.1; echo 'done'", background=True)
    assert "Task started in background with ID: 1" in res
    
    tasks = await list_tasks_tool.fn()
    assert "ID: 1" in tasks
    
    out = await get_output_tool.fn(task_id=1, wait=True, timeout=5)
    assert "done" in out

@pytest.mark.asyncio
async def test_all_tasks_registered():
    # Run a sync task
    res = await shell_execute(command="echo 'sync task'")
    assert res.strip() == "sync task"
    
    # It should be in the task manager anyway
    tasks = await list_tasks_tool.fn()
    assert "ID: 1" in tasks
    
    # Retrieve it
    out = await get_output_tool.fn(task_id=1)
    assert "sync task" in out

@pytest.mark.asyncio
async def test_truncation_note_with_id():
    # Trigger truncation
    res = await shell_execute(command="echo '1234567890'", truncate_at=5)
    assert "truncated" in res
    assert "Full output available via `tasks_get_output(task_id=1)`" in res
    
    # Get full output
    full = await get_output_tool.fn(task_id=1)
    assert "1234567890" in full

@pytest.mark.asyncio
async def test_auto_cleanup():
    # Set limit very low for testing: keep 1 finished task
    manager._max_finished_tasks = 1
    
    await shell_execute(command="echo 'task 1'")
    # Task 1 is finished now.
    
    await shell_execute(command="echo 'task 2'")
    # Task 2 is finished. Cleanup should drop Task 1.
    
    tasks = await list_tasks_tool.fn()
    assert "ID: 1" not in tasks 
    assert "ID: 2" in tasks

@pytest.mark.asyncio
async def test_auto_cleanup_keeps_running():
    manager._max_finished_tasks = 1
    
    # Start a background task (running)
    await shell_execute(command="sleep 2", background=True) # ID 1
    
    # Run two sync tasks (finished)
    await shell_execute(command="echo 'task 2'") # ID 2
    await shell_execute(command="echo 'task 3'") # ID 3 
    
    tasks = await list_tasks_tool.fn()
    
    # ID 1 should still be there because it's running
    assert "ID: 1" in tasks
    # ID 2 should be gone because it was the oldest finished
    assert "ID: 2" not in tasks
    # ID 3 should be there
    assert "ID: 3" in tasks

@pytest.mark.asyncio
async def test_kill_task():
    res = await shell_execute(command="sleep 10", background=True)
    kill_res = await kill_task_tool.fn(task_id=1)
    assert "Task 1 has been terminated" in kill_res
    
    status = await get_output_tool.fn(task_id=1)
    assert "FINISHED" in status
