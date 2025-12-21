import pytest
import asyncio
from coding_assistant_mcp.shell import execute as shell_execute
from coding_assistant_mcp.bg_tasks import (
    list_tasks as list_tasks_tool,
    get_output as get_output_tool,
    kill_task as kill_task_tool,
    manager
)

@pytest.fixture(autouse=True)
def clean_manager():
    # Clear tasks between tests
    for task_id in list(manager._tasks.keys()):
        manager.remove_task(task_id)
    yield

@pytest.mark.asyncio
async def test_background_explicit():
    # Start a background task
    res = await shell_execute(command="sleep 1; echo 'done'", background=True)
    assert "Task started in background with ID:" in res
    task_id = int(res.split(":")[-1].strip())
    
    # Check it exists in list
    tasks = await list_tasks_tool.fn()
    assert f"ID: {task_id}" in tasks
    assert "Running" in tasks
    
    # Wait for it and get output
    out = await get_output_tool.fn(task_id=task_id, wait=True, timeout=5)
    assert "Status: FINISHED" in out
    assert "done" in out

@pytest.mark.asyncio
async def test_auto_background_on_timeout():
    # Run a command that will timeout
    res = await shell_execute(command="echo 'starting'; sleep 2; echo 'finished'", timeout=1)
    
    assert "taking longer than 1s" in res
    assert "background task with ID:" in res
    task_id = int(res.split(":")[-1].split(".")[0].strip())
    
    # Check partial output
    out = await get_output_tool.fn(task_id=task_id, wait=False)
    assert "starting" in out
    assert "finished" not in out
    
    # Wait for completion
    out = await get_output_tool.fn(task_id=task_id, wait=True, timeout=5)
    assert "finished" in out

@pytest.mark.asyncio
async def test_kill_task():
    res = await shell_execute(command="sleep 10", background=True)
    task_id = int(res.split(":")[-1].strip())
    
    kill_res = await kill_task_tool.fn(task_id=task_id)
    assert f"Task {task_id} has been terminated" in kill_res
    
    status = await get_output_tool.fn(task_id=task_id)
    assert "Status: FINISHED" in status # terminate() marks it finished
    assert "Return code:" in status
