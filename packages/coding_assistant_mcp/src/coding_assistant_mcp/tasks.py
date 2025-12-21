from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from fastmcp import FastMCP

from coding_assistant_mcp.proc import ProcessHandle
from coding_assistant_mcp.utils import truncate_output

task_server = FastMCP()


@dataclass
class Task:
    id: int
    name: str
    handle: ProcessHandle
    start_time: datetime = field(default_factory=datetime.now)


class TaskManager:
    def __init__(self, max_finished_tasks: int = 10):
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._max_finished_tasks = max_finished_tasks

    def register_task(self, name: str, handle: ProcessHandle) -> int:
        self._cleanup_finished_tasks()

        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = Task(id=task_id, name=name, handle=handle)
        return task_id

    def _cleanup_finished_tasks(self):
        """Keep only the last N finished tasks; never remove running ones."""
        # Refresh the finished tasks list based on current state
        current_finished = [
            tid for tid, task in self._tasks.items() 
            if not task.handle.is_running
        ]
        
        # Sort by ID (chronological) to ensure we keep the NEWEST finished ones
        current_finished.sort()

        # If we exceed the limit, remove the oldest finished ones
        if len(current_finished) > self._max_finished_tasks:
            to_remove = current_finished[:-self._max_finished_tasks]
            for tid in to_remove:
                self.remove_task(tid)

    def get_task(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        # Always run cleanup before listing to ensure we respect the limit
        self._cleanup_finished_tasks()
        return list(self._tasks.values())

    def remove_task(self, task_id: int):
        if task_id in self._tasks:
            task = self._tasks[task_id]
            if task.handle.is_running:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(task.handle.terminate())
                except RuntimeError:
                    # No active loop (likely during test cleanup)
                    pass
            del self._tasks[task_id]


# Global manager instance
manager = TaskManager()


@task_server.tool()
async def list_tasks() -> str:
    """List all recent tasks and their current status."""
    tasks = manager.list_tasks()
    if not tasks:
        return "No tasks found."

    lines = []
    # Show most recent first
    for t in reversed(tasks):
        status = "Running" if t.handle.is_running else f"Finished (Exit code: {t.handle.returncode})"
        lines.append(f"ID: {t.id} | Name: {t.name} | Status: {status} | Started: {t.start_time.strftime('%H:%M:%S')}")

    return "\n".join(lines)


@task_server.tool()
async def get_output(
    task_id: int,
    wait: bool = False,
    timeout: int = 30,
    truncate_at: int = 50_000,
) -> str:
    """
    Get the output of a task.
    If wait=True, it will block until the task finishes or the timeout is reached.
    Use this to retrieve full output if a previous tool call returned truncated results.
    """
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found. (It might have been cleaned up if it was old)"

    if wait and task.handle.is_running:
        await task.handle.wait(timeout=timeout)

    output = task.handle.stdout
    status = "RUNNING" if task.handle.is_running else "FINISHED"
    result = f"Task {task_id} ({task.name}) - Status: {status}\n"
    result += "-" * 20 + "\n"
    result += truncate_output(output, truncate_at)

    if not task.handle.is_running:
        result += f"\n\nReturn code: {task.handle.returncode}"

    return result


@task_server.tool()
async def kill_task(task_id: int) -> str:
    """Terminate a running task."""
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found."

    await task.handle.terminate()
    return f"Task {task_id} has been terminated."


@task_server.tool()
async def remove_task(task_id: int) -> str:
    """Remove a task from the manager history."""
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found."

    manager.remove_task(task_id)
    return f"Task {task_id} removed from history."
