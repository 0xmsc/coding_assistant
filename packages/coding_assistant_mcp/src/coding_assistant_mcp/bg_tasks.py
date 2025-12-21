from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

from fastmcp import FastMCP

from coding_assistant_mcp.proc import ProcessHandle
from coding_assistant_mcp.utils import truncate_output

bg_server = FastMCP()


@dataclass
class BackgroundTask:
    id: int
    name: str
    handle: ProcessHandle
    start_time: datetime = field(default_factory=datetime.now)


class BackgroundTaskManager:
    def __init__(self):
        self._tasks: Dict[int, BackgroundTask] = {}
        self._next_id = 1

    def register_task(self, name: str, handle: ProcessHandle) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = BackgroundTask(id=task_id, name=name, handle=handle)
        return task_id

    def get_task(self, task_id: int) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def remove_task(self, task_id: int):
        if task_id in self._tasks:
            del self._tasks[task_id]


# Global manager instance
manager = BackgroundTaskManager()


@bg_server.tool()
async def list_tasks() -> str:
    """List all background tasks and their current status."""
    tasks = manager.list_tasks()
    if not tasks:
        return "No background tasks running."

    lines = []
    for t in tasks:
        status = "Running" if t.handle.is_running else f"Finished (Exit code: {t.handle.returncode})"
        lines.append(f"ID: {t.id} | Name: {t.name} | Status: {status} | Started: {t.start_time.strftime('%H:%M:%S')}")

    return "\n".join(lines)


@bg_server.tool()
async def get_output(
    task_id: int,
    wait: bool = False,
    timeout: int = 30,
    truncate_at: int = 50_000,
) -> str:
    """
    Get the output of a background task.
    If wait=True, it will block until the task finishes or the timeout is reached.
    """
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found."

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


@bg_server.tool()
async def kill_task(task_id: int) -> str:
    """Terminate a background task."""
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found."

    await task.handle.terminate()
    return f"Task {task_id} has been terminated."


@bg_server.tool()
async def remove_task(task_id: int) -> str:
    """Remove a background task from the manager. Use this to clean up finished tasks."""
    task = manager.get_task(task_id)
    if not task:
        return f"Error: Task {task_id} not found."

    if task.handle.is_running:
        return f"Error: Task {task_id} is still running. Kill it first."

    manager.remove_task(task_id)
    return f"Task {task_id} removed."
