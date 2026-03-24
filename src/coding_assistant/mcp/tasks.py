from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict

from fastmcp import FastMCP

from coding_assistant.mcp.proc import ProcessHandle
from coding_assistant.mcp.utils import truncate_output


@dataclass
class Task:
    """Tracked background process plus its display metadata."""

    id: int
    name: str
    handle: ProcessHandle


class TaskManager:
    """Track background subprocess tasks exposed through MCP tools."""

    def __init__(self, max_finished_tasks: int = 10):
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._max_finished_tasks = max_finished_tasks

    def register_task(self, name: str, handle: ProcessHandle) -> int:
        """Register a new task and return its numeric identifier."""
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = Task(id=task_id, name=name, handle=handle)
        self._cleanup_finished_tasks()
        return task_id

    def _cleanup_finished_tasks(self) -> None:
        """Drop old finished tasks once the retention limit is exceeded."""
        current_finished = [tid for tid, task in self._tasks.items() if not task.handle.is_running]
        if len(current_finished) > self._max_finished_tasks:
            num_to_remove = len(current_finished) - self._max_finished_tasks
            to_remove = current_finished[:num_to_remove]
            for tid in to_remove:
                self.remove_task(tid)

    def get_task(self, task_id: int) -> Task | None:
        """Return a tracked task by ID, if it still exists."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        """Return all tracked tasks in insertion order."""
        return list(self._tasks.values())

    def remove_task(self, task_id: int) -> None:
        """Remove a task and terminate its process in the background."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            loop = asyncio.get_running_loop()
            loop.create_task(task.handle.terminate())
            del self._tasks[task_id]


def create_task_server(*, manager: TaskManager) -> FastMCP:
    """Create the MCP server for listing and managing background tasks."""
    task_server = FastMCP("TaskManager")

    @task_server.tool()
    async def list_tasks() -> str:
        """List all tracked tasks with their IDs and current status."""
        tasks = manager.list_tasks()
        if not tasks:
            return "No tasks found."

        lines = []
        for t in tasks:
            status = "Running" if t.handle.is_running else f"Finished (Exit code: {t.handle.exit_code})"
            lines.append(f"ID: {t.id} | Name: {t.name} | Status: {status}")

        return "\n".join(lines)

    @task_server.tool()
    async def get_output(
        task_id: int,
        wait: bool = False,
        timeout: int = 30,
        truncate_at: int = 50_000,
    ) -> str:
        """Return captured output for a task, optionally waiting for completion."""
        task = manager.get_task(task_id)
        if not task:
            return f"Error: Task {task_id} not found."

        if wait:
            await task.handle.wait(timeout=timeout)

        result = f"Task {task_id} ({task.name})\n"

        if task.handle.is_running:
            result += "Status: running\n"
        else:
            result += f"Status: finished (Exit code: {task.handle.exit_code})\n"

        result += "\n\n"
        output = task.handle.consume_text()
        result += truncate_output(output, truncate_at)

        return result

    @task_server.tool()
    async def get_status(task_id: int) -> str:
        """Get the current status of a task without its output."""
        task = manager.get_task(task_id)
        if not task:
            return f"Error: Task {task_id} not found."

        status = "running" if task.handle.is_running else f"finished (Exit code: {task.handle.exit_code})"
        return f"Task {task_id} ({task.name}) | Status: {status}"

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

    return task_server
