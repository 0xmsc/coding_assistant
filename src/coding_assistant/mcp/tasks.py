from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict

from fastmcp import FastMCP

from coding_assistant.mcp.proc import ProcessHandle
from coding_assistant.mcp.utils import truncate_output


@dataclass
class Task:
    id: int
    name: str
    handle: ProcessHandle


class TaskManager:
    def __init__(self, max_finished_tasks: int = 10):
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._max_finished_tasks = max_finished_tasks

    def register_task(self, name: str, handle: ProcessHandle) -> int:
        task_id = self._next_id
        self._next_id += 1
        self._tasks[task_id] = Task(id=task_id, name=name, handle=handle)
        self._cleanup_finished_tasks()
        return task_id

    def _cleanup_finished_tasks(self):
        current_finished = [tid for tid, task in self._tasks.items() if not task.handle.is_running]
        if len(current_finished) > self._max_finished_tasks:
            num_to_remove = len(current_finished) - self._max_finished_tasks
            to_remove = current_finished[:num_to_remove]
            for tid in to_remove:
                self.remove_task(tid)

    def get_task(self, task_id: int) -> Task | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        return list(self._tasks.values())

    def remove_task(self, task_id: int):
        if task_id in self._tasks:
            task = self._tasks[task_id]
            loop = asyncio.get_running_loop()
            loop.create_task(task.handle.terminate())
            del self._tasks[task_id]


def create_task_server(manager: TaskManager) -> FastMCP:
    task_server = FastMCP("TaskManager")

    @task_server.tool()
    async def list_tasks() -> str:
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
        output = task.handle.stdout
        result += truncate_output(output, truncate_at)

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

    return task_server
