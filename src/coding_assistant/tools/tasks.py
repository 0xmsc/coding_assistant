from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic import BaseModel, Field

from coding_assistant.tools.base import StructuredTool
from coding_assistant.tools.process import ProcessHandle, truncate_output
from coding_assistant.llm.types import Tool


@dataclass
class Task:
    """Tracked background process plus its display metadata."""

    id: int
    name: str
    handle: ProcessHandle


class TaskManager:
    """Track background subprocess tasks exposed through local tools."""

    def __init__(self, max_finished_tasks: int = 10) -> None:
        self._tasks: dict[int, Task] = {}
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
        current_finished = [task_id for task_id, task in self._tasks.items() if not task.handle.is_running]
        if len(current_finished) <= self._max_finished_tasks:
            return

        number_to_remove = len(current_finished) - self._max_finished_tasks
        for task_id in current_finished[:number_to_remove]:
            self.remove_task(task_id)

    def get_task(self, task_id: int) -> Task | None:
        """Return a tracked task by ID, if it still exists."""
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        """Return all tracked tasks in insertion order."""
        return list(self._tasks.values())

    def remove_task(self, task_id: int) -> None:
        """Remove a task and terminate its process in the background."""
        task = self._tasks.get(task_id)
        if task is None:
            return

        asyncio.get_running_loop().create_task(task.handle.terminate())
        del self._tasks[task_id]


class EmptyInput(BaseModel):
    """Schema for tools that do not take any arguments."""


class TasksGetOutputInput(BaseModel):
    task_id: int = Field(description="Numeric ID of the background task.")
    wait: bool = Field(
        default=False,
        description="If true, wait for the task to finish before reading output.",
    )
    timeout: int = Field(
        default=30,
        description="Maximum number of seconds to wait when `wait` is true.",
    )
    truncate_at: int = Field(
        default=50_000,
        description="Maximum number of characters to return from the task output.",
    )


class TaskIdInput(BaseModel):
    task_id: int = Field(description="Numeric ID of the background task.")


def create_task_tools(*, manager: TaskManager) -> list[Tool]:
    """Create tools for listing and managing background tasks."""

    async def list_tasks(_: EmptyInput) -> str:
        tasks = manager.list_tasks()
        if not tasks:
            return "No tasks found."

        lines: list[str] = []
        for task in tasks:
            status = "Running" if task.handle.is_running else f"Finished (Exit code: {task.handle.exit_code})"
            lines.append(f"ID: {task.id} | Name: {task.name} | Status: {status}")
        return "\n".join(lines)

    async def get_output(validated: TasksGetOutputInput) -> str:
        task = manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        if validated.wait:
            await task.handle.wait(timeout=validated.timeout)

        result = f"Task {validated.task_id} ({task.name})\n"
        if task.handle.is_running:
            result += "Status: running\n"
        else:
            result += f"Status: finished (Exit code: {task.handle.exit_code})\n"

        output = truncate_output(task.handle.consume_text(), validated.truncate_at)
        return f"{result}\n\n{output}"

    async def get_status(validated: TaskIdInput) -> str:
        task = manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        status = "running" if task.handle.is_running else f"finished (Exit code: {task.handle.exit_code})"
        return f"Task {validated.task_id} ({task.name}) | Status: {status}"

    async def kill_task(validated: TaskIdInput) -> str:
        task = manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        await task.handle.terminate()
        return f"Task {validated.task_id} has been terminated."

    async def remove_task(validated: TaskIdInput) -> str:
        task = manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        manager.remove_task(validated.task_id)
        return f"Task {validated.task_id} removed from history."

    return [
        StructuredTool(
            name="tasks_list_tasks",
            description="List all tracked background tasks with their IDs and current status.",
            schema_model=EmptyInput,
            handler=list_tasks,
        ),
        StructuredTool(
            name="tasks_get_output",
            description="Read the captured output for one background task.",
            schema_model=TasksGetOutputInput,
            handler=get_output,
        ),
        StructuredTool(
            name="tasks_get_status",
            description="Get the current status of one background task without reading its output.",
            schema_model=TaskIdInput,
            handler=get_status,
        ),
        StructuredTool(
            name="tasks_kill_task",
            description="Terminate a running background task.",
            schema_model=TaskIdInput,
            handler=kill_task,
        ),
        StructuredTool(
            name="tasks_remove_task",
            description="Remove a task from the retained task history.",
            schema_model=TaskIdInput,
            handler=remove_task,
        ),
    ]
