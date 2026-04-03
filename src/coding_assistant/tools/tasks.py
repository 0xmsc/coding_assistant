from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool
from coding_assistant.tools.process import ProcessHandle, truncate_output


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


def _format_task_status(task: Task) -> str:
    """Return the user-visible status string for one tracked task."""
    if task.handle.is_running:
        return "running"
    return f"finished (Exit code: {task.handle.exit_code})"


class TasksListTasksTool(Tool):
    """List all tracked background tasks."""

    def __init__(self, *, manager: TaskManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "tasks_list_tasks"

    def description(self) -> str:
        return "List all tracked background tasks with their IDs and current status."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        EmptyInput.model_validate(parameters)
        tasks = self._manager.list_tasks()
        if not tasks:
            return "No tasks found."

        lines: list[str] = []
        for task in tasks:
            status = "Running" if task.handle.is_running else f"Finished (Exit code: {task.handle.exit_code})"
            lines.append(f"ID: {task.id} | Name: {task.name} | Status: {status}")
        return "\n".join(lines)


class TasksGetOutputTool(Tool):
    """Read the captured output for one background task."""

    def __init__(self, *, manager: TaskManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "tasks_get_output"

    def description(self) -> str:
        return "Read the captured output for one background task."

    def parameters(self) -> dict[str, Any]:
        return TasksGetOutputInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = TasksGetOutputInput.model_validate(parameters)
        task = self._manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        if validated.wait:
            await task.handle.wait(timeout=validated.timeout)

        result = f"Task {validated.task_id} ({task.name})\n"
        result += f"Status: {_format_task_status(task)}\n"

        output = truncate_output(task.handle.consume_text(), validated.truncate_at)
        return f"{result}\n\n{output}"


class TasksGetStatusTool(Tool):
    """Return the status of one background task without consuming output."""

    def __init__(self, *, manager: TaskManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "tasks_get_status"

    def description(self) -> str:
        return "Get the current status of one background task without reading its output."

    def parameters(self) -> dict[str, Any]:
        return TaskIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = TaskIdInput.model_validate(parameters)
        task = self._manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        return f"Task {validated.task_id} ({task.name}) | Status: {_format_task_status(task)}"


class TasksKillTaskTool(Tool):
    """Terminate a running background task."""

    def __init__(self, *, manager: TaskManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "tasks_kill_task"

    def description(self) -> str:
        return "Terminate a running background task."

    def parameters(self) -> dict[str, Any]:
        return TaskIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = TaskIdInput.model_validate(parameters)
        task = self._manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        await task.handle.terminate()
        return f"Task {validated.task_id} has been terminated."


class TasksRemoveTaskTool(Tool):
    """Remove one task from the retained task history."""

    def __init__(self, *, manager: TaskManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "tasks_remove_task"

    def description(self) -> str:
        return "Remove a task from the retained task history."

    def parameters(self) -> dict[str, Any]:
        return TaskIdInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = TaskIdInput.model_validate(parameters)
        task = self._manager.get_task(validated.task_id)
        if task is None:
            return f"Error: Task {validated.task_id} not found."

        self._manager.remove_task(validated.task_id)
        return f"Task {validated.task_id} removed from history."


def create_task_tools(*, manager: TaskManager) -> list[Tool]:
    """Create tools for listing and managing background tasks."""
    return [
        TasksListTasksTool(manager=manager),
        TasksGetOutputTool(manager=manager),
        TasksGetStatusTool(manager=manager),
        TasksKillTaskTool(manager=manager),
        TasksRemoveTaskTool(manager=manager),
    ]
