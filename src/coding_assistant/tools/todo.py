from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool


@dataclass
class Todo:
    """One tracked TODO item for the in-memory task list."""

    id: int
    description: str
    completed: bool = False
    result: str | None = None


class TodoManager:
    """Mutable in-memory TODO list exposed through local tools."""

    def __init__(self) -> None:
        self._todos: dict[int, Todo] = {}
        self._next_id = 1

    def format(self) -> str:
        """Render the current TODO list as markdown-style checkboxes."""
        lines: list[str] = []
        for todo in self._todos.values():
            box = "x" if todo.completed else " "
            if todo.result:
                lines.append(f"- [{box}] {todo.id}: {todo.description} -> {todo.result}")
            else:
                lines.append(f"- [{box}] {todo.id}: {todo.description}")
        return "\n".join(lines)

    def add(self, descriptions: list[str]) -> str:
        """Add TODO items and return the updated list."""
        for description in descriptions:
            if not description:
                raise ValueError("Description must not be empty.")
            todo = Todo(id=self._next_id, description=description)
            self._todos[todo.id] = todo
            self._next_id += 1
        return self.format()

    def list_todos(self) -> str:
        """Return the formatted TODO list."""
        return self.format()

    def complete(self, task_id: int, result: str | None = None) -> str:
        """Mark a task complete and return the updated list or an error."""
        todo = self._todos.get(task_id)
        if todo is None:
            return f"TODO {task_id} not found."

        todo.completed = True
        if result:
            todo.result = result
        return self.format()


class TodoAddInput(BaseModel):
    descriptions: list[str] = Field(description="List of non-empty TODO description strings.")


class EmptyInput(BaseModel):
    """Schema for tools that do not take any arguments."""


class TodoCompleteInput(BaseModel):
    task_id: int = Field(description="ID of the TODO item to mark complete.")
    result: str | None = Field(
        default=None,
        description="Optional one-line result to attach to the completed item.",
    )


class TodoAddTool(Tool):
    """Add items to the in-memory TODO list."""

    def __init__(self, *, manager: TodoManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "todo_add"

    def description(self) -> str:
        return "Add one or more TODO items and return the updated TODO list."

    def parameters(self) -> dict[str, Any]:
        return TodoAddInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = TodoAddInput.model_validate(parameters)
        return TextToolResult(content=self._manager.add(validated.descriptions))


class TodoListTool(Tool):
    """Return the current TODO list."""

    def __init__(self, *, manager: TodoManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "todo_list_todos"

    def description(self) -> str:
        return "Return the current TODO list."

    def parameters(self) -> dict[str, Any]:
        return EmptyInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        EmptyInput.model_validate(parameters)
        return TextToolResult(content=self._manager.list_todos())


class TodoCompleteTool(Tool):
    """Mark one TODO item complete."""

    def __init__(self, *, manager: TodoManager) -> None:
        self._manager = manager

    def name(self) -> str:
        return "todo_complete"

    def description(self) -> str:
        return "Mark a TODO item complete and optionally record its result."

    def parameters(self) -> dict[str, Any]:
        return TodoCompleteInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = TodoCompleteInput.model_validate(parameters)
        return TextToolResult(content=self._manager.complete(validated.task_id, validated.result))


def create_todo_tools(*, manager: TodoManager) -> list[Tool]:
    """Create tools for managing the in-memory TODO list."""
    return [
        TodoAddTool(manager=manager),
        TodoListTool(manager=manager),
        TodoCompleteTool(manager=manager),
    ]
