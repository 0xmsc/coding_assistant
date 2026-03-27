from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from coding_assistant.tools.base import StructuredTool
from coding_assistant.llm.types import Tool


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


def create_todo_tools(*, manager: TodoManager) -> list[Tool]:
    """Create tools for managing the in-memory TODO list."""

    async def add(validated: TodoAddInput) -> str:
        return manager.add(validated.descriptions)

    async def list_todos(_: EmptyInput) -> str:
        return manager.list_todos()

    async def complete(validated: TodoCompleteInput) -> str:
        return manager.complete(validated.task_id, validated.result)

    return [
        StructuredTool(
            name="todo_add",
            description="Add one or more TODO items and return the updated TODO list.",
            schema_model=TodoAddInput,
            handler=add,
        ),
        StructuredTool(
            name="todo_list_todos",
            description="Return the current TODO list.",
            schema_model=EmptyInput,
            handler=list_todos,
        ),
        StructuredTool(
            name="todo_complete",
            description="Mark a TODO item complete and optionally record its result.",
            schema_model=TodoCompleteInput,
            handler=complete,
        ),
    ]
