from __future__ import annotations


from dataclasses import dataclass
from typing import Optional
from fastmcp import FastMCP


@dataclass
class Todo:
    id: int
    description: str
    completed: bool = False


class TodoManager:
    def __init__(self) -> None:
        self._todos: dict[int, Todo] = dict()
        self._next_id = 1

    def add(self, description: str) -> Todo:
        todo = Todo(id=self._next_id, description=description)
        self._todos[todo.id] = todo
        self._next_id += 1
        return todo

    def complete(self, task_id: int) -> Todo | None:
        todo = self._todos.get(task_id)
        if todo:
            todo.completed = True
            return todo
        return None

    def format(
        self,
    ) -> str:
        lines: list[str] = []
        for t in self._todos.values():
            box = "x" if t.completed else " "
            lines.append(f"- [{box}] {t.id}: {t.description}")
        return "\n".join(lines)


_MANAGER: Optional[TodoManager] = None


def get_manager() -> TodoManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = TodoManager()
    return _MANAGER


def reset_state() -> None:
    """Reset TODO state."""
    global _MANAGER
    _MANAGER = None


def add(descriptions: list[str]) -> str:
    """Add one or more TODO items. Accepts a string or a list of strings."""
    manager = get_manager()
    for desc in descriptions:
        if not desc:
            raise ValueError("Description must not be empty.")
        todo = manager.add(desc)
    return manager.format()


def list_todos() -> str:
    """List all TODO items (or only pending), rendered as a markdown task list."""
    return get_manager().format()


def complete(task_id: int) -> str:
    """Mark a TODO item as completed by its ID, then show remaining items as a markdown task list."""
    manager = get_manager()

    result = ""
    if todo := manager.complete(task_id):
        result += f"Completed TODO {task_id}: {todo.description}\n"
    else:
        result += f"TODO {task_id} not found\n"

    result += "\n"
    result += manager.format()

    return result


todo_server = FastMCP()
todo_server.tool(complete)
todo_server.tool(list_todos)
todo_server.tool(add)
# todo_server.tool(reset_state)
