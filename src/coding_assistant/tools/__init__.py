from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.llm.types import Tool

from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import create_skill_tools, format_skills_instructions
from coding_assistant.tools.tasks import TaskManager, create_task_tools
from coding_assistant.tools.todo import TodoManager, create_todo_tools

LOCAL_TOOL_INSTRUCTIONS = """
# Local tools

## Shell

- Use `shell_execute` to execute shell commands.
- `shell_execute` can run multi-line scripts.
- Example commands: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Create a temporary directory (via `mktemp -d`) if you want to write temporary files.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Interactive commands (e.g., `git rebase -i`) are not supported and will block.

## Python

- You have access to a Python interpreter via `python_execute`.
- `python_execute` can run multi-line scripts.
- Prefer Python over Shell for readability.
- Add comments to your scripts to explain your logic.

## TODO

- Always manage a TODO list while working on your task.
- Use the `todo_*` tools for managing the list.

## Filesystem

- Use filesystem tools to read, write, and edit files.
- Try not to use shell commands for file operations.

## Tasks

- Use tasks tools to monitor and manage background tasks.
""".strip()


@dataclass(slots=True)
class LocalToolBundle:
    """Local built-in tools plus their instruction block."""

    tools: list[Tool]
    instructions: str


def create_local_tool_bundle(*, skills_directories: Sequence[Path]) -> LocalToolBundle:
    """Build the in-process tool bundle used by the default CLI."""
    task_manager = TaskManager()
    todo_manager = TodoManager()

    skill_tools, skills = create_skill_tools(skills_directories=skills_directories)
    instructions = LOCAL_TOOL_INSTRUCTIONS
    skill_instructions = format_skills_instructions(skills)
    if skill_instructions:
        instructions = f"{instructions}\n\n{skill_instructions}"

    tools = [
        *create_todo_tools(manager=todo_manager),
        *create_shell_tools(manager=task_manager),
        *create_python_tools(manager=task_manager),
        *create_filesystem_tools(),
        *create_task_tools(manager=task_manager),
        *skill_tools,
    ]
    return LocalToolBundle(tools=tools, instructions=instructions)
