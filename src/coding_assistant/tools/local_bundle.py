from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.infra.paths import get_builtin_instructions_dir, get_builtin_skills_dir
from coding_assistant.llm.types import Tool
from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import create_skill_tools, format_skills_instructions
from coding_assistant.tools.tasks import TaskManager, create_task_tools
from coding_assistant.tools.todo import TodoManager, create_todo_tools
from coding_assistant.tools.workers import WorkerToolRuntime

WORKER_TOOL_INSTRUCTIONS = """
## Remotes

- Use `remotes_discover()` to find other locally advertised `coding-assistant` instances on this machine.
- Use `remote_connect(endpoint=...)` with the remote endpoint printed when `coding-assistant` starts, then use the returned local `remote_id` with the other remote tools.
- Use `remote_prompt(remote_id=..., prompt=...)` only when the remote is idle. If it is busy, wait for it to finish or use `remote_cancel(...)` before prompting again.
""".strip()


@dataclass(slots=True)
class LocalToolBundle:
    """Local built-in tools plus their instruction block."""

    tools: list[Tool]
    instructions: str
    _worker_runtime: WorkerToolRuntime

    async def close(self) -> None:
        await self._worker_runtime.close()


def load_tool_instructions() -> str:
    """Return the bundled instruction document for local tools."""
    return (get_builtin_instructions_dir() / "tools.md").read_text(encoding="utf-8").strip()


def create_local_tool_bundle(
    *,
    skills_directories: Sequence[Path],
) -> LocalToolBundle:
    """Build the in-process tool bundle used by the default CLI."""
    task_manager = TaskManager()
    todo_manager = TodoManager()
    worker_runtime = WorkerToolRuntime()

    skill_tools, skills = create_skill_tools(skills_directories=[get_builtin_skills_dir(), *skills_directories])
    instructions = f"{load_tool_instructions()}\n\n{WORKER_TOOL_INSTRUCTIONS}"
    skill_instructions = format_skills_instructions(skills)
    if skill_instructions:
        instructions = f"{instructions}\n\n{skill_instructions}"

    tools: list[Tool] = [
        *create_todo_tools(manager=todo_manager),
        *create_shell_tools(manager=task_manager),
        *create_python_tools(manager=task_manager),
        *create_filesystem_tools(),
        *create_task_tools(manager=task_manager),
        *skill_tools,
    ]

    tools.extend(worker_runtime.tools)

    return LocalToolBundle(
        tools=tools,
        instructions=instructions,
        _worker_runtime=worker_runtime,
    )
