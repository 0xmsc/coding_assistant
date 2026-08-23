from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.infra.paths import get_builtin_instructions_dir, get_builtin_skills_dir
from coding_assistant.llm.types import Tool
from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import create_skill_tools, format_instructions_with_skills, load_skills
from coding_assistant.tools.tasks import TaskManager, create_task_tools


@dataclass(slots=True)
class CliToolBundle:
    """Built-in CLI tools plus their instruction block."""

    tools: list[Tool]
    instructions: str
    _task_manager: TaskManager

    async def close(self) -> None:
        await self._task_manager.close()


def load_tool_instructions() -> str:
    """Return the bundled instruction document for local tools."""
    return (get_builtin_instructions_dir() / "tools.md").read_text(encoding="utf-8").strip()


def create_cli_tool_bundle(
    *,
    skills_directories: Sequence[Path],
    working_directory: Path | None = None,
) -> CliToolBundle:
    """Build the in-process tool bundle used by the interactive CLI."""
    task_manager = TaskManager()

    skills = load_skills(skills_directories=[get_builtin_skills_dir(), *skills_directories])
    skill_tools = create_skill_tools(skills=skills)
    instructions = format_instructions_with_skills(instructions=load_tool_instructions(), skills=skills)

    tools: list[Tool] = [
        *create_shell_tools(manager=task_manager),
        *create_python_tools(manager=task_manager),
        *create_filesystem_tools(),
        *create_task_tools(manager=task_manager),
        *skill_tools,
    ]

    return CliToolBundle(
        tools=tools,
        instructions=instructions,
        _task_manager=task_manager,
    )
