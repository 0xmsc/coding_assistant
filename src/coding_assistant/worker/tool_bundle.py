from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.infra.paths import get_builtin_instructions_dir, get_builtin_skills_dir
from coding_assistant.llm.types import Tool
from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.load_image import create_load_image_tools
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.session_title import SessionTitleState, create_session_title_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import create_skill_tools, format_skills_instructions
from coding_assistant.tools.tasks import TaskManager, create_task_tools


@dataclass(slots=True)
class WorkerToolBundle:
    """Built-in worker tools plus their instruction block."""

    tools: list[Tool]
    instructions: str
    session_title_state: SessionTitleState
    _task_manager: TaskManager

    async def close(self) -> None:
        await self._task_manager.close()


def load_tool_instructions() -> str:
    """Return the bundled instruction document for worker tools."""
    return (get_builtin_instructions_dir() / "worker_tools.md").read_text(encoding="utf-8").strip()


def create_worker_tool_bundle(
    *,
    workspace: Path,
    skills_directories: Sequence[Path],
    process_env: dict[str, str] | None = None,
) -> WorkerToolBundle:
    """Build the local execution tool bundle used by worker agents."""
    task_manager = TaskManager()
    session_title_state = SessionTitleState()
    tool_process_env = process_env or {}

    skill_tools, skills = create_skill_tools(skills_directories=[get_builtin_skills_dir(), *skills_directories])
    instructions = load_tool_instructions()
    skill_instructions = format_skills_instructions(skills)
    if skill_instructions:
        instructions = f"{instructions}\n\n{skill_instructions}"

    tools: list[Tool] = [
        *create_shell_tools(manager=task_manager, process_env=tool_process_env),
        *create_python_tools(manager=task_manager, process_env=tool_process_env),
        *create_filesystem_tools(),
        *create_load_image_tools(workspace=workspace),
        *create_session_title_tools(state=session_title_state),
        *create_task_tools(manager=task_manager),
        *skill_tools,
    ]

    return WorkerToolBundle(
        tools=tools,
        instructions=instructions,
        session_title_state=session_title_state,
        _task_manager=task_manager,
    )
