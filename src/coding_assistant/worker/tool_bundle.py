from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.infra.paths import get_builtin_instructions_dir, get_builtin_skills_dir
from coding_assistant.llm.types import Tool
from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.load_image import create_load_image_tools
from coding_assistant.tools.session_title import SessionTitleState, create_session_title_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import Skill, create_skill_tools, format_instructions_with_skills, load_skills
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


def load_worker_tool_context(*, skills_directories: Sequence[Path]) -> tuple[list[Skill], str]:
    """Load worker skills and the instruction text that describes them."""
    skills = load_skills(skills_directories=[get_builtin_skills_dir(), *skills_directories])
    instructions = format_instructions_with_skills(instructions=load_tool_instructions(), skills=skills)
    return skills, instructions


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

    skills, instructions = load_worker_tool_context(skills_directories=skills_directories)
    skill_tools = create_skill_tools(skills=skills)

    tools: list[Tool] = [
        *create_shell_tools(manager=task_manager, process_env=tool_process_env),
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
