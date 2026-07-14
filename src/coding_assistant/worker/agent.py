from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from coding_assistant.core.instructions import get_instructions
from coding_assistant.llm.types import Tool
from coding_assistant.tools.session_title import SessionTitleState
from coding_assistant.worker.tool_bundle import create_worker_tool_bundle, load_worker_tool_context


@dataclass(slots=True)
class WorkerAgentConfig:
    """Configuration for a session worker agent."""

    working_directory: Path
    skills_directories: tuple[str, ...] = ()
    user_instructions: tuple[str, ...] = ()
    process_env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class WorkerAgentBundle:
    """Resolved instructions and tools needed to run a worker agent."""

    tools: list[Tool]
    instructions: str
    session_title_state: SessionTitleState


def _skills_directories(config: WorkerAgentConfig) -> list[Path]:
    directories = [Path(path).resolve() for path in config.skills_directories]
    workspace_skills = config.working_directory / ".agents" / "skills"
    if workspace_skills.is_dir():
        directories.append(workspace_skills.resolve())
    return list(dict.fromkeys(directories))


def build_worker_instructions(*, config: WorkerAgentConfig) -> str:
    """Resolve the worker system instructions without starting an agent session."""
    _, tool_instructions = load_worker_tool_context(skills_directories=_skills_directories(config))
    return get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[tool_instructions],
    )


@asynccontextmanager
async def create_worker_agent(*, config: WorkerAgentConfig) -> AsyncIterator[WorkerAgentBundle]:
    """Resolve instructions and tools for a worker run."""
    tool_bundle = create_worker_tool_bundle(
        workspace=config.working_directory,
        skills_directories=_skills_directories(config),
        process_env=config.process_env,
    )
    instructions = get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[tool_bundle.instructions],
    )
    try:
        yield WorkerAgentBundle(
            tools=tool_bundle.tools,
            instructions=instructions,
            session_title_state=tool_bundle.session_title_state,
        )
    finally:
        await tool_bundle.close()
