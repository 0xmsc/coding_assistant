from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from coding_assistant.core.instructions import get_instructions
from coding_assistant.llm.types import Tool
from coding_assistant.worker.tool_bundle import create_worker_tool_bundle


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


def build_worker_instructions(*, config: WorkerAgentConfig) -> str:
    """Resolve the worker system instructions without starting an agent session."""
    tool_bundle = create_worker_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
        process_env=config.process_env,
    )
    return get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[tool_bundle.instructions],
    )


@asynccontextmanager
async def create_worker_agent(*, config: WorkerAgentConfig) -> AsyncIterator[WorkerAgentBundle]:
    """Resolve instructions and tools for a worker run."""
    tool_bundle = create_worker_tool_bundle(
        skills_directories=[Path(path).resolve() for path in config.skills_directories],
        process_env=config.process_env,
    )
    instructions = get_instructions(
        working_directory=config.working_directory,
        user_instructions=list(config.user_instructions),
        extra_sections=[tool_bundle.instructions],
    )
    yield WorkerAgentBundle(
        tools=tool_bundle.tools,
        instructions=instructions,
    )
