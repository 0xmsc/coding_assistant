from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from coding_assistant.infra.paths import get_builtin_instructions_dir, get_builtin_skills_dir
from coding_assistant.llm.types import Tool
from coding_assistant.tools.filesystem import create_filesystem_tools
from coding_assistant.tools.mcp_manager import MCPServerConfig, MCPServerManager
from coding_assistant.tools.mcp_tools import create_mcp_tools
from coding_assistant.tools.python import create_python_tools
from coding_assistant.tools.shell import create_shell_tools
from coding_assistant.tools.skills import create_skill_tools, format_skills_instructions
from coding_assistant.tools.tasks import TaskManager, create_task_tools
from coding_assistant.tools.todo import TodoManager, create_todo_tools
from coding_assistant.tools.workers import WorkerToolRuntime


@dataclass(slots=True)
class LocalToolBundle:
    """Local built-in tools plus their instruction block."""

    tools: list[Tool]
    instructions: str
    _worker_runtime: WorkerToolRuntime
    _mcp_manager: MCPServerManager | None = None

    async def close(self) -> None:
        if self._mcp_manager:
            await self._mcp_manager.close()
        await self._worker_runtime.close()


def load_tool_instructions() -> str:
    """Return the bundled instruction document for local tools."""
    return (get_builtin_instructions_dir() / "tools.md").read_text(encoding="utf-8").strip()


def create_local_tool_bundle(
    *,
    skills_directories: Sequence[Path],
    mcp_server_configs: Sequence[MCPServerConfig] = (),
    working_directory: Path | None = None,
) -> LocalToolBundle:
    """Build the in-process tool bundle used by the default CLI."""
    task_manager = TaskManager()
    todo_manager = TodoManager()
    worker_runtime = WorkerToolRuntime()

    skill_tools, skills = create_skill_tools(skills_directories=[get_builtin_skills_dir(), *skills_directories])
    instructions = load_tool_instructions()
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

    # Add MCP tools if configs provided
    mcp_manager: MCPServerManager | None = None
    if mcp_server_configs and working_directory:
        mcp_manager = MCPServerManager(
            configs=list(mcp_server_configs),
            working_directory=working_directory,
        )
        tools.extend(create_mcp_tools(mcp_manager))

    return LocalToolBundle(
        tools=tools,
        instructions=instructions,
        _worker_runtime=worker_runtime,
        _mcp_manager=mcp_manager,
    )
