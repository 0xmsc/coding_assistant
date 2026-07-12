from __future__ import annotations

from pathlib import Path

import pytest

from coding_assistant.llm.types import TextToolResult
from coding_assistant.worker.agent import WorkerAgentConfig, build_worker_instructions, create_worker_agent


def test_build_worker_instructions_discovers_workspace_skills(tmp_path: Path) -> None:
    skill_root = tmp_path / ".agents" / "skills" / "apps-api"
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n",
        encoding="utf-8",
    )

    instructions = build_worker_instructions(config=WorkerAgentConfig(working_directory=tmp_path))

    assert "apps-api" in instructions
    assert "Use apps REST APIs." in instructions


@pytest.mark.asyncio
async def test_create_worker_agent_closes_background_tasks(tmp_path: Path) -> None:
    tasks_get_status = None
    async with create_worker_agent(config=WorkerAgentConfig(working_directory=tmp_path)) as bundle:
        shell_execute = next(tool for tool in bundle.tools if tool.name() == "shell_execute")
        tasks_get_status = next(tool for tool in bundle.tools if tool.name() == "tasks_get_status")
        result = await shell_execute.execute({"command": "sleep 10", "background": True})
        assert isinstance(result, TextToolResult)

    assert tasks_get_status is not None
    status = await tasks_get_status.execute({"task_id": 1})
    assert isinstance(status, TextToolResult)
    assert status.content == "Error: Task 1 not found."
