from __future__ import annotations

from pathlib import Path

from coding_assistant.worker.agent import WorkerAgentConfig, build_worker_instructions


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
