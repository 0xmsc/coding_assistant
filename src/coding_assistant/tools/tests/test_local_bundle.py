from pathlib import Path

import pytest

from coding_assistant.tools.local_bundle import create_local_tool_bundle, load_local_tool_instructions


def test_create_local_tool_bundle_includes_builtin_skills() -> None:
    bundle = create_local_tool_bundle(skills_directories=[])
    tool_names = {tool.name() for tool in bundle.tools}

    assert bundle.instructions.startswith(load_local_tool_instructions())
    assert "- **brainstorm**: Structured ideation and tradeoff analysis without implementation." in bundle.instructions
    assert "- **develop**: General implementation and refactoring guidance for codebase work." in bundle.instructions
    assert "- **plan**: Turn a concrete non-trivial task into an implementation plan before editing files." in (
        bundle.instructions
    )
    assert "- **todo**: Maintain a short execution task list while implementing changes." in bundle.instructions
    assert {"skills_list_resources", "skills_read"} <= tool_names


def test_create_local_tool_bundle_raises_on_skill_name_collision(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    skill_directory = skill_root / "develop"
    skill_directory.mkdir()
    (skill_directory / "SKILL.md").write_text(
        "---\nname: develop\ndescription: Development guidance\n---\nUse this skill."
    )

    with pytest.raises(RuntimeError, match="Duplicate skill name 'develop'"):
        create_local_tool_bundle(skills_directories=[skill_root])
