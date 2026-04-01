from pathlib import Path

import pytest

from coding_assistant.tools.local_bundle import create_local_tool_bundle, load_tool_instructions


def test_create_local_tool_bundle_includes_builtin_skills() -> None:
    bundle = create_local_tool_bundle(skills_directories=[])
    tool_names = {tool.name() for tool in bundle.tools}

    assert bundle.instructions.startswith(load_tool_instructions())
    assert "- **example**: Example packaged skill showing the expected SKILL.md structure." in bundle.instructions
    assert {
        "skills_list_resources",
        "skills_read",
        "remote_connect",
        "remotes_list",
        "remote_prompt",
        "remote_wait",
        "remotes_wait_any",
        "remote_cancel",
        "remote_disconnect",
    } <= tool_names


def test_create_local_tool_bundle_raises_on_skill_name_collision(tmp_path: Path) -> None:
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    skill_directory = skill_root / "example"
    skill_directory.mkdir()
    (skill_directory / "SKILL.md").write_text("---\nname: example\ndescription: Example override\n---\nUse this skill.")

    with pytest.raises(RuntimeError, match="Duplicate skill name 'example'"):
        create_local_tool_bundle(skills_directories=[skill_root])
