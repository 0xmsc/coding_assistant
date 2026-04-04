from pathlib import Path
from typing import Any

import pytest

from coding_assistant.llm.types import TextToolResult, ToolResult
from coding_assistant.tools.skills import (
    Skill,
    create_skill_tools,
    format_skills_instructions,
    load_skills_from_directory,
    parse_skill_file,
)


def _text(result: ToolResult) -> str:
    assert isinstance(result, TextToolResult)
    return result.content


def test_load_skills_from_directory(tmp_path: Any) -> None:
    # Create a temporary skills directory structure
    skill1_dir = tmp_path / "skill1"
    skill1_dir.mkdir()
    skill1_file = skill1_dir / "SKILL.md"
    skill1_file.write_text("---\nname: skill1\ndescription: First test skill\n---\nSome instructions here")

    skill2_dir = tmp_path / "skill2"
    skill2_dir.mkdir()
    skill2_file = skill2_dir / "SKILL.md"
    skill2_file.write_text("---\nname: skill2\ndescription: Second test skill\nextra: field\n---\nMore instructions")

    # Load skills
    skills = load_skills_from_directory(tmp_path)

    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"skill1", "skill2"}

    descriptions = {s.description for s in skills}
    assert descriptions == {"First test skill", "Second test skill"}


def test_parse_skill_file_name_with_spaces(tmp_path: Any) -> None:
    content = "---\nname: name with spaces\ndescription: test\n---"
    identifier = str(tmp_path / "SKILL.md")

    skill = parse_skill_file(content, identifier, tmp_path)
    assert skill is not None
    assert skill.name == "name with spaces"


def test_format_skills_instructions() -> None:
    skills = [
        Skill(name="skill1", description="desc1", root=Path("/path/1"), resources=["SKILL.md", "script.py"]),
        Skill(name="skill2", description="desc2", root=Path("/path/2"), resources=["SKILL.md"]),
    ]

    section = format_skills_instructions(skills)

    assert "## Skills" in section
    assert "- **skill1**: desc1" in section
    assert "- **skill2**: desc2" in section
    assert "Use `skills_list_resources(name=...)` to list the resources available for a skill." in section
    assert "Use `skills_read(name=...)` to read the `SKILL.md` of a skill." in section


def test_parse_skill_file_missing_fields(tmp_path: Any) -> None:
    content = "---\nname: only-name\n---"
    identifier = str(tmp_path / "SKILL.md")

    skill = parse_skill_file(content, identifier, tmp_path)
    assert skill is None


def test_create_skill_tools(tmp_path: Any) -> None:
    # Create a CLI skill
    cli_skills_dir = tmp_path / "cli_skills"
    cli_skills_dir.mkdir()
    (cli_skills_dir / "my_cli_skill").mkdir()
    (cli_skills_dir / "my_cli_skill" / "SKILL.md").write_text("---\nname: my_cli_skill\ndescription: CLI skill\n---\n")

    tools, skills = create_skill_tools(skills_directories=[cli_skills_dir])
    instr = format_skills_instructions(skills)

    assert "my_cli_skill" in instr
    assert "skills_list_resources" in instr
    assert "skills_read" in instr
    assert {tool.name() for tool in tools} == {"skills_list_resources", "skills_read"}


def test_create_skill_tools_without_configured_skills() -> None:
    tools, skills = create_skill_tools(skills_directories=[])

    assert tools == []
    assert skills == []


def test_create_skill_tools_raises_on_duplicate_skill_names(tmp_path: Any) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    (first_root / "example").mkdir()
    (first_root / "example" / "SKILL.md").write_text("---\nname: example\ndescription: First\n---\n")

    second_root = tmp_path / "second"
    second_root.mkdir()
    (second_root / "example").mkdir()
    (second_root / "example" / "SKILL.md").write_text("---\nname: example\ndescription: Second\n---\n")

    with pytest.raises(RuntimeError, match="Duplicate skill name 'example'"):
        create_skill_tools(skills_directories=[first_root, second_root])


@pytest.mark.asyncio
async def test_skills_tools(tmp_path: Any) -> None:
    skill_dir = tmp_path / "myskill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: myskill\ndescription: desc\n---\ncontent")
    (skill_dir / "script.py").write_text("print(1)")

    tools, _ = create_skill_tools(skills_directories=[tmp_path])
    list_tool = next(tool for tool in tools if tool.name() == "skills_list_resources")
    read_tool = next(tool for tool in tools if tool.name() == "skills_read")

    result_text = _text(await list_tool.execute({"name": "myskill"}))
    assert "- SKILL.md" in result_text
    assert "- script.py" in result_text

    script_content = _text(await read_tool.execute({"name": "myskill", "resource": "script.py"}))
    assert script_content == "print(1)"

    main_content = _text(await read_tool.execute({"name": "myskill"}))
    assert "content" in main_content
