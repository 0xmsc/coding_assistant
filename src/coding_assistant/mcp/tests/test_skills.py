from typing import Any, cast
from pathlib import Path
import pytest
from mcp.types import TextContent
from coding_assistant.mcp.skills import load_skills_from_directory, parse_skill_file, load_builtin_skills


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
    from coding_assistant.mcp.skills import Skill, format_skills_instructions

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


def test_load_builtin_skills() -> None:
    skills = load_builtin_skills()

    # Built-ins are repository-driven; assert generic invariants.
    assert len(skills) >= 1

    assert all(s.name for s in skills)
    assert all(s.description for s in skills)
    assert all("SKILL.md" in s.resources for s in skills)


def test_create_skills_server(tmp_path: Any) -> None:
    from coding_assistant.mcp.skills import create_skills_server

    # Create a CLI skill
    cli_skills_dir = tmp_path / "cli_skills"
    cli_skills_dir.mkdir()
    (cli_skills_dir / "my_cli_skill").mkdir()
    (cli_skills_dir / "my_cli_skill" / "SKILL.md").write_text("---\nname: my_cli_skill\ndescription: CLI skill\n---\n")

    server, instr = create_skills_server(skills_directories=[cli_skills_dir])

    for builtin in load_builtin_skills():
        assert builtin.name in instr
    assert "my_cli_skill" in instr
    assert "skills_list_resources" in instr
    assert "skills_read" in instr


@pytest.mark.asyncio
async def test_skills_tools(tmp_path: Any) -> None:
    from coding_assistant.mcp.skills import create_skills_server

    skill_dir = tmp_path / "myskill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: myskill\ndescription: desc\n---\ncontent")
    (skill_dir / "script.py").write_text("print(1)")

    server, _ = create_skills_server(skills_directories=[tmp_path])

    # Test list_resources
    tools = await server.get_tools()
    list_tool = tools["list_resources"]
    result = await list_tool.run({"name": "myskill"})
    # FastMCP returns a ToolResult object. Content is in result.content
    result_text = cast(TextContent, result.content[0]).text
    assert "- SKILL.md" in result_text
    assert "- script.py" in result_text

    # Test read
    read_tool = tools["read"]
    result = await read_tool.run({"name": "myskill", "resource": "script.py"})
    assert cast(TextContent, result.content[0]).text == "print(1)"

    # Test read default
    result_main = await read_tool.run({"name": "myskill"})
    assert "content" in cast(TextContent, result_main.content[0]).text


def test_builtin_skills_parsing_content() -> None:
    # Verify built-in skills are parseable from their own SKILL.md file.
    skills = load_builtin_skills()
    skill = skills[0]
    assert "SKILL.md" in skill.resources

    skill_file = skill.root / "SKILL.md"
    content = skill_file.read_text()
    parsed = parse_skill_file(content, str(skill_file), skill.root)

    assert parsed is not None
    assert parsed.name == skill.name
    assert parsed.description == skill.description
