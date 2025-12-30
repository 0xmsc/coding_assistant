from pathlib import Path
from coding_assistant.mcp.skills import load_skills_from_directory, parse_skill_file, load_builtin_skills


def test_load_skills_from_directory(tmp_path):
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


def test_parse_skill_file_name_with_spaces(tmp_path):
    content = "---\nname: name with spaces\ndescription: test\n---"
    path = tmp_path / "SKILL.md"

    skill = parse_skill_file(content, path)
    assert skill is not None
    assert skill.name == "name with spaces"


def test_format_skills_section():
    from coding_assistant.mcp.skills import Skill, format_skills_section

    skills = [
        Skill(name="skill1", description="desc1", path=Path("/path/1/SKILL.md")),
        Skill(name="skill2", description="desc2", path=Path("/path/2/SKILL.md")),
    ]

    section = format_skills_section(skills)

    assert "# Skills" in section
    assert "- Name: skill1" in section
    assert "- Description: desc1" in section
    assert "- Path: /path/1/SKILL.md" in section
    assert "- Name: skill2" in section
    assert "- Description: desc2" in section
    assert "- Path: /path/2/SKILL.md" in section
    assert "If you want to use a skill, read its `SKILL.md` file" in section


def test_parse_skill_file_missing_fields(tmp_path):
    content = "---\nname: only-name\n---"
    path = tmp_path / "SKILL.md"

    skill = parse_skill_file(content, path)
    assert skill is None


def test_load_builtin_skills():
    skills = load_builtin_skills()

    # We should have at least the developing skill we just added
    assert len(skills) >= 1

    names = {s.name for s in skills}
    assert "developing" in names

    # Check that paths are provided (even if they are traversable string paths)
    for skill in skills:
        assert skill.path is not None
        assert "SKILL.md" in str(skill.path)


def test_builtin_skills_parsing_content():
    # Verify that the placeholder skill has the expected structure
    skills = load_builtin_skills()
    general_skill = next(s for s in skills if s.name == "developing")

    assert "General principles" in general_skill.description

    # Verify it has the moved content
    content = general_skill.path.expanduser().read_text()
    assert "## Exploring" in content
    assert "## Editing" in content
