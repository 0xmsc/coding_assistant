from pathlib import Path
from coding_assistant.skills import load_skills_from_directory, parse_skill_file


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
    from coding_assistant.skills import Skill, format_skills_section

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


def test_create_skills_section_includes_builtin(tmp_path):
    from coding_assistant.main import create_skills_section
    
    # Even with empty CLI directories, it should return the built-in skills section
    section = create_skills_section([])
    assert section is not None
    assert "# Skills" in section
    assert "general_coding" in section

def test_create_skills_section_merges_cli_and_builtin(tmp_path):
    from coding_assistant.main import create_skills_section
    
    # Create a CLI skill
    cli_skills_dir = tmp_path / "cli_skills"
    cli_skills_dir.mkdir()
    (cli_skills_dir / "my_cli_skill").mkdir()
    (cli_skills_dir / "my_cli_skill" / "SKILL.md").write_text("---\nname: my_cli_skill\ndescription: CLI skill\n---\n")
    
    section = create_skills_section([str(cli_skills_dir)])
    assert section is not None
    assert "general_coding" in section
    assert "my_cli_skill" in section


def test_parse_skill_file_missing_fields(tmp_path):
    content = "---\nname: only-name\n---"
    path = tmp_path / "SKILL.md"

    skill = parse_skill_file(content, path)
    assert skill is None
