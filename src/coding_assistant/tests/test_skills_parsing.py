import pytest
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

def test_parse_skill_file_invalid_name(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    # Name with spaces is invalid according to existing validation logic in skills.py
    skill_file.write_text("---\nname: invalid name\ndescription: test\n---")
    
    skill = parse_skill_file(skill_file)
    assert skill is None

def test_parse_skill_file_missing_fields(tmp_path):
    skill_file = tmp_path / "SKILL.md"
    skill_file.write_text("---\nname: only-name\n---")
    
    skill = parse_skill_file(skill_file)
    assert skill is None
