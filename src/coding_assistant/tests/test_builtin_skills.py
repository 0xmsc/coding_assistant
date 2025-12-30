from pathlib import Path
from coding_assistant.skills import load_builtin_skills, Skill

def test_load_builtin_skills():
    skills = load_builtin_skills()
    
    # We should have at least the general_developing skill we just added
    assert len(skills) >= 1
    
    names = {s.name for s in skills}
    assert "general_developing" in names
    
    # Check that paths are provided (even if they are traversable string paths)
    for skill in skills:
        assert skill.path is not None
        assert "SKILL.md" in str(skill.path)

def test_builtin_skills_parsing_content():
    # Verify that the placeholder skill has the expected structure
    skills = load_builtin_skills()
    general_skill = next(s for s in skills if s.name == "general_developing")
    
    assert "General principles" in general_skill.description
    
    # Verify it has the moved content
    content = Path(str(general_skill.path)).read_text()
    assert "## Exploring" in content
    assert "## Editing" in content
