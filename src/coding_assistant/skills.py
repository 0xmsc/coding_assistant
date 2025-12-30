"""
Agent Skills module.

Provides functions to parse and load Agent Skills from a directory according to the specification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict, List, Optional

logger = logging.getLogger(__name__)


class Skill(TypedDict):
    name: str
    description: str


def _parse_frontmatter(content: str) -> dict[str, str]:
    """
    Parse YAML frontmatter from a markdown file.
    
    Extracts lines between --- markers and parses simple key: value pairs.
    Supports quoted values and basic string values.
    """
    lines = content.splitlines()
    frontmatter_start = -1
    frontmatter_end = -1
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "---":
            if frontmatter_start == -1:
                frontmatter_start = i
            else:
                frontmatter_end = i
                break
    
    if frontmatter_start == -1 or frontmatter_end == -1:
        return {}
    
    frontmatter_lines = lines[frontmatter_start + 1:frontmatter_end]
    result = {}
    
    for line in frontmatter_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        if ":" not in line:
            continue
        
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        
        # Remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        
        result[key] = value
    
    return result


def parse_skill_file(file_path: Path) -> Optional[Skill]:
    """
    Parse a single SKILL.md file and extract name and description.
    
    Returns:
        Skill dict with name and description, or None if invalid.
    """
    try:
        content = file_path.read_text()
    except Exception as e:
        logger.warning(f"Failed to read {file_path}: {e}")
        return None
    
    frontmatter = _parse_frontmatter(content)
    
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    
    if not name:
        logger.warning(f"No 'name' field in {file_path}")
        return None
    
    if not description:
        logger.warning(f"No 'description' field in {file_path}")
        return None
    
    # Validate name format per spec
    if len(name) > 64:
        logger.warning(f"Name too long (>64 chars) in {file_path}")
        return None
    
    if not name.replace("-", "").isalnum() or name.startswith("-") or name.endswith("-") or "--" in name:
        logger.warning(f"Invalid name format in {file_path}: {name}")
        return None
    
    # Validate description length
    if len(description) > 1024:
        logger.warning(f"Description too long (>1024 chars) in {file_path}")
        # Truncate instead of rejecting
        description = description[:1024]
    
    return {"name": name, "description": description}


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    """
    Recursively scan directory for SKILL.md files and load valid skills.
    
    Returns:
        List of Skill dicts with name and description.
    """
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []
    
    skills = []
    
    # Recursively find all SKILL.md files
    for skill_file in skills_dir.rglob("SKILL.md"):
        skill = parse_skill_file(skill_file)
        if skill:
            skills.append(skill)
    
    if not skills:
        logger.info(f"No valid skills found in {skills_dir}")
    
    return skills


def format_skills_section(skills: List[Skill]) -> str:
    """
    Format skills as a markdown section for instructions.
    """
    if not skills:
        return ""
    
    lines = ["# Available Agent Skills", ""]
    for skill in skills:
        # Escape any markdown characters in description
        description = skill["description"].replace("|", "\\|")
        lines.append(f"- **{skill['name']}**: {description}")
    
    return "\n".join(lines)