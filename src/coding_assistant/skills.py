"""
Agent Skills module.

Provides functions to parse and load Agent Skills from a directory according to the specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import frontmatter

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def parse_skill_file(file_path: Path) -> Optional[Skill]:
    """
    Parse a single SKILL.md file and extract name and description.

    Returns:
        Skill object with name, description, and path, or None if invalid.
    """
    try:
        post = frontmatter.load(file_path)
    except Exception as e:
        logger.warning(f"Failed to read or parse {file_path}: {e}")
        return None

    name = post.metadata.get("name")
    description = post.metadata.get("description")

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

    return Skill(name=name, description=description, path=file_path)


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    """
    Recursively scan directory for SKILL.md files and load valid skills.

    Returns:
        List of Skill objects with name, description, and path.
    """
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []

    skills = []

    # Recursively find all SKILL.md files
    for skill_file in skills_dir.glob("*/SKILL.md"):
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
        description = skill.description.replace("|", "\\|")
        lines.append(f"- **{skill.name}**: {description}")

    return "\n".join(lines)
