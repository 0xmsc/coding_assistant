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

    return Skill(name=name, description=description, path=file_path)


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
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


def format_skills_section(skills: List[Skill]) -> str | None:
    if not skills:
        return None

    lines = [
        "# Skills",
        "",
        "- You have the following skills available to you:",
    ]

    for skill in skills:
        lines.append(f"  - `{skill.name}` – {skill.description} – `{skill.path}`")

    lines.extend(
        [
            "- If you want to use a skill, read its `SKILL.md` file, it will contain all the details.",
            "- Try to read a skill file when something that the user wants from you matches one of the descriptions.",
            "- The directory that contains the `SKILL.md` file might contain more files and subdirectories to explore, e.g. `/scripts`.",
        ]
    )

    return "\n".join(lines)
