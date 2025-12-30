"""
Agent Skills module.

Provides functions to parse and load Agent Skills from a directory according to the specification.
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import frontmatter  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    path: Union[Path, str]


def parse_skill_file(content: str, path: Union[Path, str]) -> Optional[Skill]:
    try:
        post = frontmatter.loads(content)
    except Exception as e:
        logger.warning(f"Failed to parse skill at {path}: {e}")
        return None

    name = post.metadata.get("name")
    description = post.metadata.get("description")

    if not name:
        logger.warning(f"No 'name' field in skill at {path}")
        return None

    if not description:
        logger.warning(f"No 'description' field in skill at {path}")
        return None

    return Skill(name=name, description=description, path=path)


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []

    skills = []

    # Recursively find all SKILL.md files
    for skill_file in skills_dir.glob("*/SKILL.md"):
        try:
            content = skill_file.read_text()
            skill = parse_skill_file(content, skill_file)
            if skill:
                skills.append(skill)
        except Exception as e:
            logger.warning(f"Failed to read {skill_file}: {e}")

    if not skills:
        logger.info(f"No valid skills found in {skills_dir}")

    return skills


def load_builtin_skills() -> List[Skill]:
    skills = []
    try:
        skills_root = importlib.resources.files("coding_assistant") / "skills"
        if not skills_root.is_dir():
            return []

        for skill_dir in skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                try:
                    content = skill_file.read_text()
                    # We use the string representation of the Traversable path
                    skill = parse_skill_file(content, str(skill_file))
                    if skill:
                        skills.append(skill)
                except Exception as e:
                    logger.warning(f"Failed to read built-in skill {skill_file}: {e}")
    except Exception as e:
        logger.warning(f"Failed to load built-in skills: {e}")

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
        lines.append(f"  - Name: {skill.name}")
        lines.append(f"    - Description: {skill.description}")
        lines.append(f"    - Path: {skill.path}")

    lines.extend(
        [
            "- If you want to use a skill, read its `SKILL.md` file, it will contain all the details.",
            "- Try to read a skill file when something that the user wants from you matches one of the descriptions.",
            "- The directory that contains the `SKILL.md` file might contain more files and subdirectories to explore, e.g. `/scripts`.",
        ]
    )

    return "\n".join(lines)
