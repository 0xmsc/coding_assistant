"""
Agent Skills module.

Provides functions to parse and load Agent Skills from a directory according to the specification.
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import frontmatter  # type: ignore

from coding_assistant.paths import maybe_collapse_user

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def parse_skill_file(content: str, path: Path) -> Optional[Skill]:
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


def _load_skills_from_traversable(root: Traversable) -> List[Skill]:
    if not root.is_dir():
        return []

    skills = []
    for skill_dir in root.iterdir():
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            content = skill_file.read_text()
            # TODO: Is this valid, does this work in all circumstances?
            path_obj = maybe_collapse_user(Path(str(skill_file)))
            skill = parse_skill_file(content, path_obj)
            if skill:
                skills.append(skill)

    return skills


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []

    skills = _load_skills_from_traversable(skills_dir)

    if not skills:
        logger.info(f"No valid skills found in {skills_dir}")

    return skills


def load_builtin_skills() -> List[Skill]:
    skills_root = importlib.resources.files("coding_assistant") / "skills"
    return _load_skills_from_traversable(skills_root)


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
