from __future__ import annotations

import importlib.resources
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

import frontmatter  # type: ignore
from fastmcp import FastMCP

if hasattr(importlib.resources.abc, "Traversable"):
    from importlib.resources.abc import Traversable
else:
    # Fallback for older python if needed, though we require 3.12
    Traversable = Union[Path, importlib.resources.abc.Traversable]

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # Path to the main SKILL.md
    resources: Dict[str, str] = field(default_factory=dict)  # relative path -> content


def _load_resources_recursive(
    base_path: Traversable, current_path: Traversable, resources: Dict[str, str]
) -> None:
    """Recursively load all files in a directory into the resources dict."""
    for item in current_path.iterdir():
        if item.is_file():
            # Calculate relative path manually for Traversable
            # traversable doesn't have .relative_to() like Path
            rel_path = str(item).replace(str(base_path), "").lstrip(os.sep)
            try:
                resources[rel_path] = item.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"Could not read resource {item}: {e}")
        elif item.is_dir():
            _load_resources_recursive(base_path, item, resources)


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


def load_builtin_skills() -> List[Skill]:
    try:
        skills_root = importlib.resources.files("coding_assistant") / "skills"
        if not skills_root.is_dir():
            return []

        skills = []
        for skill_dir in skills_root.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.is_file():
                content = skill_file.read_text()
                skill = parse_skill_file(content, Path(str(skill_file)))
                if skill:
                    # Load all resources for this skill
                    _load_resources_recursive(skill_dir, skill_dir, skill.resources)
                    skills.append(skill)
        return skills
    except Exception as e:
        logger.error(f"Error loading builtin skills: {e}")
        return []


def format_skills_instructions(skills: List[Skill]) -> str:
    if not skills:
        return ""

    lines = [
        "## Skills",
        "",
        "- You have the following skills available to you:",
    ]

    for skill in skills:
        lines.append(f"  - Name: {skill.name}")
        lines.append(f"    - Description: {skill.description}")

    lines.extend(
        [
            "- Use `skills_read_skill(name=...)` to read the `SKILL.md` of a skill.",
            "- Use `skills_read_skill(name=..., resource=...)` to read specific resources or scripts of a skill.",
            "- Try to read a skill file when something that the user wants from you matches one of the descriptions.",
            "- The directory that contains the `SKILL.md` file might contain more files and subdirectories to explore, e.g. `/scripts` or `/references`.",
        ]
    )

    return "\n".join(lines)


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


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []

    skills = []
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            content = skill_file.read_text(encoding="utf-8")
            skill = parse_skill_file(content, skill_file)
            if skill:
                # Load all resources for this skill
                _load_resources_recursive(skill_dir, skill_dir, skill.resources)
                skills.append(skill)

    if not skills:
        logger.info(f"No valid skills found in {skills_dir}")

    return skills


def create_skills_server(skills: Optional[List[Skill]] = None) -> FastMCP:
    skills_server = FastMCP("Skills")

    # If no skills are provided, load built-in ones
    effective_skills = skills if skills is not None else load_builtin_skills()
    skills_map = {s.name: s for s in effective_skills}

    @skills_server.tool()
    async def read_skill(
        name: Annotated[str, "The name of the skill to read."],
        resource: Annotated[str | None, "Optional sub-resource to read (e.g. 'references/spec.md')."] = None,
    ) -> str:
        """
        Read a skill's main SKILL.md or one of its resources from memory.
        """
        skill = skills_map.get(name)
        if not skill:
            return f"Error: Skill '{name}' not found."

        res_path = resource if resource else "SKILL.md"
        content = skill.resources.get(res_path)

        if content is None:
            return f"Error: Resource '{res_path}' not found in skill '{name}'."

        return content

    return skills_server
