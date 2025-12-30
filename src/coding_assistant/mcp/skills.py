from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, List, Optional

import frontmatter  # type: ignore
from fastmcp import FastMCP

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    root: Path
    resources: List[str] = field(default_factory=list)


def parse_skill_file(content: str, source_info: str, root: Path) -> Optional[Skill]:
    post = frontmatter.loads(content)

    name = post.metadata.get("name")
    description = post.metadata.get("description")

    if not name:
        logger.warning(f"No 'name' field in skill at {source_info}")
        return None

    if not description:
        logger.warning(f"No 'description' field in skill at {source_info}")
        return None

    return Skill(name=name, description=description, root=root)


def load_skills_from_root(root_dir: Path) -> List[Skill]:
    skills = []
    for skill_dir in root_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            content = skill_file.read_text(encoding="utf-8")
            skill = parse_skill_file(content, str(skill_file), skill_dir)
            if skill:
                # Collect allowed resources (files only)
                skill.resources = sorted([str(p.relative_to(skill_dir)) for p in skill_dir.glob("**/*") if p.is_file()])
                skills.append(skill)
    return skills


def load_builtin_skills() -> List[Skill]:
    files = importlib.resources.files("coding_assistant") / "skills"
    return load_skills_from_root(Path(str(files)))


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
        if skill.resources:
            resources_list = ", ".join(f"`{r}`" for r in skill.resources)
            lines.append(f"    - Resources: {resources_list}")

    lines.extend(
        [
            "- Use `skills_read_skill(name=...)` to read the `SKILL.md` of a skill.",
            "- Use `skills_read_skill(name=..., resource=...)` to read specific resources or scripts of a skill.",
            "- Try to read a skill file when something that the user wants from you matches one of the descriptions.",
            "- The directory that contains the `SKILL.md` file might contain more files and subdirectories to explore, e.g. `/scripts` or `/references`.",
        ]
    )

    return "\n".join(lines)


def load_skills_from_directory(skills_dir: Path) -> List[Skill]:
    if not skills_dir.exists() or not skills_dir.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_dir}")
        return []

    return load_skills_from_root(skills_dir)


def create_skills_server(
    skills_directories: Optional[List[Path]] = None,
) -> tuple[FastMCP, str]:
    """
    Create the skills server and return it along with the formatted instructions.
    """
    skills_server = FastMCP("Skills")

    # Load built-in skills
    all_skills = load_builtin_skills()

    # Load extra skills from directories
    if skills_directories:
        for directory in skills_directories:
            all_skills.extend(load_skills_from_directory(directory))

    skills_map = {s.name: s for s in all_skills}
    instructions = format_skills_instructions(all_skills)

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

        if res_path not in skill.resources:
            return f"Error: Resource '{res_path}' not found or not allowed in skill '{name}'."

        try:
            return (skill.root / res_path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error: Could not read resource '{res_path}' in skill '{name}': {e}"

    return skills_server, instructions
