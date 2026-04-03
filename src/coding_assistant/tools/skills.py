from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import frontmatter  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Loaded skill metadata and the files it is allowed to expose."""

    name: str
    description: str
    root: Path
    resources: list[str] = field(default_factory=list)


def parse_skill_file(content: str, source_info: str, root: Path) -> Skill | None:
    """Parse one `SKILL.md` file into a `Skill` record when valid."""
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


def load_skills_from_root(root_directory: Path) -> list[Skill]:
    """Load all skills from the immediate child directories of a root."""
    skills: list[Skill] = []
    for skill_directory in sorted(root_directory.iterdir(), key=lambda path: path.name):
        if not skill_directory.is_dir():
            continue

        skill_file = skill_directory / "SKILL.md"
        if not skill_file.is_file():
            continue

        skill = parse_skill_file(
            content=skill_file.read_text(encoding="utf-8"),
            source_info=str(skill_file),
            root=skill_directory,
        )
        if skill is None:
            continue

        skill.resources = sorted(
            [str(path.relative_to(skill_directory)) for path in skill_directory.glob("**/*") if path.is_file()],
        )
        skills.append(skill)

    return skills


def load_skills_from_directory(skills_directory: Path) -> list[Skill]:
    """Load skills from one user-provided skills directory."""
    if not skills_directory.exists() or not skills_directory.is_dir():
        logger.warning(f"Skills directory does not exist or is not a directory: {skills_directory}")
        return []
    return load_skills_from_root(skills_directory)


def format_skills_instructions(skills: list[Skill]) -> str:
    """Render the instruction block that explains the available skills."""
    if not skills:
        return ""

    lines = [
        "## Skills",
        "",
        "- You have the following skills available to you:",
    ]
    for skill in skills:
        lines.append(f"  - **{skill.name}**: {skill.description}")

    lines.extend(
        [
            "- Use `skills_list_resources(name=...)` to list the resources available for a skill.",
            "- Use `skills_read(name=...)` to read the `SKILL.md` of a skill.",
            "- Use `skills_read(name=..., resource=...)` to read specific resources or scripts of a skill.",
            "- If a skill could match the users task, you must read it.",
            "- Before implementation work, read any relevant development skill that is available.",
        ],
    )
    return "\n".join(lines)


class SkillsListResourcesInput(BaseModel):
    name: str = Field(description="The name of the skill to list resources for.")


class SkillsReadInput(BaseModel):
    name: str = Field(description="The name of the skill to read.")
    resource: str | None = Field(
        default=None,
        description="Optional sub-resource to read, for example `references/spec.md`.",
    )


class SkillsListResourcesTool(Tool):
    """List the readable files for one loaded skill."""

    def __init__(self, *, skills_by_name: dict[str, Skill]) -> None:
        self._skills_by_name = skills_by_name

    def name(self) -> str:
        return "skills_list_resources"

    def description(self) -> str:
        return "List the files that are available inside one skill."

    def parameters(self) -> dict[str, Any]:
        return SkillsListResourcesInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = SkillsListResourcesInput.model_validate(parameters)
        skill = self._skills_by_name.get(validated.name)
        if skill is None:
            return "Skill not found."
        return "\n".join(f"- {resource}" for resource in skill.resources)


class SkillsReadTool(Tool):
    """Read one skill file that has been explicitly exposed."""

    def __init__(self, *, skills_by_name: dict[str, Skill]) -> None:
        self._skills_by_name = skills_by_name

    def name(self) -> str:
        return "skills_read"

    def description(self) -> str:
        return "Read a skill's `SKILL.md` file or one of its resource files."

    def parameters(self) -> dict[str, Any]:
        return SkillsReadInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> str:
        validated = SkillsReadInput.model_validate(parameters)
        skill = self._skills_by_name.get(validated.name)
        if skill is None:
            return f"Error: Skill '{validated.name}' not found."

        resource = validated.resource or "SKILL.md"
        if resource not in skill.resources:
            return f"Error: Resource '{resource}' not found or not allowed in skill '{validated.name}'."

        try:
            return (skill.root / resource).read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error: Could not read resource '{resource}' in skill '{validated.name}': {exc}"


def create_skill_tools(*, skills_directories: Sequence[Path]) -> tuple[list[Tool], list[Skill]]:
    """Create skill-inspection tools for the configured skill directories."""
    skills: list[Skill] = []
    for directory in skills_directories:
        skills.extend(load_skills_from_directory(directory))

    if not skills:
        return [], []

    skills_by_name: dict[str, Skill] = {}
    for skill in skills:
        existing_skill = skills_by_name.get(skill.name)
        if existing_skill is not None:
            raise RuntimeError(
                f"Duplicate skill name '{skill.name}' found in '{existing_skill.root}' and '{skill.root}'. "
                "Skill names must be unique.",
            )
        skills_by_name[skill.name] = skill

    tools: list[Tool] = [
        SkillsListResourcesTool(skills_by_name=skills_by_name),
        SkillsReadTool(skills_by_name=skills_by_name),
    ]
    return tools, list(skills_by_name.values())
