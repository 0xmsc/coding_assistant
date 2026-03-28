from __future__ import annotations

import logging
from pathlib import Path

from coding_assistant.integrations.mcp_client import MCPServer
from coding_assistant.infra.paths import get_builtin_instructions_dir

logger = logging.getLogger(__name__)


def _load_default_instructions() -> str:
    """Return the built-in instruction document bundled with the package."""
    return (get_builtin_instructions_dir() / "global.md").read_text(encoding="utf-8").strip()


def get_instructions(
    *,
    working_directory: Path,
    user_instructions: list[str],
    extra_sections: list[str] | None = None,
    mcp_servers: list[MCPServer] | None = None,
) -> str:
    """Assemble instructions from defaults, project files, MCP, and user input."""
    sections: list[str] = []

    sections.append(_load_default_instructions())

    for path in [
        working_directory / ".coding_assistant" / "instructions.md",
        working_directory / "AGENTS.md",
    ]:
        if not path.exists():
            continue

        content = path.read_text().strip()
        if not content:
            continue

        sections.append(content)

    for section in extra_sections or []:
        if section and section.strip():
            sections.append(section.strip())

    for server in mcp_servers or []:
        instructions = server.instructions
        if instructions and instructions.strip():
            sections.append(f"# MCP `{server.name}` instructions\n\n{instructions.strip()}")

    if user_instructions:
        sections.append("# User-provided instructions")
        for user_instruction in user_instructions:
            if user_instruction and user_instruction.strip():
                sections.append(user_instruction.strip())

    for section in sections:
        if not section.startswith("# "):
            logger.warning(f"Instruction section {section} does not start with a top-level heading")

    return "\n\n".join(sections)
