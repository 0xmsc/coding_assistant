from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import aiofiles
from fastmcp import FastMCP
from pydantic import Field

filesystem_server = FastMCP()


@dataclass
class TextEdit:
    old_text: Annotated[str, Field(description="The text to be replaced.")]
    new_text: Annotated[str, Field(description="The text to replace with.")]


async def write_file(
    path: Annotated[Path, "The file path to write (will be created or overwritten)."],
    content: Annotated[str, "The content to write to the file."],
) -> str:
    """Overwrite (or create) a file with the given content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(content)

    return f"Successfully wrote file {path}"


async def edit_file(
    path: Annotated[Path, "The file to edit."],
    edits: Annotated[list[TextEdit], "A list of edit operations."],
) -> str:
    """
    Apply multiple unique text replacements to a file and return a unified diff.

    Semantics:
    - All edits are validated against the current content at the time they are applied.
    - For each edit, the old_text must occur exactly once; otherwise a ValueError is raised.
    - If any edit fails validation, no changes are written (operation is atomic).
    - Edits are applied in the given order.
    """

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        original = await f.read()

    updated = original

    for edit in edits:
        old_text = edit.old_text
        new_text = edit.new_text

        count = updated.count(old_text)

        if count == 0:
            raise ValueError(f"{old_text} not found in {path}; no changes made")

        if count > 1:
            raise ValueError(f"{old_text} occurs multiple times in {path}; edit is not unique")

        updated = updated.replace(old_text, new_text, 1)

    # Write back the updated content only after all validations pass
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(updated)

    # Build unified diff for the entire operation
    diff_lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
    )

    return "\n".join(diff_lines)


filesystem_server.tool(write_file)
filesystem_server.tool(edit_file)
