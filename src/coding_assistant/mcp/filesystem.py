from __future__ import annotations

import difflib
from pathlib import Path
from typing import Annotated

import aiofiles
from fastmcp import FastMCP

filesystem_server = FastMCP()


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
    old_text: Annotated[str, "The text to be replaced."],
    new_text: Annotated[str, "The text to replace with."],
    replace_all: Annotated[bool, "Whether to replace all occurrences (default false)."] = False,
) -> str:
    """
    Apply text replacement(s) to a file and return a unified diff.

    Semantics:
    - The edit is validated against the current content.
    - The old_text must occur exactly once if replace_all is false; otherwise a ValueError is raised.
    - If replace_all is true, all occurrences are replaced.
    - If validation fails, no changes are written.
    """

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        original = await f.read()

    count = original.count(old_text)

    if count == 0:
        raise ValueError(f"{old_text} not found in {path}; no changes made")

    if not replace_all and count > 1:
        raise ValueError(f"{old_text} occurs multiple times in {path}; edit is not unique")

    replace_count = -1 if replace_all else 1
    updated = original.replace(old_text, new_text, replace_count)

    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(updated)

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
