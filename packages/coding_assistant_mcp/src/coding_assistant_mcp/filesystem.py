from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union

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
    edit: Annotated[Union[TextEdit, str], "A text edit operation (as an object or JSON string)."],
) -> str:
    """
    Apply a single text replacement to a file and return a unified diff.

    Semantics:
    - The edit is validated against the current content.
    - The old_text must occur exactly once; otherwise a ValueError is raised.
    - If validation fails, no changes are written.
    - The edit parameter can be a TextEdit object or a JSON string representation.
    """

    # Parse edit if provided as a JSON string
    if isinstance(edit, str):
        try:
            parsed_data = json.loads(edit)
            edit = TextEdit(old_text=parsed_data["old_text"], new_text=parsed_data["new_text"])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid JSON format for edit: {e}. Expected a JSON string representing "
                "an object with 'old_text' and 'new_text' keys."
            )

    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        original = await f.read()

    old_text = edit.old_text
    new_text = edit.new_text

    count = original.count(old_text)

    if count == 0:
        raise ValueError(f"{old_text} not found in {path}; no changes made")

    if count > 1:
        raise ValueError(f"{old_text} occurs multiple times in {path}; edit is not unique")

    updated = original.replace(old_text, new_text, 1)

    # Write back the updated content only after validation passes
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(updated)

    # Build unified diff for the operation
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
