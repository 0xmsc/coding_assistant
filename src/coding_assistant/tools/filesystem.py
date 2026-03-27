from __future__ import annotations

import difflib
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from coding_assistant.tools.base import StructuredTool
from coding_assistant.llm.types import Tool


class FilesystemWriteFileInput(BaseModel):
    path: str = Field(description="The file path to write. Existing files will be overwritten.")
    content: str = Field(description="The full file content to write.")


class FilesystemEditFileInput(BaseModel):
    path: str = Field(description="The file path to edit.")
    old_text: str = Field(description="The exact text to replace.")
    new_text: str = Field(description="The replacement text.")
    replace_all: bool = Field(
        default=False,
        description="If true, replace every occurrence of `old_text` instead of requiring a unique match.",
    )


async def write_file(path: Path, content: str) -> str:
    """Overwrite or create a file with the given content."""
    expanded_path = path.expanduser()
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(expanded_path, "w", encoding="utf-8") as file_handle:
        await file_handle.write(content)

    return f"Successfully wrote file {expanded_path}"


async def edit_file(path: Path, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Apply one validated text replacement to a file and return a unified diff."""
    expanded_path = path.expanduser()
    async with aiofiles.open(expanded_path, "r", encoding="utf-8") as file_handle:
        original = await file_handle.read()

    count = original.count(old_text)
    if count == 0:
        raise ValueError(f"{old_text} not found in {expanded_path}; no changes made")
    if not replace_all and count > 1:
        raise ValueError(f"{old_text} occurs multiple times in {expanded_path}; edit is not unique")

    replace_count = -1 if replace_all else 1
    updated = original.replace(old_text, new_text, replace_count)

    async with aiofiles.open(expanded_path, "w", encoding="utf-8") as file_handle:
        await file_handle.write(updated)

    diff_lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=str(expanded_path),
        tofile=str(expanded_path),
        lineterm="",
    )
    return "\n".join(diff_lines)


def create_filesystem_tools() -> list[Tool]:
    """Create the local filesystem tools."""

    async def execute_write(validated: FilesystemWriteFileInput) -> str:
        return await write_file(Path(validated.path), validated.content)

    async def execute_edit(validated: FilesystemEditFileInput) -> str:
        return await edit_file(
            Path(validated.path),
            validated.old_text,
            validated.new_text,
            replace_all=validated.replace_all,
        )

    return [
        StructuredTool(
            name="filesystem_write_file",
            description="Create or overwrite a file with the exact content you provide.",
            schema_model=FilesystemWriteFileInput,
            handler=execute_write,
        ),
        StructuredTool(
            name="filesystem_edit_file",
            description="Apply a targeted text replacement to a file and return the resulting unified diff.",
            schema_model=FilesystemEditFileInput,
            handler=execute_edit,
        ),
    ]
