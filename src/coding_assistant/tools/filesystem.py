from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import aiofiles
from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool


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


class FilesystemWriteFileTool(Tool):
    """Create or overwrite a file with caller-provided content."""

    def name(self) -> str:
        return "filesystem_write_file"

    def description(self) -> str:
        return "Create or overwrite a file with the exact content you provide."

    def parameters(self) -> dict[str, Any]:
        return FilesystemWriteFileInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = FilesystemWriteFileInput.model_validate(parameters)
        return TextToolResult(content=await write_file(Path(validated.path), validated.content))


class FilesystemEditFileTool(Tool):
    """Apply one validated text replacement to a file."""

    def name(self) -> str:
        return "filesystem_edit_file"

    def description(self) -> str:
        return "Apply a targeted text replacement to a file and return the resulting unified diff."

    def parameters(self) -> dict[str, Any]:
        return FilesystemEditFileInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = FilesystemEditFileInput.model_validate(parameters)
        return TextToolResult(
            content=await edit_file(
                Path(validated.path),
                validated.old_text,
                validated.new_text,
                replace_all=validated.replace_all,
            ),
        )


async def write_file(path: Path, content: str) -> str:
    """Overwrite or create a file with the given content."""
    expanded_path = path.expanduser()
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(expanded_path, "w", encoding="utf-8") as file_handle:
        await file_handle.write(content)

    return f"Successfully wrote file {expanded_path}"


async def edit_file(path: Path, old_text: str, new_text: str, replace_all: bool = False) -> str:
    """Apply one validated text replacement to a file and return a clear diff summary."""
    expanded_path = path.expanduser()
    async with aiofiles.open(expanded_path, "r", encoding="utf-8") as file_handle:
        original = await file_handle.read()

    count = original.count(old_text)
    if count == 0:
        raise ValueError(f"{old_text!r} not found in {expanded_path}; no changes made")
    if not replace_all and count > 1:
        raise ValueError(f"{old_text!r} occurs multiple times in {expanded_path}; edit is not unique")

    replace_count = -1 if replace_all else 1
    updated = original.replace(old_text, new_text, replace_count)

    async with aiofiles.open(expanded_path, "w", encoding="utf-8") as file_handle:
        await file_handle.write(updated)

    # Generate summary
    summary = _format_edit_summary(expanded_path, old_text, new_text, replace_all, count)

    # Generate unified diff
    diff_lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=str(expanded_path),
        tofile=str(expanded_path),
        lineterm="",
    )
    diff = "\n".join(diff_lines)

    return f"{summary}\n\n{diff}"


def _format_edit_summary(
    path: Path,
    old_text: str,
    new_text: str,
    replace_all: bool,
    match_count: int,
) -> str:
    """Format a clear summary of what was changed."""
    lines = []

    # Header with file path
    lines.append(f"Applied edit to {path}:")

    # Truncate for display
    old_display = _truncate_for_display(old_text)
    new_display = _truncate_for_display(new_text)
    lines.append(f"- Old: {old_display}")
    lines.append(f"+ New: {new_display}")

    # Show replacement scope
    if replace_all:
        lines.append(f"  (replaced all {match_count} occurrences)")
    elif match_count > 1:
        lines.append(f"  (1 of {match_count} occurrences replaced)")

    return "\n".join(lines)


def _truncate_for_display(text: str, max_length: int = 100) -> str:
    """Truncate text for display, showing first and last portions if too long."""
    # Replace newlines with visible markers
    single_line = text.replace("\n", "\\n")
    if len(single_line) <= max_length:
        return f'"{single_line}"'

    # Show start and end for longer texts
    prefix_len = (max_length - 5) // 2
    return f'"{single_line[:prefix_len]} ... {single_line[-prefix_len:]}"'


def create_filesystem_tools() -> list[Tool]:
    """Create the local filesystem tools."""
    return [
        FilesystemWriteFileTool(),
        FilesystemEditFileTool(),
    ]
