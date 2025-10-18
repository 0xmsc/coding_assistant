from __future__ import annotations

import difflib
import re
from dataclasses import astuple, dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Annotated, Optional

from fastmcp import FastMCP


@dataclass
class ClipboardEntry:
    lines: list[str]


@dataclass
class EditRecord:
    path: Path
    old_lines: list[str]
    new_lines: list[str]


@dataclass
class Region:
    start: int  # inclusive, 0-based line index
    end: int  # inclusive, 0-based line index

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class CopyCutResult:
    clipped_lines: list[str]
    edit: EditRecord | None = None


class EditMode(Enum):
    BEFORE = auto()
    AFTER = auto()
    REPLACE = auto()


def _join_lines(lines: list[str]) -> str:
    return "\n".join(lines)


def _unified_diff(path: Path, edit: EditRecord) -> str:
    diff_iter = difflib.unified_diff(
        edit.old_lines,
        edit.new_lines,
        fromfile=str(path),
        tofile=str(path),
        lineterm="",
    )
    return "\n".join(diff_iter)


def _find_matching_region(
    content: str,
    pattern: str,
    *,
    enforce_unique_match: bool = True,
) -> Region:
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    if not matches:
        raise ValueError(f"No match found for {pattern}")
    if enforce_unique_match and len(matches) != 1:
        raise ValueError(f"No unique match found for {pattern}.")

    m = matches[0]
    start_line = content.count("\n", 0, m.start())
    end_line = content.count("\n", 0, m.end())

    return Region(start=start_line, end=end_line)


def _copy_cut_range(
    path: Path,
    start: str,
    enforce_unique_start: bool,
    cut: bool,
) -> CopyCutResult:
    content = path.read_text(encoding="utf-8")
    file_lines = content.splitlines()

    region = _find_matching_region(content, start, enforce_unique_match=enforce_unique_start)
    clipped_lines = file_lines[region.start : region.end + 1]

    if not cut:
        return CopyCutResult(clipped_lines=clipped_lines)

    new_lines = file_lines[: region.start] + file_lines[region.end + 1 :]
    path.write_text(_join_lines(new_lines) + "\n", encoding="utf-8")

    return CopyCutResult(clipped_lines=clipped_lines, edit=EditRecord(path, file_lines, new_lines))


def _edit(
    path: Path,
    pattern: str,
    insert_lines: list[str],
    mode: EditMode,
    enforce_unique_start: bool,
) -> EditRecord:
    content = path.read_text(encoding="utf-8")
    file_lines = content.splitlines()

    region = _find_matching_region(content, pattern, enforce_unique_match=enforce_unique_start)

    if mode == EditMode.BEFORE:
        updated_lines = file_lines[: region.start] + insert_lines + file_lines[region.start :]
    elif mode == EditMode.AFTER:
        updated_lines = file_lines[: region.end + 1] + insert_lines + file_lines[region.end + 1 :]
    elif mode == EditMode.REPLACE:
        updated_lines = file_lines[: region.start] + insert_lines + file_lines[region.end + 1 :]

    path.write_text(_join_lines(updated_lines) + "\n", encoding="utf-8")

    return EditRecord(path, file_lines, updated_lines)


class FilesystemManager:
    def __init__(self) -> None:
        self._entry: Optional[ClipboardEntry] = None
        self._last_edit: Optional[EditRecord] = None

    def _get_clipboard_text(self) -> str:
        if not self._entry:
            raise ValueError("Clipboard is empty.")
        return _join_lines(self._entry.lines)

    def copy_range(
        self,
        path: Annotated[Path, "File path."],
        pattern: Annotated[str, "Regex for range."],
        enforce_unique_match: Annotated[bool, "Fail if match is not unique."] = True,
    ) -> str:
        result = _copy_cut_range(path, pattern, enforce_unique_match, cut=False)
        self._entry = ClipboardEntry(lines=result.clipped_lines)
        return f"Copied\n\n{self._get_clipboard_text()}"

    def cut_range(
        self,
        path: Annotated[Path, "File path."],
        pattern: Annotated[str, "Regex for range."],
        enforce_unique_match: Annotated[bool, "Fail if match is not unique."] = True,
    ) -> str:
        result = _copy_cut_range(path, pattern, enforce_unique_match, cut=True)
        self._entry = ClipboardEntry(lines=result.clipped_lines)
        assert result.edit
        self._last_edit = result.edit
        return f"Cut\n\n{_join_lines(result.clipped_lines)}\n\nwith diff\n\n{_unified_diff(path, self._last_edit)}"

    def paste(
        self,
        path: Annotated[Path, "File path."],
        pattern: Annotated[str, "Regex for range."],
        position: Annotated[str, "One of 'before', 'after', 'replace'."],
        enforce_unique_match: Annotated[bool, "Fail if match is not unique."] = True,
    ) -> str:
        if not self._entry:
            raise ValueError("Clipboard is empty.")
        result = _edit(path, pattern, self._entry.lines, EditMode[position.upper()], enforce_unique_match)
        self._last_edit = result
        return f"Pasted\n\n{_join_lines(self._entry.lines)}\n\nwith diff\n\n{_unified_diff(path, result)}"

    def undo_last_edit(self) -> str:
        """Undo the last mutating edit."""
        if not self._last_edit:
            return "Nothing to undo."

        last_edit = self._last_edit
        current_content = last_edit.path.read_text(encoding="utf-8")
        if current_content.splitlines() != last_edit.new_lines:
            raise ValueError("Cannot undo: file contents have changed since the last edit.")

        last_edit.path.write_text(_join_lines(last_edit.old_lines) + "\n", encoding="utf-8")
        return _unified_diff(last_edit.path, EditRecord(last_edit.path, last_edit.new_lines, last_edit.old_lines))

    def show_clipboard(self) -> str:
        return self._get_clipboard_text()

    def clear_clipboard(self) -> str:
        self._entry = None
        return "Cleared clipboard."
    def write_file(
        self,
        path: Annotated[Path, "File path, parent directories created as needed."],
        content: Annotated[str, "Content to write; replaces entire file."],
    ) -> str:
        old_contents = path.read_text(encoding="utf-8") if path.exists() else ""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Enforce a trailing newline for consistency with edit operations
        content_to_write = content if content.endswith("\n") else content + "\n"
        path.write_text(content_to_write, encoding="utf-8")
        self._last_edit = EditRecord(path, old_contents.splitlines(), content_to_write.splitlines())
        return _unified_diff(path, self._last_edit)
    def edit_file(
        self,
        path: Annotated[Path, "File path."],
        pattern: Annotated[str, "Regex for range."],
        text: Annotated[str, "Text to insert."],
        position: Annotated[str, "One of 'before', 'after', 'replace'."],
        enforce_unique_match: Annotated[bool, "Fail if match is not unique."] = True,
    ) -> str:
        result = _edit(path, pattern, text.splitlines(), EditMode[position.upper()], enforce_unique_match)
        self._last_edit = result
        return f"Edited with diff\n\n{_unified_diff(path, self._last_edit)}"


def create_filesystem_server() -> FastMCP:
    manager = FilesystemManager()
    server = FastMCP()

    server.tool(manager.write_file)
    server.tool(manager.edit_file)

    server.tool(manager.copy_range)
    server.tool(manager.cut_range)
    server.tool(manager.paste)
    server.tool(manager.undo_last_edit)
    server.tool(manager.show_clipboard)
    server.tool(manager.clear_clipboard)

    return server
