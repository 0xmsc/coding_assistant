from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool, ToolMessageResult, ToolResult

MAX_TEXT_BYTES = 512 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGE_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}
SUPPORTED_TEXT_TYPES = {
    "application/csv",
    "application/json",
    "application/markdown",
    "application/x-ndjson",
    "text/csv",
    "text/markdown",
}


class LoadFileInput(BaseModel):
    path: str = Field(
        description=(
            "Path to a session attachment, such as attachments/att_123-meal.jpg. "
            "Only files under the session attachments directory can be loaded."
        ),
    )
    mode: Literal["auto", "text", "image"] = Field(
        default="auto",
        description="How to load the file. Use auto unless the file extension is misleading.",
    )


class LoadFileTool(Tool):
    """Load a bounded workspace attachment into conversation context."""

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    def name(self) -> str:
        return "load_file"

    def description(self) -> str:
        return (
            "Load a text or image file from the session attachments directory into conversation context. "
            "Use this before reasoning from an uploaded file."
        )

    def parameters(self) -> dict[str, Any]:
        return LoadFileInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        validated = LoadFileInput.model_validate(parameters)
        target = _attachment_path(workspace=self._workspace, raw_path=validated.path)
        data = target.read_bytes()
        mime_type = _detect_mime_type(target, data)

        if validated.mode == "image" or (validated.mode == "auto" and mime_type in SUPPORTED_IMAGE_TYPES):
            return _load_image(path=target, workspace=self._workspace, data=data, mime_type=mime_type)
        if validated.mode == "text" or (validated.mode == "auto" and _is_text_mime(mime_type)):
            return _load_text(path=target, workspace=self._workspace, data=data, mime_type=mime_type)
        raise ValueError(f"Unsupported attachment type for {target.relative_to(self._workspace)}: {mime_type}.")


def _attachment_path(*, workspace: Path, raw_path: str) -> Path:
    if not raw_path.strip():
        raise ValueError("load_file requires a non-empty path.")

    attachments_root = (workspace / "attachments").resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=True)

    try:
        resolved.relative_to(attachments_root)
    except ValueError as exc:
        raise ValueError("load_file can only read files under the session attachments directory.") from exc
    if not resolved.is_file():
        raise ValueError(f"Attachment path is not a file: {resolved.relative_to(workspace)}.")
    return resolved


def _detect_mime_type(path: Path, data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _is_text_mime(mime_type: str) -> bool:
    return mime_type.startswith("text/") or mime_type in SUPPORTED_TEXT_TYPES


def _load_text(*, path: Path, workspace: Path, data: bytes, mime_type: str) -> TextToolResult:
    if len(data) > MAX_TEXT_BYTES:
        raise ValueError(f"Text attachment is too large to load: {path.relative_to(workspace)}.")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Text attachment is not valid UTF-8: {path.relative_to(workspace)}.") from exc

    relative_path = path.relative_to(workspace).as_posix()
    content = f"Loaded text attachment {relative_path} ({mime_type}, {len(data)} bytes):\n\n{text}"
    return TextToolResult(content=content)


def _load_image(*, path: Path, workspace: Path, data: bytes, mime_type: str) -> ToolMessageResult:
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image attachment type for {path.relative_to(workspace)}: {mime_type}.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image attachment is too large to load: {path.relative_to(workspace)}.")

    relative_path = path.relative_to(workspace).as_posix()
    text = f"Loaded image attachment {relative_path} ({mime_type}, {len(data)} bytes)."
    return ToolMessageResult(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}",
                },
            },
        ],
    )


def create_load_file_tools(*, workspace: Path) -> list[Tool]:
    """Create tools that load files from the worker workspace."""
    return [LoadFileTool(workspace=workspace)]
