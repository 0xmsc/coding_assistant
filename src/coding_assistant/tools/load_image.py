from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool, ToolMessageResult, ToolResult

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_EDGE = 1600
JPEG_QUALITY = 85
SUPPORTED_IMAGE_TYPES = {"image/gif", "image/jpeg", "image/png", "image/webp"}


@dataclass(frozen=True)
class PreparedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    original_width: int
    original_height: int


class LoadImageInput(BaseModel):
    path: str = Field(
        description=(
            "Path to a readable image file. Relative paths are resolved from the worker workspace; "
            "absolute paths may be used for files visible to the worker."
        ),
    )
    mode: Literal["auto", "image"] = Field(
        default="auto",
        description="How to load the image. Use auto unless the file extension is misleading.",
    )


class LoadImageTool(Tool):
    """Load a bounded image file into conversation context."""

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace.resolve()

    def name(self) -> str:
        return "load_image"

    def description(self) -> str:
        return (
            "Load a readable image file into conversation context. "
            "Use this before reasoning from an uploaded image or another worker-visible image."
        )

    def parameters(self) -> dict[str, Any]:
        return LoadImageInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        validated = LoadImageInput.model_validate(parameters)
        target = _file_path(workspace=self._workspace, raw_path=validated.path)
        data = target.read_bytes()
        mime_type = _detect_mime_type(target, data)

        if validated.mode == "image" or (validated.mode == "auto" and mime_type in SUPPORTED_IMAGE_TYPES):
            return _load_image(path=target, workspace=self._workspace, data=data, mime_type=mime_type)
        raise ValueError(
            f"Unsupported image file type for {_display_path(path=target, workspace=self._workspace)}: {mime_type}."
        )


def _file_path(*, workspace: Path, raw_path: str) -> Path:
    if not raw_path.strip():
        raise ValueError("load_image requires a non-empty path.")

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=True)

    if not resolved.is_file():
        raise ValueError(f"Path is not a file: {_display_path(path=resolved, workspace=workspace)}.")
    return resolved


def _display_path(*, path: Path, workspace: Path) -> str:
    try:
        return path.relative_to(workspace).as_posix()
    except ValueError:
        return path.as_posix()


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


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info)


def _encoded_image(image: Image.Image) -> tuple[str, bytes]:
    buffer = BytesIO()
    if _has_alpha(image):
        image.convert("RGBA").save(buffer, format="PNG", optimize=True)
        return "image/png", buffer.getvalue()

    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return "image/jpeg", buffer.getvalue()


def _prepare_image(*, data: bytes, mime_type: str, display_path: str) -> PreparedImage:
    try:
        with Image.open(BytesIO(data)) as raw_image:
            original_width, original_height = raw_image.size
            if original_width <= 0 or original_height <= 0:
                raise ValueError(f"Image file has invalid dimensions: {display_path}.")

            should_process = max(original_width, original_height) > MAX_IMAGE_EDGE or len(data) > MAX_IMAGE_BYTES
            if not should_process:
                return PreparedImage(
                    data=data,
                    mime_type=mime_type,
                    width=original_width,
                    height=original_height,
                    original_width=original_width,
                    original_height=original_height,
                )

            image = ImageOps.exif_transpose(raw_image)
            image.load()
            original_width, original_height = image.size
            if max(image.size) > MAX_IMAGE_EDGE:
                image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)

            output_mime_type, output_data = _encoded_image(image)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not decode image file: {display_path}.") from exc
    except OSError as exc:
        raise ValueError(f"Could not decode image file: {display_path}.") from exc

    if len(output_data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Processed image is too large to load: {display_path}.")

    return PreparedImage(
        data=output_data,
        mime_type=output_mime_type,
        width=image.width,
        height=image.height,
        original_width=original_width,
        original_height=original_height,
    )


def _load_image(*, path: Path, workspace: Path, data: bytes, mime_type: str) -> ToolMessageResult:
    display_path = _display_path(path=path, workspace=workspace)
    if mime_type not in SUPPORTED_IMAGE_TYPES:
        raise ValueError(f"Unsupported image file type for {display_path}: {mime_type}.")

    image = _prepare_image(data=data, mime_type=mime_type, display_path=display_path)
    text = f"Loaded image file {display_path} ({image.mime_type}, {len(image.data)} bytes, {image.width}x{image.height}"
    if (image.width, image.height) != (image.original_width, image.original_height):
        text += f", resized from {image.original_width}x{image.original_height}"
    if image.mime_type != mime_type:
        text += f", re-encoded from {mime_type}"
    text += ")."

    return ToolMessageResult(
        content=[
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image.mime_type};base64,{base64.b64encode(image.data).decode('ascii')}",
                },
            },
        ],
    )


def create_load_image_tools(*, workspace: Path) -> list[Tool]:
    """Create tools that load images from the worker workspace."""
    return [LoadImageTool(workspace=workspace)]
