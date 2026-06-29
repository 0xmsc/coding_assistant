from pathlib import Path

import pytest

from coding_assistant.llm.types import ToolMessageResult
from coding_assistant.tools.load_image import LoadImageTool

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.mark.asyncio
async def test_load_image_loads_image_file_as_model_context(tmp_path: Path) -> None:
    image_file = tmp_path / "attachments" / "att_123-meal.png"
    image_file.parent.mkdir()
    image_file.write_bytes(PNG_BYTES)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = await LoadImageTool(workspace=workspace).execute({"path": str(image_file)})

    assert isinstance(result, ToolMessageResult)
    assert isinstance(result.content, list)
    assert result.content[0] == {
        "type": "text",
        "text": f"Loaded image file {image_file.as_posix()} (image/png, 24 bytes).",
    }
    image_block = result.content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_load_image_rejects_unsupported_binary_file(tmp_path: Path) -> None:
    binary_file = tmp_path / "archive.bin"
    binary_file.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="Unsupported image file type"):
        await LoadImageTool(workspace=tmp_path).execute({"path": "archive.bin"})


@pytest.mark.asyncio
async def test_load_image_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="load_image requires a non-empty path"):
        await LoadImageTool(workspace=tmp_path).execute({"path": ""})
