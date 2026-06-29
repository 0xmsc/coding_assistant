from pathlib import Path

import pytest

from coding_assistant.llm.types import TextToolResult, ToolMessageResult
from coding_assistant.tools.load_file import LoadFileTool

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.mark.asyncio
async def test_load_file_loads_relative_text_file(tmp_path: Path) -> None:
    text_file = tmp_path / "notes" / "menu.txt"
    text_file.parent.mkdir()
    text_file.write_text("eggs and toast", encoding="utf-8")

    result = await LoadFileTool(workspace=tmp_path).execute({"path": "notes/menu.txt"})

    assert isinstance(result, TextToolResult)
    assert "Loaded text file notes/menu.txt" in result.content
    assert "eggs and toast" in result.content


@pytest.mark.asyncio
async def test_load_file_loads_absolute_text_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside text", encoding="utf-8")

    result = await LoadFileTool(workspace=workspace).execute({"path": str(outside)})

    assert isinstance(result, TextToolResult)
    assert f"Loaded text file {outside.as_posix()}" in result.content
    assert "outside text" in result.content


@pytest.mark.asyncio
async def test_load_file_loads_image_file_as_model_context(tmp_path: Path) -> None:
    image_file = tmp_path / "attachments" / "att_123-meal.png"
    image_file.parent.mkdir()
    image_file.write_bytes(PNG_BYTES)

    result = await LoadFileTool(workspace=tmp_path).execute({"path": "attachments/att_123-meal.png"})

    assert isinstance(result, ToolMessageResult)
    assert isinstance(result.content, list)
    assert result.content[0] == {
        "type": "text",
        "text": "Loaded image file attachments/att_123-meal.png (image/png, 24 bytes).",
    }
    image_block = result.content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_load_file_loads_symlink_targets(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(outside)

    result = await LoadFileTool(workspace=tmp_path).execute({"path": "link.txt"})

    assert isinstance(result, TextToolResult)
    assert "Loaded text file outside.txt" in result.content
    assert "secret" in result.content


@pytest.mark.asyncio
async def test_load_file_rejects_unsupported_binary_file(tmp_path: Path) -> None:
    binary_file = tmp_path / "archive.bin"
    binary_file.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="Unsupported file type"):
        await LoadFileTool(workspace=tmp_path).execute({"path": "archive.bin"})
