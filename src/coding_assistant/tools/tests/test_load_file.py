from pathlib import Path

import pytest

from coding_assistant.llm.types import ToolContextResult, UserMessage
from coding_assistant.tools.load_file import LoadFileTool

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


@pytest.mark.asyncio
async def test_load_file_loads_text_attachment(tmp_path: Path) -> None:
    attachment = tmp_path / "attachments" / "att_123-menu.txt"
    attachment.parent.mkdir()
    attachment.write_text("eggs and toast", encoding="utf-8")

    result = await LoadFileTool(workspace=tmp_path).execute({"path": "attachments/att_123-menu.txt"})

    assert isinstance(result, ToolContextResult)
    assert "Loaded text attachment attachments/att_123-menu.txt" in result.content
    assert "eggs and toast" in result.content
    assert result.extra_messages == [UserMessage(content=result.content)]


@pytest.mark.asyncio
async def test_load_file_loads_image_attachment_as_model_context(tmp_path: Path) -> None:
    attachment = tmp_path / "attachments" / "att_123-meal.png"
    attachment.parent.mkdir()
    attachment.write_bytes(PNG_BYTES)

    result = await LoadFileTool(workspace=tmp_path).execute({"path": "attachments/att_123-meal.png"})

    assert result.content == (
        "Loaded image attachment attachments/att_123-meal.png (image/png, 24 bytes). "
        "The image is now available in conversation context."
    )
    [message] = result.extra_messages
    assert isinstance(message, UserMessage)
    assert isinstance(message.content, list)
    assert message.content[0] == {
        "type": "text",
        "text": "Loaded image attachment attachments/att_123-meal.png (image/png, 24 bytes).",
    }
    image_block = message.content[1]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_load_file_rejects_paths_outside_attachments(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(ValueError, match="session attachments directory"):
        await LoadFileTool(workspace=tmp_path).execute({"path": str(outside)})


@pytest.mark.asyncio
async def test_load_file_rejects_attachment_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    attachments = tmp_path / "attachments"
    attachments.mkdir()
    (attachments / "escape.txt").symlink_to(outside)

    with pytest.raises(ValueError, match="session attachments directory"):
        await LoadFileTool(workspace=tmp_path).execute({"path": "attachments/escape.txt"})


@pytest.mark.asyncio
async def test_load_file_rejects_unsupported_binary_attachment(tmp_path: Path) -> None:
    attachment = tmp_path / "attachments" / "archive.bin"
    attachment.parent.mkdir()
    attachment.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="Unsupported attachment type"):
        await LoadFileTool(workspace=tmp_path).execute({"path": "attachments/archive.bin"})
