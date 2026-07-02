import base64
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from coding_assistant.llm.types import ToolMessageResult
from coding_assistant.tools.load_image import MAX_IMAGE_EDGE, LoadImageTool


def _image_bytes(*, image_format: str, size: tuple[int, int], mode: str = "RGB") -> bytes:
    buffer = BytesIO()
    color = (255, 0, 0, 128) if mode == "RGBA" else (255, 0, 0)
    Image.new(mode, size, color).save(buffer, format=image_format)
    return buffer.getvalue()


def _image_url_data(result: ToolMessageResult) -> tuple[str, bytes]:
    assert isinstance(result.content, list)
    image_block = result.content[1]
    assert image_block["type"] == "image_url"
    url = image_block["image_url"]["url"]
    assert isinstance(url, str)
    prefix, encoded = url.split(",", 1)
    return prefix, base64.b64decode(encoded)


def _decoded_size(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as image:
        return image.size


@pytest.mark.asyncio
async def test_load_image_loads_image_file_as_model_context(tmp_path: Path) -> None:
    image_file = tmp_path / "attachments" / "att_123-meal.png"
    image_file.parent.mkdir()
    image_bytes = _image_bytes(image_format="PNG", size=(32, 16))
    image_file.write_bytes(image_bytes)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = await LoadImageTool(workspace=workspace).execute({"path": str(image_file)})

    assert isinstance(result, ToolMessageResult)
    assert isinstance(result.content, list)
    assert result.content[0] == {
        "type": "text",
        "text": f"Loaded image file {image_file.as_posix()} (image/png, {len(image_bytes)} bytes, 32x16).",
    }
    prefix, data = _image_url_data(result)
    assert prefix == "data:image/png;base64"
    assert data == image_bytes


@pytest.mark.asyncio
async def test_load_image_resizes_large_image_for_model_context(tmp_path: Path) -> None:
    image_file = tmp_path / "attachments" / "att_123-meal.jpg"
    image_file.parent.mkdir()
    original_data = _image_bytes(image_format="JPEG", size=(3200, 2400))
    image_file.write_bytes(original_data)

    result = await LoadImageTool(workspace=tmp_path).execute({"path": str(image_file)})

    assert isinstance(result, ToolMessageResult)
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "text"
    assert result.content[0]["text"].startswith("Loaded image file attachments/att_123-meal.jpg (image/jpeg, ")
    assert f"{MAX_IMAGE_EDGE}x1200, resized from 3200x2400)." in result.content[0]["text"]
    prefix, data = _image_url_data(result)
    assert prefix == "data:image/jpeg;base64"
    assert _decoded_size(data) == (MAX_IMAGE_EDGE, 1200)
    assert image_file.read_bytes() == original_data


@pytest.mark.asyncio
async def test_load_image_resized_transparent_image_stays_png(tmp_path: Path) -> None:
    image_file = tmp_path / "transparent.png"
    original_data = _image_bytes(image_format="PNG", size=(2000, 1000), mode="RGBA")
    image_file.write_bytes(original_data)

    result = await LoadImageTool(workspace=tmp_path).execute({"path": str(image_file)})

    assert isinstance(result, ToolMessageResult)
    assert isinstance(result.content, list)
    assert result.content[0]["type"] == "text"
    assert f"{MAX_IMAGE_EDGE}x800, resized from 2000x1000" in result.content[0]["text"]
    prefix, data = _image_url_data(result)
    assert prefix == "data:image/png;base64"
    assert _decoded_size(data) == (MAX_IMAGE_EDGE, 800)


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
