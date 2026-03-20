import base64
import io
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from coding_assistant.image import get_image


@pytest.fixture
def small_image() -> bytes:
    image = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def large_image() -> bytes:
    image = Image.new("RGB", (3000, 2000), color="blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_get_image_local_small(small_image: bytes, tmp_path: Path) -> None:
    file_path = tmp_path / "small.jpg"
    file_path.write_bytes(small_image)

    data_uri = await get_image(str(file_path))

    assert data_uri.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as image:
        assert image.size[0] <= 1024 and image.size[1] <= 1024


@pytest.mark.asyncio
async def test_get_image_local_large_downscaled(large_image: bytes, tmp_path: Path) -> None:
    file_path = tmp_path / "large.png"
    file_path.write_bytes(large_image)

    data_uri = await get_image(str(file_path))

    assert data_uri.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as image:
        assert max(image.size) <= 1024
        assert abs((image.size[0] / image.size[1]) - 1.5) < 0.01


@pytest.mark.asyncio
async def test_get_image_url_success(small_image: bytes) -> None:
    mock_response = MagicMock()
    mock_response.content = small_image
    mock_response.headers = {"content-type": "image/jpeg"}

    with patch("coding_assistant.image.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        data_uri = await get_image("https://example.com/image.jpg")

        assert data_uri.startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_get_image_url_failure() -> None:
    with patch("coding_assistant.image.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Network error"))

        with pytest.raises(Exception, match="Network error"):
            await get_image("https://example.com/missing.jpg")


@pytest.mark.asyncio
async def test_get_image_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError, match="File not found"):
        await get_image("/tmp/nonexistent_image_12345.jpg")


@pytest.mark.asyncio
async def test_get_image_invalid_image(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.txt"
    file_path.write_text("This is not an image")

    with pytest.raises(Exception):
        await get_image(str(file_path))


@pytest.mark.asyncio
async def test_get_image_png_to_jpeg(tmp_path: Path) -> None:
    image = Image.new("RGBA", (200, 200), color=(255, 0, 0, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    file_path = tmp_path / "alpha.png"
    file_path.write_bytes(buffer.getvalue())

    data_uri = await get_image(str(file_path))
    assert data_uri.startswith("data:image/jpeg;base64,")
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as result:
        assert result.mode == "RGB"
        assert result.size == (200, 200)


@pytest.mark.asyncio
async def test_get_image_tilde_expansion(small_image: bytes, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    home_file = tmp_path / "test.jpg"
    home_file.write_bytes(small_image)
    data_uri = await get_image("~/test.jpg")
    assert data_uri.startswith("data:image/jpeg;base64,")
