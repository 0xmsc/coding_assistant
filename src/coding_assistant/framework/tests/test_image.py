"""Tests for image.py"""
import io
import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from PIL import Image

from coding_assistant.framework.image import get_image


@pytest.fixture
def small_image() -> bytes:
    """Create a small test image (100x100)."""
    img = Image.new('RGB', (100, 100), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    return buf.getvalue()


@pytest.fixture
def large_image() -> bytes:
    """Create a large test image (3000x2000)."""
    img = Image.new('RGB', (3000, 2000), color='blue')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.mark.asyncio
async def test_get_image_local_small(small_image: bytes, tmp_path: Path) -> None:
    """Local image that does not need downsampling."""
    file_path = tmp_path / "small.jpg"
    file_path.write_bytes(small_image)
    
    data_uri, mime = await get_image(str(file_path))
    
    assert mime == "image/jpeg"
    assert data_uri.startswith("data:image/jpeg;base64,")
    # Ensure the image is still roughly the same size (maybe compressed)
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as img:
        # Should still be ~100x100 (maybe slightly different due to JPEG)
        assert img.size[0] <= 1024 and img.size[1] <= 1024


@pytest.mark.asyncio
async def test_get_image_local_large_downscaled(large_image: bytes, tmp_path: Path) -> None:
    """Large local image that should be downsampled to max 1024px."""
    file_path = tmp_path / "large.png"
    file_path.write_bytes(large_image)
    
    data_uri, mime = await get_image(str(file_path))
    
    assert mime == "image/jpeg"
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as img:
        # Should be downsampled to max 1024px (aspect ratio preserved)
        max_dim = max(img.size)
        assert max_dim <= 1024
        # Aspect ratio should be 3000/2000 = 1.5
        ratio = img.size[0] / img.size[1]
        assert abs(ratio - 1.5) < 0.01


@pytest.mark.asyncio
async def test_get_image_url_success(small_image: bytes) -> None:
    """URL that returns an image."""
    # Mock httpx.AsyncClient
    mock_response = MagicMock()
    mock_response.content = small_image
    mock_response.headers = {"content-type": "image/jpeg"}
    
    with patch("coding_assistant.framework.image.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
        
        data_uri, mime = await get_image("https://example.com/image.jpg")
        
        assert mime == "image/jpeg"
        assert data_uri.startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_get_image_url_failure() -> None:
    """URL that returns an error."""
    with patch("coding_assistant.framework.image.httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Network error"))
        
        with pytest.raises(Exception, match="Network error"):
            await get_image("https://example.com/missing.jpg")


@pytest.mark.asyncio
async def test_get_image_nonexistent_file() -> None:
    """Non-existent local file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        await get_image("/tmp/nonexistent_image_12345.jpg")


@pytest.mark.asyncio
async def test_get_image_invalid_image(tmp_path: Path) -> None:
    """Invalid image file (not a valid image)."""
    file_path = tmp_path / "invalid.txt"
    file_path.write_text("This is not an image")
    
    with pytest.raises(Exception):
        await get_image(str(file_path))


@pytest.mark.asyncio
async def test_get_image_png_to_jpeg(tmp_path: Path) -> None:
    """PNG image should be converted to JPEG."""
    img = Image.new('RGBA', (200, 200), color=(255, 0, 0, 128))  # Red with alpha
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    file_path = tmp_path / "alpha.png"
    file_path.write_bytes(buf.getvalue())
    
    data_uri, mime = await get_image(str(file_path))
    assert mime == "image/jpeg"
    decoded = base64.b64decode(data_uri.split(",")[1])
    with Image.open(io.BytesIO(decoded)) as result:
        # Should be RGB, not RGBA
        assert result.mode == "RGB"
        assert result.size == (200, 200)