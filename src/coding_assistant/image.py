"""Image loading and processing utilities."""

import base64
import io
import pathlib
from urllib.parse import urlparse

import httpx
from PIL import Image


async def get_image(path_or_url: str) -> str:
    """Load image from local file or URL, downscale if needed, convert to JPEG."""
    max_dimension = 1024
    is_url = urlparse(path_or_url).scheme in ("http", "https")

    if is_url:
        async with httpx.AsyncClient() as client:
            response = await client.get(path_or_url, timeout=15.0)
            response.raise_for_status()
            content = response.content
    else:
        path = pathlib.Path(path_or_url).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "rb") as file:
            content = file.read()

    with Image.open(io.BytesIO(content)) as image:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")  # type: ignore[assignment]
        width, height = image.size
        if width > max_dimension or height > max_dimension:
            ratio = max_dimension / max(width, height)
            resized = (int(width * ratio), int(height * ratio))
            image = image.resize(resized, Image.Resampling.LANCZOS)  # type: ignore[assignment]
        output = io.BytesIO()
        image.save(output, format="JPEG", optimize=True, quality=85)
        content = output.getvalue()

    encoded_string = base64.b64encode(content).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_string}"
