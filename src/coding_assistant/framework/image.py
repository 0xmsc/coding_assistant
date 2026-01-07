"""Image loading and processing utilities."""
import base64
import io
import pathlib
from urllib.parse import urlparse

import httpx
from PIL import Image


async def get_image(path_or_url: str) -> tuple[str, str]:
    """Load image from local file or URL, downscale if needed, convert to JPEG."""
    MAX_DIMENSION = 1024

    is_url = urlparse(path_or_url).scheme in ('http', 'https')
    
    if is_url:
        async with httpx.AsyncClient() as client:
            response = await client.get(path_or_url, timeout=15.0)
            response.raise_for_status()
            content = response.content
    else:
        path = pathlib.Path(path_or_url)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path_or_url}")
        with open(path, "rb") as f:
            content = f.read()
    
    # Convert to JPEG and downscale if needed
    with Image.open(io.BytesIO(content)) as img:
        # Convert to RGB (remove alpha)
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')  # type: ignore
        width, height = img.size
        if width > MAX_DIMENSION or height > MAX_DIMENSION:
            ratio = MAX_DIMENSION / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            img = img.resize(new_size, Image.LANCZOS)  # type: ignore
        output = io.BytesIO()
        img.save(output, format='JPEG', optimize=True, quality=85)
        content = output.getvalue()
    
    mime_type = 'image/jpeg'
    encoded_string = base64.b64encode(content).decode('ascii')
    data_uri = f"data:{mime_type};base64,{encoded_string}"
    return data_uri, mime_type