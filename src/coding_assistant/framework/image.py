import httpx
from typing import Any, cast


async def get_image(path_or_url: str) -> str:
    from coding_assistant.framework.actors.agent import image_io

    # Keep compatibility with callers/tests that patch framework.image.httpx.
    cast(Any, image_io).httpx = httpx

    return await image_io.get_image(path_or_url)
