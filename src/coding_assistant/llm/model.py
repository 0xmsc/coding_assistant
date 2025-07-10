import logging
import os

import litellm

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True


async def complete(
    messages: list[dict],
    model: str,
    tools: list = [],
):
    completion = await litellm.acompletion(
        messages=messages,
        tools=tools,
        model=model,
        drop_params=True,
    )

    if not completion["choices"]:
        raise RuntimeError(f"No choices returned from the model: {completion}")

    return completion["choices"][0]["message"]
