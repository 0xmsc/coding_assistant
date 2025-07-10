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
        merge_reasoning_content_in_choices=True,
    )

    if not completion["choices"]:
        error = f"No choices returned from the model: {completion}"
        logger.error(error)
        raise RuntimeError(error)

    return completion["choices"][0]["message"]
