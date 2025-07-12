from dataclasses import dataclass
import logging
import os

import litellm

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True


@dataclass
class Completion:
    message: litellm.Message
    tokens: int


async def complete(
    messages: list[dict],
    model: str,
    tools: list = [],
):
    try:
        completion = await litellm.acompletion(
            messages=messages,
            tools=tools,
            model=model,
            drop_params=True,
        )

        if not completion["choices"]:
            raise RuntimeError(f"No choices returned from the model: {completion}")

        return Completion(message=completion["choices"][0]["message"], tokens=completion["usage"]["total_tokens"])
    except Exception as e:
        logger.error(f"Error during model completion: {e}, last messages: {messages[-5:]}")
        raise e
