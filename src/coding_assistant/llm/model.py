from dataclasses import dataclass
import logging
import os

import litellm

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True
litellm.drop_params = True


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
        response = await litellm.acompletion(
            messages=messages,
            tools=tools,
            model=model,
            stream=True,
        )

        chunks = []
        async for chunk in response:
            chunks.append(chunk)

        completion = litellm.stream_chunk_builder(chunks)

        return Completion(
            message=completion.choices[0].message,
            tokens=completion.usage.total_tokens,
        )
    except Exception as e:
        logger.error(f"Error during model completion: {e}, last messages: {messages[-5:]}")
        raise e
