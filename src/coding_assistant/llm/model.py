import functools
import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import litellm

from coding_assistant.agents.callbacks import AgentProgressCallbacks

logger = logging.getLogger(__name__)

litellm.telemetry = False
litellm.modify_params = True
litellm.drop_params = True


@dataclass
class Completion:
    message: litellm.Message
    tokens: int


@functools.cache
def _parse_model_and_reasoning(
    model: str,
) -> tuple[str, Literal["low", "medium", "high"] | None]:
    s = model.strip()
    m = re.match(r"^(.+?) \(([^)]*)\)$", s)

    if not m:
        return s, None

    base = m.group(1).strip()
    effort = m.group(2).strip().lower()

    if effort not in ("low", "medium", "high"):
        raise ValueError(f"Invalid reasoning effort level {effort} in {model}")

    return base, effort


async def complete(
    messages: list[dict],
    model: str,
    tools: list,
    callbacks: AgentProgressCallbacks,
):
    try:
        model, reasoning_effort = _parse_model_and_reasoning(model)

        response = await litellm.acompletion(
            messages=messages,
            tools=tools,
            model=model,
            stream=True,
            reasoning_effort=reasoning_effort,
        )

        chunks = []
        async for chunk in response:
            if (
                len(chunk["choices"]) > 0
                and "content" in chunk["choices"][0]["delta"]
                and chunk["choices"][0]["delta"]["content"] is not None
            ):
                content = chunk["choices"][0]["delta"]["content"]
                callbacks.on_chunk(content)

            # Drop created_at so that `ChunkProcessor` does not sort according to it. It seems buggy and seems to create out-of-order chunks.
            chunk._hidden_params.pop("created_at", None)

            chunks.append(chunk)

        callbacks.on_chunks_end()

        completion = litellm.stream_chunk_builder(chunks)

        return Completion(
            message=completion["choices"][0]["message"],
            tokens=completion["usage"]["total_tokens"],
        )
    except Exception as e:
        logger.error(f"Error during model completion: {e}, last messages: {messages[-5:]}")
        raise e
