import os
import litellm


def complete(
    messages: list[dict],
    tools: list = [],
    model: str = "o4-mini",
) -> dict:
    return litellm.completion(
        messages=messages,
        tools=tools,
        model=model,
        reasoning_effort="high",
    )
