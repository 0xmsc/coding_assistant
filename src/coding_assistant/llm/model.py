import os
import litellm


def complete(
    mesages: list[dict],
    tools: list = [],
    model: str = "o4-mini",
) -> dict:
    return litellm.completion(
        messages=mesages,
        tools=tools,
        model=model,
    )
