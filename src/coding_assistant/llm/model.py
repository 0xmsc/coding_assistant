import os
import litellm


def complete(
    messages: list[dict],
    tools: list = [],
    model: str = "o4-mini",
):
    completion = litellm.completion(
        messages=messages,
        tools=tools,
        model=model,
        reasoning_effort="high",
        drop_params=True,
    )
    return completion["choices"][0]["message"]
