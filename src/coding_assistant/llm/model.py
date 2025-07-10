import os

import litellm

litellm.telemetry = False
litellm.modify_params = True


async def complete(
    messages: list[dict],
    tools: list = [],
    model: str = "o4-mini",
):
    completion = await litellm.acompletion(
        messages=messages,
        tools=tools,
        model=model,
        drop_params=True,
        merge_reasoning_content_in_choices=True,
    )

    return completion["choices"][0]["message"]
