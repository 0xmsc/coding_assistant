import os
import litellm


def complete(
    mesages: list[dict],
    model: str = "o4-mini",
) -> dict:
    return litellm.completion(
        messages=mesages,
        model=model,
    )
