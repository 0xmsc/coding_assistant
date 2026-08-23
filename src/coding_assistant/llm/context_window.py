from __future__ import annotations

import httpx

from coding_assistant.llm.openai import _parse_model_and_reasoning, list_models
from coding_assistant.llm.provider_config import ProviderConfig

DEFAULT_MODEL_CONTEXT_WINDOW = 128_000
DEFAULT_COMPACTION_RATIO = 0.8


def get_model_context_window(model: str) -> int | None:
    """Return estimated maximum context window in tokens for known models."""
    model_lower = model.lower()
    if "gemini" in model_lower:
        return 1_000_000
    if any(k in model_lower for k in ("gpt-5", "claude", "o1", "o3")):
        return 200_000
    if any(k in model_lower for k in ("gpt-4", "deepseek")):
        return 128_000
    return None


async def fetch_model_context_window(
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_config: ProviderConfig | None = None,
) -> int | None:
    """Query provider /models (e.g. OpenRouter) for advertised context length, falling back to static lookup."""
    base_model, _ = _parse_model_and_reasoning(model)
    try:
        models = await list_models(transport=transport, provider_config=provider_config)
        for m in models:
            if m.id == base_model and m.context_length is not None:
                return m.context_length
    except Exception:
        pass
    return get_model_context_window(base_model)


async def resolve_auto_compaction_budget(
    model: str,
    *,
    configured_budget: int | None = None,
    ratio: float = DEFAULT_COMPACTION_RATIO,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_config: ProviderConfig | None = None,
) -> int:
    """Return the effective token budget before triggering automatic compaction."""
    if configured_budget is not None:
        return configured_budget
    context_window = (
        await fetch_model_context_window(model, transport=transport, provider_config=provider_config)
        or DEFAULT_MODEL_CONTEXT_WINDOW
    )
    return int(context_window * ratio)
