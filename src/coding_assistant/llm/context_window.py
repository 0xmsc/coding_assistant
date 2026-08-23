from __future__ import annotations

import httpx

from coding_assistant.llm.openai import _parse_model_and_reasoning, list_models
from coding_assistant.llm.provider_config import ProviderConfig

DEFAULT_MODEL_CONTEXT_WINDOW = 128_000
DEFAULT_COMPACTION_RATIO = 0.8


async def fetch_model_context_window(
    model: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_config: ProviderConfig | None = None,
) -> int | None:
    """Query provider /models (e.g. OpenRouter) for advertised context length."""
    base_model, _ = _parse_model_and_reasoning(model)
    try:
        models = await list_models(transport=transport, provider_config=provider_config)
        for m in models:
            if m.id == base_model and m.context_length is not None:
                return m.context_length
    except Exception:
        pass
    return None


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
