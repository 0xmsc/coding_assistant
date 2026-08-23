from __future__ import annotations

from dataclasses import dataclass

import httpx

from coding_assistant.llm.openai import _parse_model_and_reasoning, list_models
from coding_assistant.llm.provider_config import ProviderConfig

DEFAULT_MODEL_CONTEXT_WINDOW = 128_000
DEFAULT_COMPACTION_RATIO = 0.8


@dataclass(frozen=True)
class ModelLimits:
    context_window: int | None
    compaction_budget: int


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


async def resolve_model_limits(
    model: str,
    *,
    configured_budget: int | None = None,
    ratio: float = DEFAULT_COMPACTION_RATIO,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_config: ProviderConfig | None = None,
) -> ModelLimits:
    """Resolve the advertised context window and effective compaction budget."""
    context_window = await fetch_model_context_window(
        model,
        transport=transport,
        provider_config=provider_config,
    )
    maximum_compaction_budget = int((context_window or DEFAULT_MODEL_CONTEXT_WINDOW) * ratio)
    if configured_budget is None:
        compaction_budget = maximum_compaction_budget
    elif context_window is None:
        compaction_budget = configured_budget
    else:
        compaction_budget = min(configured_budget, maximum_compaction_budget)
    return ModelLimits(context_window=context_window, compaction_budget=compaction_budget)


async def resolve_auto_compaction_budget(
    model: str,
    *,
    configured_budget: int | None = None,
    ratio: float = DEFAULT_COMPACTION_RATIO,
    transport: httpx.AsyncBaseTransport | None = None,
    provider_config: ProviderConfig | None = None,
) -> int:
    """Return the effective token budget before triggering automatic compaction."""
    limits = await resolve_model_limits(
        model,
        configured_budget=configured_budget,
        ratio=ratio,
        transport=transport,
        provider_config=provider_config,
    )
    return limits.compaction_budget
