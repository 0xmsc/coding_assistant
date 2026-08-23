from __future__ import annotations

from coding_assistant.llm.openai import _parse_model_and_reasoning

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


def resolve_auto_compaction_budget(
    model: str,
    *,
    configured_budget: int | None = None,
    ratio: float = DEFAULT_COMPACTION_RATIO,
) -> int:
    """Return the effective token budget before triggering automatic compaction."""
    if configured_budget is not None:
        return configured_budget
    base_model, _ = _parse_model_and_reasoning(model)
    context_window = get_model_context_window(base_model) or DEFAULT_MODEL_CONTEXT_WINDOW
    return int(context_window * ratio)
