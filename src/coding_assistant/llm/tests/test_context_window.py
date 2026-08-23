from __future__ import annotations

import pytest

from coding_assistant.llm.context_window import (
    DEFAULT_COMPACTION_RATIO,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    get_model_context_window,
    resolve_auto_compaction_budget,
)


@pytest.mark.parametrize(
    ("model", "expected_context"),
    [
        ("gemini-2.5-pro", 1_000_000),
        ("gemini-flash", 1_000_000),
        ("gpt-5", 200_000),
        ("gpt-5.5", 200_000),
        ("claude-3-7-sonnet", 200_000),
        ("anthropic/claude-3-opus", 200_000),
        ("o1", 200_000),
        ("o3-mini", 200_000),
        ("gpt-4o", 128_000),
        ("gpt-4-turbo", 128_000),
        ("deepseek-chat", 128_000),
        ("deepseek-reasoner", 128_000),
        ("my-custom-model", None),
        ("llama-3-8b", None),
    ],
)
def test_get_model_context_window(model: str, expected_context: int | None) -> None:
    assert get_model_context_window(model) == expected_context


def test_resolve_auto_compaction_budget_explicit_budget() -> None:
    assert resolve_auto_compaction_budget("gpt-4o", configured_budget=50_000) == 50_000
    assert resolve_auto_compaction_budget("unknown-model", configured_budget=12_345) == 12_345


def test_resolve_auto_compaction_budget_known_models() -> None:
    assert resolve_auto_compaction_budget("gpt-4o") == int(128_000 * DEFAULT_COMPACTION_RATIO)
    assert resolve_auto_compaction_budget("claude-3-7-sonnet") == int(200_000 * DEFAULT_COMPACTION_RATIO)
    assert resolve_auto_compaction_budget("gemini-2.5-pro") == int(1_000_000 * DEFAULT_COMPACTION_RATIO)


def test_resolve_auto_compaction_budget_with_reasoning_annotation() -> None:
    assert resolve_auto_compaction_budget("openai/gpt-5.5 (high)") == int(200_000 * DEFAULT_COMPACTION_RATIO)
    assert resolve_auto_compaction_budget("o3-mini (medium)") == int(200_000 * DEFAULT_COMPACTION_RATIO)


def test_resolve_auto_compaction_budget_unknown_model_fallback() -> None:
    assert resolve_auto_compaction_budget("local-llama") == int(DEFAULT_MODEL_CONTEXT_WINDOW * DEFAULT_COMPACTION_RATIO)


def test_resolve_auto_compaction_budget_custom_ratio() -> None:
    assert resolve_auto_compaction_budget("gpt-4o", ratio=0.5) == 64_000
