from __future__ import annotations

import httpx
import pytest

from coding_assistant.llm.context_window import (
    DEFAULT_COMPACTION_RATIO,
    DEFAULT_MODEL_CONTEXT_WINDOW,
    fetch_model_context_window,
    resolve_auto_compaction_budget,
)
from coding_assistant.llm.provider_config import ProviderConfig


@pytest.mark.asyncio
async def test_resolve_auto_compaction_budget_explicit_budget() -> None:
    assert await resolve_auto_compaction_budget("gpt-4o", configured_budget=50_000) == 50_000
    assert await resolve_auto_compaction_budget("unknown-model", configured_budget=12_345) == 12_345


@pytest.mark.asyncio
async def test_resolve_auto_compaction_budget_fallback_default_128k() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    # When context length cannot be determined, default 128k is used for all models
    expected = int(DEFAULT_MODEL_CONTEXT_WINDOW * DEFAULT_COMPACTION_RATIO)
    assert await resolve_auto_compaction_budget("gpt-4o", transport=transport) == expected
    assert await resolve_auto_compaction_budget("claude-3-7-sonnet", transport=transport) == expected
    assert await resolve_auto_compaction_budget("gemini-2.5-pro", transport=transport) == expected
    assert await resolve_auto_compaction_budget("local-llama", transport=transport) == expected


@pytest.mark.asyncio
async def test_resolve_auto_compaction_budget_with_reasoning_annotation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "openai/gpt-5.5", "context_length": 200_000},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    assert await resolve_auto_compaction_budget("openai/gpt-5.5 (high)", transport=transport) == int(
        200_000 * DEFAULT_COMPACTION_RATIO,
    )


@pytest.mark.asyncio
async def test_resolve_auto_compaction_budget_custom_ratio() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    assert await resolve_auto_compaction_budget("unknown-model", ratio=0.5, transport=transport) == int(
        DEFAULT_MODEL_CONTEXT_WINDOW * 0.5,
    )


@pytest.mark.asyncio
async def test_fetch_model_context_window_from_provider_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"id": "meta-llama/llama-3.3-70b-instruct", "context_length": 131072},
                    {"id": "qwen/qwen-2.5-72b-instruct", "context_length": 32768},
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    config = ProviderConfig(base_url="https://openrouter.ai/api/v1", api_key="test-key")

    context = await fetch_model_context_window(
        "meta-llama/llama-3.3-70b-instruct",
        transport=transport,
        provider_config=config,
    )
    assert context == 131072

    budget = await resolve_auto_compaction_budget(
        "meta-llama/llama-3.3-70b-instruct",
        transport=transport,
        provider_config=config,
    )
    assert budget == int(131072 * DEFAULT_COMPACTION_RATIO)


@pytest.mark.asyncio
async def test_fetch_model_context_window_falls_back_to_none_when_api_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "server error"})

    transport = httpx.MockTransport(handler)
    config = ProviderConfig(base_url="https://openrouter.ai/api/v1", api_key="test-key")

    context = await fetch_model_context_window(
        "claude-3-7-sonnet",
        transport=transport,
        provider_config=config,
    )
    assert context is None

    budget = await resolve_auto_compaction_budget(
        "claude-3-7-sonnet",
        transport=transport,
        provider_config=config,
    )
    assert budget == int(DEFAULT_MODEL_CONTEXT_WINDOW * DEFAULT_COMPACTION_RATIO)
