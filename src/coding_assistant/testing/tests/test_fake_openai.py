from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pytest

from coding_assistant.llm.openai import stream_completion
from coding_assistant.llm.types import AssistantMessage, CompletionEvent, ContentDeltaEvent, UserMessage
from coding_assistant.testing.fake_openai import run_fake_openai_server


def test_fake_openai_health_and_streaming_response() -> None:
    with run_fake_openai_server() as server:
        with urlopen(f"{server.base_url.removesuffix('/v1')}/health", timeout=2) as response:
            health = json.loads(response.read().decode("utf-8"))

        request = Request(
            f"{server.base_url}/chat/completions",
            data=json.dumps(
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=2) as response:
            body = response.read().decode("utf-8")

    assert health == {"ok": True}
    assert "data:" in body
    chunks = [
        json.loads(line.removeprefix("data: "))["choices"][0]["delta"].get("content", "")
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    assert "".join(chunks) == "fake response: hello"
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_openai_adapter_streams_against_fake_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    with run_fake_openai_server() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        events = [
            event
            async for event in stream_completion(
                messages=[UserMessage(content="smoke")],
                tools=[],
                model="fake-model",
            )
        ]

    assert "".join(event.content for event in events if isinstance(event, ContentDeltaEvent)) == "fake response: smoke"
    assert isinstance(events[-1], CompletionEvent)
    assert events[-1].completion.message == AssistantMessage(
        content="fake response: smoke",
        provider_specific_fields={"reasoning_details": []},
    )
    assert events[-1].completion.usage is not None
    assert events[-1].completion.usage.cost == 0.0


@pytest.mark.asyncio
async def test_openai_adapter_uses_configured_fake_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_OPENAI_RESPONSE", "configured smoke response")
    with run_fake_openai_server() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        events = [
            event
            async for event in stream_completion(
                messages=[UserMessage(content="ignored")],
                tools=[],
                model="fake-model",
            )
        ]

    assert isinstance(events[-1], CompletionEvent)
    assert events[-1].completion.message.content == "configured smoke response"
