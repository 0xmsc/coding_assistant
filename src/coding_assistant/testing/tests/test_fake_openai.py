from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pytest

from coding_assistant.llm.openai import stream_completion
from coding_assistant.llm.types import AssistantMessage, CompletionEvent, ContentDeltaEvent, ToolMessage, UserMessage
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
    )
    assert events[-1].completion.usage is not None
    assert events[-1].completion.usage.cost == 0.0


@pytest.mark.asyncio
async def test_openai_adapter_uses_configured_fake_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODING_ASSISTANT_FAKE_OPENAI_RESPONSE", "configured smoke response")
    with run_fake_openai_server() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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


@pytest.mark.asyncio
async def test_openai_adapter_streams_fake_tool_call_and_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CODING_ASSISTANT_FAKE_OPENAI_RESPONSES_JSON",
        json.dumps(
            [
                {
                    "tool_calls": [
                        {
                            "id": "call_shell",
                            "name": "shell_execute",
                            "arguments": {"command": "cat smoke.txt"},
                        },
                    ],
                },
                {"content": "configured final answer"},
            ],
        ),
    )
    with run_fake_openai_server() as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        tool_call_events = [
            event
            async for event in stream_completion(
                messages=[UserMessage(content="read smoke.txt")],
                tools=[],
                model="fake-model",
            )
        ]
        tool_call_completion = tool_call_events[-1]
        assert isinstance(tool_call_completion, CompletionEvent)
        tool_call = tool_call_completion.completion.message.tool_calls[0]

        final_events = [
            event
            async for event in stream_completion(
                messages=[
                    UserMessage(content="read smoke.txt"),
                    tool_call_completion.completion.message,
                    ToolMessage(content="file contents", tool_call_id=tool_call.id),
                ],
                tools=[],
                model="fake-model",
            )
        ]

    assert tool_call.function.name == "shell_execute"
    assert tool_call.function.arguments == '{"command": "cat smoke.txt"}'
    assert isinstance(final_events[-1], CompletionEvent)
    assert final_events[-1].completion.message.content == "configured final answer"
