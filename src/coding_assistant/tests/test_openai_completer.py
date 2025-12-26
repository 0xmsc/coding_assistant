import json
import pytest
from unittest.mock import MagicMock

import httpx
from coding_assistant.framework.callbacks import NullProgressCallbacks
from coding_assistant.llm import openai as openai_model
from coding_assistant.llm.types import UserMessage, AssistantMessage
from coding_assistant.llm.openai import _merge_chunks, _get_base_url_and_api_key, _prepare_messages


class _CB(NullProgressCallbacks):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.end = False
        self.reasoning = []

    def on_assistant_reasoning(self, context_name: str, content: str):
        self.reasoning.append(content)

    def on_content_chunk(self, chunk: str):
        self.chunks.append(chunk)

    def on_reasoning_chunk(self, chunk: str):
        self.reasoning.append(chunk)

    def on_chunks_end(self):
        self.end = True


class FakeSource:
    def __init__(self, events_data):
        self.events_data = events_data

    async def aiter_sse(self):
        for data in self.events_data:
            event = MagicMock()
            event.data = data
            yield event


class FakeContext:
    def __init__(self, events_data):
        self.source = FakeSource(events_data)

    async def __aenter__(self):
        return self.source

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def test_get_base_url_and_api_key_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    url, key = _get_base_url_and_api_key()
    assert url == "https://api.openai.com/v1"
    assert key == "sk-openai"


def test_get_base_url_and_api_key_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    url, key = _get_base_url_and_api_key()
    assert url == "https://openrouter.ai/api/v1"
    assert key == "sk-openrouter"


def test_get_base_url_and_api_key_custom(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-custom")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    url, key = _get_base_url_and_api_key()
    assert url == "https://custom.api/v1"
    assert key == "sk-custom"


def test_prepare_messages():
    msgs = [
        UserMessage(content="user stuff"),
        AssistantMessage(
            role="assistant",
            content="assistant stuff",
            provider_specific_fields={"reasoning_details": [{"thought": "planned"}]},
        ),
    ]
    prepared = _prepare_messages(msgs)
    assert len(prepared) == 2
    assert prepared[0]["role"] == "user"
    assert prepared[1]["role"] == "assistant"
    assert prepared[1]["reasoning_details"] == [{"thought": "planned"}]
    assert "provider_specific_fields" not in prepared[1]


def test_merge_chunks_content():
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.content == "Hello world"
    assert msg.reasoning_content is None
    assert msg.tool_calls == []


def test_merge_chunks_reasoning():
    chunks = [
        {"choices": [{"delta": {"reasoning": "Thinking"}}]},
        {"choices": [{"delta": {"reasoning": " step by step"}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.content is None
    assert msg.reasoning_content == "Thinking step by step"
    assert msg.tool_calls == []


def test_merge_chunks_reasoning_content_alt():
    # Test alternate field name used by some providers
    chunks = [
        {"choices": [{"delta": {"reasoning_content": "Deep"}}]},
        {"choices": [{"delta": {"reasoning_content": " thought"}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.reasoning_content == "Deep thought"


def test_merge_chunks_reasoning_details_openrouter():
    chunks = [
        {"choices": [{"delta": {"reasoning_details": [{"thought": "step 1"}]}}]},
        {"choices": [{"delta": {"reasoning_details": [{"thought": "step 2"}]}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.provider_specific_fields["reasoning_details"] == [{"thought": "step 1"}, {"thought": "step 2"}]


def test_merge_chunks_tool_calls():
    chunks = [
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "id": "call_123", "function": {"name": "", "arguments": ""}}]}}
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"location": "New York"}'}}]}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.content is None
    assert msg.reasoning_content is None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_123"
    assert msg.tool_calls[0].function.name == "get_weather"
    assert msg.tool_calls[0].function.arguments == '{"location": "New York"}'


def test_merge_chunks_multiple_tool_calls():
    chunks = [
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f1", "arguments": ""}}]}}
            ]
        },
        {
            "choices": [
                {"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "f2", "arguments": ""}}]}}
            ]
        },
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "arg1"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "arg2"}}]}}]},
    ]
    msg = _merge_chunks(chunks)
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0].id == "c1"
    assert msg.tool_calls[0].function.arguments == "arg1"
    assert msg.tool_calls[1].id == "c2"
    assert msg.tool_calls[1].function.arguments == "arg2"


def test_merge_chunks_mixed():
    chunks = [
        {"choices": [{"delta": {"content": "I am"}}]},
        {"choices": [{"delta": {"reasoning": "Planning"}}]},
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_456", "function": {"name": "calc", "arguments": ""}}]
                    }
                }
            ]
        },
        {"choices": [{"delta": {"content": " searching"}}]},
    ]
    msg = _merge_chunks(chunks)
    assert msg.content == "I am searching"
    assert msg.reasoning_content == "Planning"
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_456"
    assert msg.tool_calls[0].function.name == "calc"


def test_merge_chunks_empty():
    chunks = []
    msg = _merge_chunks(chunks)
    assert msg.content is None
    assert msg.reasoning_content is None
    assert msg.tool_calls == []


@pytest.mark.asyncio
async def test_openai_complete_streaming_happy_path(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    fake_events = [
        json.dumps({"choices": [{"delta": {"content": "Hello"}}]}),
        json.dumps({"choices": [{"delta": {"content": " world"}}]}),
    ]
    mock_context_instance = FakeContext(fake_events)
    mock_ac = MagicMock(return_value=mock_context_instance)
    monkeypatch.setattr(openai_model, "aconnect_sse", mock_ac)

    cb = _CB()
    msgs = [UserMessage(content="Hello")]
    ret = await openai_model.complete(msgs, "gpt-4o", [], cb)
    assert ret.message.content == "Hello world"
    assert ret.message.tool_calls == []
    assert cb.chunks == ["Hello", " world"]
    assert cb.end is True


@pytest.mark.asyncio
async def test_openai_complete_tool_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    fake_events = [
        json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "function": {"name": "get_weather", "arguments": '{"location": "New York"}'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    ]
    mock_context_instance = FakeContext(fake_events)
    mock_ac = MagicMock(return_value=mock_context_instance)
    monkeypatch.setattr(openai_model, "aconnect_sse", mock_ac)

    cb = _CB()

    msgs = [UserMessage(content="What's the weather in New York")]
    tools = []

    ret = await openai_model.complete(msgs, "gpt-4o", tools, cb)

    assert ret.message.tool_calls[0].id == "call_123"
    assert ret.message.tool_calls[0].function.name == "get_weather"
    assert ret.message.tool_calls[0].function.arguments == '{"location": "New York"}'


@pytest.mark.asyncio
async def test_openai_complete_with_reasoning(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
    fake_events = [
        json.dumps({"choices": [{"delta": {"reasoning": "Thinking"}}]}),
        json.dumps({"choices": [{"delta": {"reasoning": " step by step"}}]}),
        json.dumps({"choices": [{"delta": {"content": "Answer"}}]}),
    ]
    mock_context_instance = FakeContext(fake_events)
    mock_ac = MagicMock(return_value=mock_context_instance)
    monkeypatch.setattr(openai_model, "aconnect_sse", mock_ac)

    cb = _CB()
    msgs = [UserMessage(content="Reason")]
    ret = await openai_model.complete(msgs, "o1-preview", [], cb)
    assert ret.message.content == "Answer"
    assert ret.message.reasoning_content == "Thinking step by step"
    assert cb.reasoning == ["Thinking", " step by step"]


@pytest.mark.asyncio
async def test_openai_complete_with_reasoning_effort(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    # We want to check if reasoning_effort is passed to the payload.
    # We'll mock the AsyncClient.post or just check what's passed to aconnect_sse
    captured_payload = None

    def mock_aconnect_sse(client, method, url, **kwargs):
        nonlocal captured_payload
        captured_payload = kwargs.get("json")
        return FakeContext([json.dumps({"choices": [{"delta": {"content": "ok"}}]})])

    monkeypatch.setattr(openai_model, "aconnect_sse", mock_aconnect_sse)

    cb = _CB()
    msgs = [UserMessage(content="Reason")]
    # Mock _parse_model_and_reasoning
    monkeypatch.setattr("coding_assistant.llm.openai._parse_model_and_reasoning", lambda m: ("o1", "high"))

    await openai_model.complete(msgs, "o1:high", [], cb)

    assert captured_payload["model"] == "o1"
    assert captured_payload["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_openai_complete_error_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    call_count = 0

    def mock_aconnect_sse(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("Timeout")

    monkeypatch.setattr(openai_model, "aconnect_sse", mock_aconnect_sse)

    # Patch sleep to avoid waiting
    async def mocked_sleep(delay):
        pass

    monkeypatch.setattr("asyncio.sleep", mocked_sleep)

    cb = _CB()
    # Now that we have max_retries = 3, it should call 3 times before failing
    with pytest.raises(httpx.ReadTimeout):
        await openai_model.complete([UserMessage(content="hi")], "gpt-4o", [], cb)

    assert call_count == 3


@pytest.mark.asyncio
async def test_openai_complete_error_recovery(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake_key")

    call_count = 0

    def mock_aconnect_sse(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ReadTimeout("Timeout")
        return FakeContext([json.dumps({"choices": [{"delta": {"content": "Recovered"}}]})])

    monkeypatch.setattr(openai_model, "aconnect_sse", mock_aconnect_sse)

    # Patch sleep to avoid waiting
    async def mocked_sleep(delay):
        pass

    monkeypatch.setattr("asyncio.sleep", mocked_sleep)

    cb = _CB()
    ret = await openai_model.complete([UserMessage(content="hi")], "gpt-4o", [], cb)

    assert ret.message.content == "Recovered"
    assert call_count == 2
