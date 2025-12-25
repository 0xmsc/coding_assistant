import json
import pytest
from unittest.mock import MagicMock

from coding_assistant.framework.callbacks import NullProgressCallbacks
from coding_assistant.llm import openai as openai_model
from coding_assistant.llm.types import UserMessage
from coding_assistant.llm.openai import _merge_chunks


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

    tools = [
        dict(
            name="get_weather",
            description="Get weather",
            fn_sig='{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}',
        )
    ]

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
