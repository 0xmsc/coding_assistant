import pytest

from coding_assistant.agents.callbacks import AgentProgressCallbacks
from coding_assistant.llm import model as llm_model


class _CB(AgentProgressCallbacks):
    def __init__(self):
        self.chunks = []
        self.end = False
        self.reasoning = []

    def on_agent_start(self, agent_name: str, model: str, is_resuming: bool = False):
        pass

    def on_agent_end(self, agent_name: str, result: str, summary: str):
        pass

    def on_user_message(self, agent_name: str, content: str):
        pass

    def on_assistant_message(self, agent_name: str, content: str):
        pass

    def on_assistant_reasoning(self, agent_name: str, content: str):
        self.reasoning.append(content)

    def on_tool_start(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict):
        pass

    def on_tool_message(self, agent_name: str, tool_call_id: str, tool_name: str, arguments: dict, result: str):
        pass

    def on_chunk(self, chunk: str):
        self.chunks.append(chunk)

    def on_chunks_end(self):
        self.end = True


class _Chunk:
    """Minimal shim to mimic a litellm streaming chunk.

    Behaves like a mapping for item access (e.g., chunk["choices"]) and
    exposes a `_hidden_params` dict so production code can safely mutate it.
    """

    def __init__(self, data: dict):
        self._data = data
        # include created_at to verify it's safe to pop
        self._hidden_params = {"created_at": 0}

    def __getitem__(self, key):
        return self._data[key]


@pytest.mark.asyncio
async def test_complete_streaming_happy_path(monkeypatch):
    # Build a fake async generator that yields chunks with delta.content
    async def fake_acompletion(**kwargs):
        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "Hello"}}]})
            yield _Chunk({"choices": [{"delta": {"content": " world"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        # Simulate final message with model_dump_json available
        class _Msg:
            def __init__(self):
                self.content = "Hello world"

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        return {"choices": [{"message": _Msg()}], "usage": {"total_tokens": 42}}

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    cb = _CB()
    comp = await llm_model.complete(messages=[], model="m", tools=[], callbacks=cb)

    # Chunks were forwarded and end signaled
    assert cb.chunks == ["Hello", " world"]
    assert cb.end is True

    # Completion assembled
    assert comp.tokens == 42
    assert comp.message.content == "Hello world"


@pytest.mark.asyncio
async def test_complete_error_path_logs_and_raises(monkeypatch):
    class Boom(Exception):
        pass

    async def fake_acompletion(**kwargs):
        raise Boom("fail")

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)

    cb = _CB()
    with pytest.raises(Boom):
        await llm_model.complete(messages=[{"role": "user", "content": "x"}], model="m", tools=[], callbacks=cb)


@pytest.mark.asyncio
async def test_complete_parses_reasoning_effort_from_model_string(monkeypatch):
    captured = {}

    # Fake streaming completion that also asserts incoming args
    async def fake_acompletion(**kwargs):
        # capture for assertion outside
        captured.update(kwargs)

        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "A"}}]})
            yield _Chunk({"choices": [{"delta": {"content": "B"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        class _Msg:
            def __init__(self):
                self.content = "AB"

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        return {"choices": [{"message": _Msg()}], "usage": {"total_tokens": 2}}

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    cb = _CB()
    comp = await llm_model.complete(messages=[], model="openai/gpt-5 (low)", tools=[], callbacks=cb)

    # Ensure model and reasoning_effort were parsed and forwarded
    assert captured.get("model") == "openai/gpt-5"
    assert captured.get("reasoning_effort") == "low"

    # And content still streamed
    assert cb.chunks == ["A", "B"]
    assert comp.tokens == 2
    assert comp.message.content == "AB"


@pytest.mark.asyncio
async def test_complete_forwards_image_url_openai_format(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "ok"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        class _Msg:
            def __init__(self):
                self.content = "ok"

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        return {"choices": [{"message": _Msg()}], "usage": {"total_tokens": 1}}

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
            ],
        }
    ]

    cb = _CB()
    _ = await llm_model.complete(messages=messages, model="m", tools=[], callbacks=cb)

    sent = captured.get("messages")
    assert isinstance(sent, list)
    assert sent and isinstance(sent[0], dict)
    parts = sent[0]["content"]
    assert parts[0] == {"type": "text", "text": "What's in this image?"}
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/cat.png", "detail": "high"},
    }


@pytest.mark.asyncio
async def test_complete_forwards_base64_image_openai_format(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "ok"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        class _Msg:
            def __init__(self):
                self.content = "ok"

            def model_dump(self):
                return {"role": "assistant", "content": self.content}

        return {"choices": [{"message": _Msg()}], "usage": {"total_tokens": 1}}

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    base64_payload = "AAAABASE64STRING"

    # Provide content using the OpenAI/LiteLLM standard format with a base64 data URL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_payload}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_payload}"}},
            ],
        }
    ]

    cb = _CB()
    _ = await llm_model.complete(messages=messages, model="m", tools=[], callbacks=cb)

    sent = captured.get("messages")
    parts = sent[0]["content"]

    # ensure we forwarded without modification
    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[0]["image_url"]["url"].endswith(base64_payload)

    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"]["url"].endswith(base64_payload)
