import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks
from coding_assistant.llm import litellm as llm_model
from coding_assistant.llm.types import UserMessage


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


class _Chunk:
    """Minimal shim to mimic a litellm streaming chunk.

    Behaves like a mapping for item access (e.g., chunk["choices"]) and
    exposes a `_hidden_params` dict so production code can safely mutate it.
    """

    def __init__(self, data: dict):
        self._data = data
        self._hidden_params = {"created_at": 0}

    def __getitem__(self, key):
        return self._data[key]


def _make_mock_response(data: dict):
    class _Msg:
        def __init__(self, msg_data):
            self.content = msg_data.get("content")

        def model_dump(self):
            return {"role": "assistant", "content": self.content}

    class _Response(dict):
        def model_dump(self):
            def _dump(o):
                if o is self:
                    return {k: _dump(v) for k, v in self.items()}
                if hasattr(o, "model_dump"):
                    return o.model_dump()
                if isinstance(o, list):
                    return [_dump(i) for i in o]
                if isinstance(o, dict):
                    return {k: _dump(v) for k, v in o.items()}
                return o

            return _dump(self)

    if "choices" in data:
        for choice in data["choices"]:
            if "message" in choice and isinstance(choice["message"], dict):
                choice["message"] = _Msg(choice["message"])

    return _Response(data)


@pytest.mark.asyncio
async def test_complete_streaming_happy_path(monkeypatch):
    async def fake_acompletion(**kwargs):
        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "Hello"}}]})
            yield _Chunk({"choices": [{"delta": {"content": " world"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        return _make_mock_response(
            {"choices": [{"message": {"content": "Hello world"}}], "usage": {"total_tokens": 42}}
        )

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    cb = _CB()
    comp = await llm_model.complete(messages=[], model="m", tools=[], callbacks=cb)

    assert cb.chunks == ["Hello", " world"]
    assert cb.end is True

    assert comp.tokens == 42
    assert comp.message.content == "Hello world"


@pytest.mark.asyncio
async def test_complete_streaming_with_reasoning(monkeypatch):
    async def fake_acompletion(**kwargs):
        async def agen():
            yield _Chunk({"choices": [{"delta": {"reasoning": "Thinking..."}}]})
            yield _Chunk({"choices": [{"delta": {"content": "Hello"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        return _make_mock_response({"choices": [{"message": {"content": "Hello"}}], "usage": {"total_tokens": 42}})

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    cb = _CB()
    comp = await llm_model.complete(messages=[], model="m", tools=[], callbacks=cb)

    assert cb.chunks == ["Hello"]
    assert cb.reasoning == ["Thinking..."]
    assert cb.end is True

    assert comp.tokens == 42
    assert comp.message.content == "Hello"


@pytest.mark.asyncio
async def test_complete_error_path_logs_and_raises(monkeypatch):
    class Boom(Exception):
        pass

    async def fake_acompletion(**kwargs):
        raise Boom("fail")

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)

    cb = _CB()
    with pytest.raises(Boom):
        await llm_model.complete(messages=[UserMessage(content="x")], model="m", tools=[], callbacks=cb)


@pytest.mark.asyncio
async def test_complete_parses_reasoning_effort_from_model_string(monkeypatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)

        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "A"}}]})
            yield _Chunk({"choices": [{"delta": {"content": "B"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        return _make_mock_response({"choices": [{"message": {"content": "AB"}}], "usage": {"total_tokens": 2}})

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    cb = _CB()
    comp = await llm_model.complete(messages=[], model="openai/gpt-5 (low)", tools=[], callbacks=cb)

    assert captured.get("model") == "openai/gpt-5"
    assert captured.get("reasoning_effort") == "low"

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
        return _make_mock_response({"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}})

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    messages = [
        UserMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "high"}},
            ]
        )
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
        return _make_mock_response({"choices": [{"message": {"content": "ok"}}], "usage": {"total_tokens": 1}})

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    base64_payload = "AAAABASE64STRING"

    messages = [
        UserMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_payload}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_payload}"}},
            ]
        )
    ]

    cb = _CB()
    _ = await llm_model.complete(messages=messages, model="m", tools=[], callbacks=cb)

    sent = captured.get("messages")
    parts = sent[0]["content"]

    assert parts[0]["type"] == "image_url"
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[0]["image_url"]["url"].endswith(base64_payload)

    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert parts[1]["image_url"]["url"].endswith(base64_payload)
