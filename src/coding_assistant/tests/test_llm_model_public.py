import asyncio
import types
import pytest

from coding_assistant.llm import model as llm_model
from coding_assistant.agents.callbacks import AgentCallbacks


class _CB(AgentCallbacks):
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

    def on_tool_message(self, agent_name: str, tool_name: str, arguments: dict, result: str):
        pass

    def on_chunk(self, chunk: str):
        self.chunks.append(chunk)

    def on_chunks_end(self):
        self.end = True


@pytest.mark.asyncio
async def test_complete_streaming_happy_path(monkeypatch):
    # Build a fake async generator that yields chunks with delta.content
    async def fake_acompletion(**kwargs):
        async def agen():
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
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
