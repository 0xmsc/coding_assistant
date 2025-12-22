import pytest
import litellm
from unittest.mock import AsyncMock, patch
from coding_assistant.llm import litellm as llm_model
from coding_assistant.framework.callbacks import ProgressCallbacks


class _CB(ProgressCallbacks):
    def on_user_message(self, context_name, content, force=False):
        pass

    def on_assistant_message(self, context_name, content, force=False):
        pass

    def on_assistant_reasoning(self, context_name, content):
        pass

    def on_tool_start(self, context_name, tool_call_id, tool_name, arguments):
        pass

    def on_tool_message(self, context_name, tool_call_id, tool_name, arguments, result):
        pass

    def on_content_chunk(self, chunk):
        pass

    def on_reasoning_chunk(self, chunk):
        pass

    def on_chunks_end(self):
        pass


class _Chunk:
    def __init__(self, data):
        self._data = data
        self._hidden_params = {"created_at": 0}

    def __getitem__(self, key):
        return self._data[key]


def _make_mock_response(content):
    class _Msg:
        def __init__(self, c):
            self.content = c

        def model_dump(self):
            return {"role": "assistant", "content": self.content}

    class _Response(dict):
        def model_dump(self):
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self["choices"][0]["message"].content,
                        }
                    }
                ],
                "usage": self["usage"],
            }

    res = _Response({"choices": [{"message": _Msg(content)}], "usage": {"total_tokens": 10}})
    return res


@pytest.mark.asyncio
async def test_complete_retries_on_rate_limit(monkeypatch):
    call_count = 0

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise litellm.RateLimitError("Rate limit exceeded", model="m", llm_provider="openai")

        async def agen():
            yield _Chunk({"choices": [{"delta": {"content": "Success"}}]})

        return agen()

    def fake_stream_chunk_builder(chunks):
        return _make_mock_response("Success")

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(llm_model.litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    # Mock sleep to speed up the test
    with patch("asyncio.sleep", AsyncMock()):
        cb = _CB()
        comp = await llm_model.complete(messages=[], model="m", tools=[], callbacks=cb)

    assert call_count == 3
    assert comp.message.content == "Success"


@pytest.mark.asyncio
async def test_complete_raises_after_max_retries(monkeypatch):
    call_count = 0

    async def fake_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        raise litellm.ServiceUnavailableError("Service Down", model="m", llm_provider="openai")

    monkeypatch.setattr(llm_model.litellm, "acompletion", fake_acompletion)

    # Mock sleep to speed up the test
    with patch("asyncio.sleep", AsyncMock()):
        cb = _CB()
        with pytest.raises(litellm.ServiceUnavailableError):
            await llm_model.complete(messages=[], model="m", tools=[], callbacks=cb)

    assert call_count == 3
