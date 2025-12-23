import pytest
from unittest.mock import AsyncMock, MagicMock

from coding_assistant.framework.callbacks import NullProgressCallbacks
from coding_assistant.llm import openai as openai_model
from coding_assistant.llm.types import UserMessage, AssistantMessage


class _CB(NullProgressCallbacks):
    def __init__(self):
        super().__init__()
        self.chunks = []
        self.end = False
        self.reasoning = []

    def on_assistant_reasoning(self, context_name: str, content: str): self.reasoning.append(content)
    def on_content_chunk(self, chunk: str): self.chunks.append(chunk)
    def on_reasoning_chunk(self, chunk: str): self.reasoning.append(chunk)
    def on_chunks_end(self): self.end = True


@pytest.mark.asyncio
async def test_openai_complete_streaming_happy_path(monkeypatch):
    mock_client_instance = MagicMock()
    mock_client_class = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(openai_model, "AsyncOpenAI", mock_client_class)

    async def fake_stream(**kwargs):
        class Chunk:
            def __init__(self, content=None, reasoning=None, tool_calls=None):
                self.choices = [MagicMock()]
                self.choices[0].delta = MagicMock()
                self.choices[0].delta.content = content
                self.choices[0].delta.reasoning_content = reasoning
                self.choices[0].delta.reasoning = None
                self.choices[0].delta.tool_calls = tool_calls

        yield Chunk(content="Hello")
        yield Chunk(content=" world")

    mock_client_instance.chat.completions.create = AsyncMock(side_effect=fake_stream)

    cb = _CB()
    comp = await openai_model.complete(messages=[UserMessage(content="hi")], model="gpt-4", tools=[], callbacks=cb)

    assert cb.chunks == ["Hello", " world"]
    assert cb.end is True
    assert comp.message.content == "Hello world"
    assert isinstance(comp.message, AssistantMessage)


@pytest.mark.asyncio
async def test_openai_complete_with_reasoning(monkeypatch):
    mock_client_instance = MagicMock()
    mock_client_class = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(openai_model, "AsyncOpenAI", mock_client_class)

    async def fake_stream(**kwargs):
        class Chunk:
            def __init__(self, content=None, reasoning=None):
                self.choices = [MagicMock()]
                self.choices[0].delta = MagicMock()
                self.choices[0].delta.content = content
                self.choices[0].delta.reasoning_content = reasoning
                self.choices[0].delta.reasoning = None
                self.choices[0].delta.tool_calls = None

        yield Chunk(reasoning="Thinking...")
        yield Chunk(content="Done")

    mock_client_instance.chat.completions.create = AsyncMock(side_effect=fake_stream)

    cb = _CB()
    comp = await openai_model.complete(messages=[], model="gpt-4", tools=[], callbacks=cb)

    assert cb.reasoning == ["Thinking..."]
    assert cb.chunks == ["Done"]
    assert comp.message.reasoning_content == "Thinking..."
    assert comp.message.content == "Done"


@pytest.mark.asyncio
async def test_openai_complete_tool_calls(monkeypatch):
    mock_client_instance = MagicMock()
    mock_client_class = MagicMock(return_value=mock_client_instance)
    monkeypatch.setattr(openai_model, "AsyncOpenAI", mock_client_class)

    async def fake_stream(**kwargs):
        class ToolCallChunk:
            def __init__(self, index, id=None, name=None, args=None):
                self.index = index
                self.id = id
                self.function = MagicMock()
                self.function.name = name
                self.function.arguments = args

        class Chunk:
            def __init__(self, tool_calls):
                self.choices = [MagicMock()]
                self.choices[0].delta = MagicMock()
                self.choices[0].delta.content = None
                self.choices[0].delta.reasoning_content = None
                self.choices[0].delta.reasoning = None
                self.choices[0].delta.tool_calls = tool_calls

        yield Chunk(tool_calls=[ToolCallChunk(index=0, id="call_1", name="get_weather")])
        yield Chunk(tool_calls=[ToolCallChunk(index=0, args='{"city": "Berlin"}')])

    mock_client_instance.chat.completions.create = AsyncMock(side_effect=fake_stream)

    cb = _CB()
    comp = await openai_model.complete(messages=[], model="gpt-4", tools=[], callbacks=cb)

    assert len(comp.message.tool_calls) == 1
    assert comp.message.tool_calls[0].function.name == "get_weather"
    assert comp.message.tool_calls[0].function.arguments == '{"city": "Berlin"}'
    assert comp.message.tool_calls[0].id == "call_1"
