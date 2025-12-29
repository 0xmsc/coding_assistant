import json
import pytest
from unittest.mock import MagicMock

import httpx
from coding_assistant.framework.callbacks import NullProgressCallbacks
from coding_assistant.llm import openai as openai_model
from coding_assistant.llm.types import (
    Completion,
    Usage,
    AssistantMessage,
    ToolCall,
    FunctionCall,
    UserMessage,
)
from coding_assistant.llm.openai import (
    _merge_chunks,
    _extract_usage,
    _get_base_url_and_api_key,
    _prepare_messages,
)


class _CB(NullProgressCallbacks):
    """Test callback tracker."""

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


class TestMergeChunks:
    """Tests for the _merge_chunks function."""

    def test_merge_chunks_basic_content(self):
        """Test merging chunks with basic content."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": " world"},
                        "finish_reason": None,
                    }
                ]
            },
        ]

        result = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert result.content == "Hello world"
        assert result.role == "assistant"
        assert result.reasoning_content is None
        assert usage is None

    def test_merge_chunks_with_reasoning(self):
        """Test merging chunks with reasoning content."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning": "Thinking"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"reasoning": " more thoughts"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Answer"},
                        "finish_reason": None,
                    }
                ]
            },
        ]

        result = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert result.content == "Answer"
        assert result.reasoning_content == "Thinking more thoughts"
        assert usage is None

    def test_merge_chunks_with_tool_calls(self):
        """Test merging chunks with tool calls."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [{"index": 0, "id": "call_", "function": {"name": "test", "arguments": ""}}],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"key"'}}],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ': "value"}'}}],
                        },
                        "finish_reason": None,
                    }
                ]
            },
        ]

        result = _merge_chunks(chunks)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_"
        assert result.tool_calls[0].function.name == "test"
        assert result.tool_calls[0].function.arguments == '{"key": "value"}'

    def test_merge_chunks_empty(self):
        """Test merging empty chunk list."""
        result = _merge_chunks([])
        usage = _extract_usage([])

        assert result.content is None
        assert result.role == "assistant"
        assert result.tool_calls == []
        assert usage is None

    def test_merge_chunks_only_role(self):
        """Test chunks with only role in first delta."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ]
            },
        ]

        result = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert result.content is None
        assert result.role == "assistant"
        assert usage is None

    def test_merge_chunks_with_usage(self):
        """Test merging chunks with usage information."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0015,
                },
            },
        ]

        result = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert result.content == "Hello"
        assert usage.tokens == 150
        assert usage.cost == 0.0015

    def test_merge_chunks_usage_overwritten(self):
        """Test that usage is taken from the last chunk with usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "Part 1"},
                        "finish_reason": None,
                    }
                ],
                "usage": {
                    "total_tokens": 10,
                    "cost": 0.0001,
                },
            },
            {
                "choices": [
                    {
                        "delta": {"content": " Part 2"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "total_tokens": 20,
                    "cost": 0.0002,
                },
            },
        ]

        result = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert result.content == "Part 1 Part 2"
        assert usage.tokens == 20
        assert usage.cost == 0.0002

    def test_merge_chunks_reasoning_content_alt(self):
        """Test alternate field name used by some providers."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Deep"}}]},
            {"choices": [{"delta": {"reasoning_content": " thought"}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        assert msg.reasoning_content == "Deep thought"
        assert usage is None

    def test_merge_chunks_reasoning_details_openrouter(self):
        """Test OpenRouter reasoning_details field."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.thought", "text": "step 1"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.thought", "text": "step 2"}]}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        assert msg.provider_specific_fields["reasoning_details"] == [
            {"type": "reasoning.thought", "text": "step 1"},
            {"type": "reasoning.thought", "text": "step 2"},
        ]
        assert usage is None

    def test_merge_chunks_reasoning_details_merge_text_chunks(self):
        """Test merging reasoning.text chunks with same index."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Part 1"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Part 2"}]}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        # Chunks with same index should be merged
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "Part 1Part 2"
        assert usage is None

    def test_merge_chunks_reasoning_details_merge_with_signature(self):
        """Test merging reasoning.text chunks updates signature."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "index": 0, "text": "Step 1", "signature": "sig1"}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "index": 0, "text": "Step 2", "signature": "sig2"}
                            ]
                        }
                    }
                ]
            },
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        # Signature should be updated to the latest one
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "Step 1Step 2"
        assert msg.provider_specific_fields["reasoning_details"][0]["signature"] == "sig2"
        assert usage is None

    def test_merge_chunks_reasoning_details_different_indices(self):
        """Test that reasoning.text chunks with different indices are not merged."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "First"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 1, "text": "Second"}]}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        # Chunks with different indices should remain separate
        assert len(msg.provider_specific_fields["reasoning_details"]) == 2
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "First"
        assert msg.provider_specific_fields["reasoning_details"][1]["text"] == "Second"
        assert usage is None

    def test_merge_chunks_reasoning_details_mixed_types(self):
        """Test that different reasoning_detail types are not merged."""
        chunks = [
            {
                "choices": [
                    {"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Text chunk"}]}}
                ]
            },
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.other", "index": 0, "data": "value"}]}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        # Different types should remain separate even with same index
        assert len(msg.provider_specific_fields["reasoning_details"]) == 2
        assert msg.provider_specific_fields["reasoning_details"][0]["type"] == "reasoning.text"
        assert msg.provider_specific_fields["reasoning_details"][1]["type"] == "reasoning.other"
        assert usage is None

    def test_merge_chunks_reasoning_details_summary(self):
        """Test merging reasoning.summary chunks."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.summary", "index": 0, "summary": "Summary part 1"}
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"delta": {"reasoning_details": [{"type": "reasoning.summary", "index": 0, "summary": " part 2"}]}}
                ]
            },
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        # Chunks with same index should be merged
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["summary"] == "Summary part 1 part 2"
        assert usage is None

    def test_merge_chunks_reasoning_details_empty_list(self):
        """Test that empty reasoning_details list is handled."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": []}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        assert msg.provider_specific_fields["reasoning_details"] == []
        assert usage is None

    def test_merge_chunks_multiple_tool_calls(self):
        """Test merging multiple tool calls."""
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
        usage = _extract_usage(chunks)
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].id == "c1"
        assert msg.tool_calls[0].function.arguments == "arg1"
        assert msg.tool_calls[1].id == "c2"
        assert msg.tool_calls[1].function.arguments == "arg2"
        assert usage is None

    def test_merge_chunks_mixed(self):
        """Test merging mixed content types."""
        chunks = [
            {"choices": [{"delta": {"content": "I am"}}]},
            {"choices": [{"delta": {"reasoning": "Planning"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_456", "function": {"name": "calc", "arguments": ""}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": " searching"}}]},
        ]
        msg = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        assert msg.content == "I am searching"
        assert msg.reasoning_content == "Planning"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_456"
        assert msg.tool_calls[0].function.name == "calc"
        assert usage is None


class TestCompletionType:
    """Tests for the Completion type with Usage."""

    def test_completion_with_usage(self):
        """Test Completion with usage object."""
        message = AssistantMessage(content="Hello")
        usage = Usage(tokens=100, cost=0.005)
        completion = Completion(message=message, usage=usage)

        assert completion.message == message
        assert completion.usage.tokens == 100
        assert completion.usage.cost == 0.005

    def test_completion_without_usage(self):
        """Test Completion without usage object."""
        message = AssistantMessage(content="Hello")
        completion = Completion(message=message)

        assert completion.message == message
        assert completion.usage is None

    def test_completion_with_tool_calls(self):
        """Test Completion with tool calls and usage info."""
        tool_call = ToolCall(id="test", function=FunctionCall(name="test_func", arguments="{}"))
        message = AssistantMessage(content=None, tool_calls=[tool_call])
        usage = Usage(tokens=200, cost=0.01)
        completion = Completion(message=message, usage=usage)

        assert completion.message.tool_calls == [tool_call]
        assert completion.usage.tokens == 200
        assert completion.usage.cost == 0.01

    def test_completion_with_reasoning(self):
        """Test Completion with reasoning content and usage info."""
        message = AssistantMessage(content="Answer", reasoning_content="Reasoning")
        usage = Usage(tokens=150, cost=0.0075)
        completion = Completion(message=message, usage=usage)

        assert completion.message.reasoning_content == "Reasoning"
        assert completion.usage.tokens == 150
        assert completion.usage.cost == 0.0075


class TestIntegration:
    """Integration tests for usage extraction workflow."""

    def test_full_workflow_with_usage(self):
        """Test complete workflow from chunks to Completion with usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "The answer"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": " is 42."},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 20,
                    "total_tokens": 520,
                    "cost": 0.00052,
                },
            },
        ]

        # Merge chunks into message and usage
        message = _merge_chunks(chunks)
        usage = _extract_usage(chunks)
        assert message.content == "The answer is 42."

        assert usage.tokens == 520
        assert usage.cost == 0.00052

        # Create completion
        completion = Completion(message=message, usage=usage)

        assert completion.message.content == "The answer is 42."
        assert completion.usage.tokens == 520
        assert completion.usage.cost == 0.00052

    def test_workflow_without_cost(self):
        """Test workflow when provider doesn't return cost."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Response"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                },
            },
        ]

        message = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert message.content == "Response"
        assert usage is not None
        assert usage.tokens == 110
        assert usage.cost is None  # cost is None when not provided

        completion = Completion(message=message, usage=usage)

        assert completion.usage is not None
        assert completion.usage.tokens == 110
        assert completion.usage.cost is None

    def test_workflow_with_reasoning_and_usage(self):
        """Test workflow with reasoning content and usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning": "Step 1"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"reasoning": " Step 2"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Final"},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 30,
                    "total_tokens": 230,
                    "cost": 0.0023,
                    "completion_tokens_details": {
                        "reasoning_tokens": 10,
                        "image_tokens": 0,
                    },
                },
            },
        ]

        message = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert message.reasoning_content == "Step 1 Step 2"
        assert message.content == "Final"
        assert usage.tokens == 230
        assert usage.cost == 0.0023

        completion = Completion(message=message, usage=usage)

        assert completion.message.reasoning_content == "Step 1 Step 2"
        assert completion.usage.tokens == 230
        assert completion.usage.cost == 0.0023

    def test_workflow_empty_chunks(self):
        """Test workflow with empty chunk list."""
        chunks = []

        message = _merge_chunks(chunks)
        usage = _extract_usage(chunks)

        assert message.content is None
        assert usage is None

        completion = Completion(message=message, usage=usage)

        assert completion.message.content is None
        assert completion.usage is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_base_url_and_api_key_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        url, key = _get_base_url_and_api_key()
        assert url == "https://api.openai.com/v1"
        assert key == "sk-openai"

    def test_get_base_url_and_api_key_openrouter(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        url, key = _get_base_url_and_api_key()
        assert url == "https://openrouter.ai/api/v1"
        assert key == "sk-openrouter"

    def test_get_base_url_and_api_key_custom(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "https://custom.api/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-custom")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        url, key = _get_base_url_and_api_key()
        assert url == "https://custom.api/v1"
        assert key == "sk-custom"

    def test_prepare_messages(self):
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


class TestOpenAIComplete:
    """Integration tests for the complete() function."""

    @pytest.mark.asyncio
    async def test_openai_complete_streaming_happy_path(self, monkeypatch):
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
    async def test_openai_complete_tool_calls(self, monkeypatch):
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
    async def test_openai_complete_with_reasoning(self, monkeypatch):
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
    async def test_openai_complete_with_reasoning_effort(self, monkeypatch):
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
    async def test_openai_complete_error_retry(self, monkeypatch):
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
        # Now that we have max_retries = 5, it should call 5 times before failing
        with pytest.raises(httpx.ReadTimeout):
            await openai_model.complete([UserMessage(content="hi")], "gpt-4o", [], cb)

        assert call_count == 5

    @pytest.mark.asyncio
    async def test_openai_complete_error_recovery(self, monkeypatch):
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
