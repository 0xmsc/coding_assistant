from collections.abc import AsyncIterator, Sequence
import json
from typing import Any, cast

import httpx
import pytest

from coding_assistant.llm import openai as openai_model
from coding_assistant.llm.openai import (
    ProviderModel,
    _extract_usage,
    _get_base_url_and_api_key,
    _merge_chunks,
    _prepare_messages,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    ToolCall,
    Usage,
    UserMessage,
)


class TestMergeChunks:
    """Tests for the _merge_chunks function."""

    def test_merge_chunks_basic_content(self) -> None:
        """Test merging chunks with basic content."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": " world"},
                        "finish_reason": None,
                    },
                ],
            },
        ]

        result = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert result.content == "Hello world"
        assert result.role == "assistant"
        assert result.reasoning_content is None
        assert usage is None

    def test_merge_chunks_with_reasoning(self) -> None:
        """Test merging chunks with reasoning content."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning": "Thinking"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"reasoning": " more thoughts"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Answer"},
                        "finish_reason": None,
                    },
                ],
            },
        ]

        result = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert result.content == "Answer"
        assert result.reasoning_content == "Thinking more thoughts"
        assert usage is None

    def test_merge_chunks_with_tool_calls(self) -> None:
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
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"key"'}}],
                        },
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": ': "value"}'}}],
                        },
                        "finish_reason": None,
                    },
                ],
            },
        ]

        result = _merge_chunks(cast(Any, chunks))

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_"
        assert result.tool_calls[0].function.name == "test"
        assert result.tool_calls[0].function.arguments == '{"key": "value"}'

    def test_merge_chunks_empty(self) -> None:
        """Test merging empty chunk list."""
        result = _merge_chunks([])
        usage = _extract_usage([])

        assert result.content is None
        assert result.role == "assistant"
        assert result.tool_calls == []
        assert usage is None

    def test_merge_chunks_only_role(self) -> None:
        """Test chunks with only role in first delta."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    },
                ],
            },
        ]

        result = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert result.content is None
        assert result.role == "assistant"
        assert usage is None

    def test_merge_chunks_with_usage(self) -> None:
        """Test merging chunks with usage information."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0015,
                },
            },
        ]

        result = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert result.content == "Hello"
        assert usage is not None
        assert cast(Any, usage).tokens == 150
        assert usage is not None
        assert cast(Any, usage).cost == 0.0015

    def test_merge_chunks_usage_overwritten(self) -> None:
        """Test that usage is taken from the last chunk with usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "Part 1"},
                        "finish_reason": None,
                    },
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
                    },
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

        result = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert result.content == "Part 1 Part 2"
        assert usage is not None
        assert cast(Any, usage).tokens == 20
        assert usage is not None
        assert cast(Any, usage).cost == 0.0002

    def test_merge_chunks_reasoning_content_alt(self) -> None:
        """Test alternate field name used by some providers."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Deep"}}]},
            {"choices": [{"delta": {"reasoning_content": " thought"}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert msg.reasoning_content == "Deep thought"
        assert usage is None

    def test_merge_chunks_reasoning_details_openrouter(self) -> None:
        """Test OpenRouter reasoning_details field."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.thought", "text": "step 1"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.thought", "text": "step 2"}]}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert msg.provider_specific_fields["reasoning_details"] == [
            {"type": "reasoning.thought", "text": "step 1"},
            {"type": "reasoning.thought", "text": "step 2"},
        ]
        assert usage is None

    def test_merge_chunks_reasoning_details_merge_text_chunks(self) -> None:
        """Test merging reasoning.text chunks with same index."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Part 1"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Part 2"}]}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        # Chunks with same index should be merged
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "Part 1Part 2"
        assert usage is None

    def test_merge_chunks_reasoning_details_merge_with_signature(self) -> None:
        """Test merging reasoning.text chunks updates signature."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "index": 0, "text": "Step 1", "signature": "sig1"},
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.text", "index": 0, "text": "Step 2", "signature": "sig2"},
                            ],
                        },
                    },
                ],
            },
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        # Signature should be updated to the latest one
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "Step 1Step 2"
        assert msg.provider_specific_fields["reasoning_details"][0]["signature"] == "sig2"
        assert usage is None

    def test_merge_chunks_reasoning_details_different_indices(self) -> None:
        """Test that reasoning.text chunks with different indices are not merged."""
        chunks = [
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "First"}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 1, "text": "Second"}]}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        # Chunks with different indices should remain separate
        assert len(msg.provider_specific_fields["reasoning_details"]) == 2
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "First"
        assert msg.provider_specific_fields["reasoning_details"][1]["text"] == "Second"
        assert usage is None

    def test_merge_chunks_reasoning_details_mixed_types(self) -> None:
        """Test that different reasoning_detail types are not merged."""
        chunks = [
            {
                "choices": [
                    {"delta": {"reasoning_details": [{"type": "reasoning.text", "index": 0, "text": "Text chunk"}]}},
                ],
            },
            {"choices": [{"delta": {"reasoning_details": [{"type": "reasoning.other", "index": 0, "data": "value"}]}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        # Different types should remain separate even with same index
        assert len(msg.provider_specific_fields["reasoning_details"]) == 2
        assert msg.provider_specific_fields["reasoning_details"][0]["type"] == "reasoning.text"
        assert msg.provider_specific_fields["reasoning_details"][1]["type"] == "reasoning.other"
        assert usage is None

    def test_merge_chunks_reasoning_details_summary(self) -> None:
        """Test merging reasoning.summary chunks."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"type": "reasoning.summary", "index": 0, "summary": "Summary part 1"},
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {"delta": {"reasoning_details": [{"type": "reasoning.summary", "index": 0, "summary": " part 2"}]}},
                ],
            },
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        # Chunks with same index should be merged
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["summary"] == "Summary part 1 part 2"
        assert usage is None

    def test_merge_chunks_reasoning_details_empty_list(self) -> None:
        """Test that empty reasoning_details list is handled."""
        chunks: Any = [
            {"choices": [{"delta": {"reasoning_details": []}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert msg.provider_specific_fields == {}
        assert usage is None

    def test_merge_chunks_reasoning_details_initial_chunk_without_text(self) -> None:
        """Test merging chunks where the first chunk only has signature/metadata and no text."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "format": "google-gemini-v1",
                                    "index": 0,
                                    "signature": "sig123",
                                    "type": "reasoning.text",
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"index": 0, "text": "Step 1", "type": "reasoning.text"},
                            ],
                        },
                    },
                ],
            },
        ]
        msg = _merge_chunks(cast(Any, chunks))
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0] == {
            "format": "google-gemini-v1",
            "index": 0,
            "signature": "sig123",
            "text": "Step 1",
            "type": "reasoning.text",
        }
        assert msg.reasoning_content == "Step 1"

    def test_merge_chunks_reasoning_details_missing_index(self) -> None:
        """Test merging chunks when index is omitted."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"text": "Step 1", "type": "reasoning.text"},
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"text": " Step 2", "type": "reasoning.text"},
                            ],
                        },
                    },
                ],
            },
        ]
        msg = _merge_chunks(cast(Any, chunks))
        assert len(msg.provider_specific_fields["reasoning_details"]) == 1
        assert msg.provider_specific_fields["reasoning_details"][0]["text"] == "Step 1 Step 2"
        assert msg.reasoning_content == "Step 1 Step 2"

    def test_merge_chunks_populates_reasoning_content_from_summary(self) -> None:
        """Test that reasoning_content is populated from summary when delta.reasoning is absent."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"index": 0, "summary": "Analyzed constraints", "type": "reasoning.summary"},
                            ],
                        },
                    },
                ],
            },
        ]
        msg = _merge_chunks(cast(Any, chunks))
        assert msg.reasoning_content == "Analyzed constraints"

    def test_merge_chunks_multiple_tool_calls(self) -> None:
        """Test merging multiple tool calls."""
        chunks = [
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f1", "arguments": ""}}]}},
                ],
            },
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "f2", "arguments": ""}}]}},
                ],
            },
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "arg1"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "arg2"}}]}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert len(msg.tool_calls) == 2
        assert msg.tool_calls[0].id == "c1"
        assert msg.tool_calls[0].function.arguments == "arg1"
        assert msg.tool_calls[1].id == "c2"
        assert msg.tool_calls[1].function.arguments == "arg2"
        assert usage is None

    def test_merge_chunks_mixed(self) -> None:
        """Test merging mixed content types."""
        chunks = [
            {"choices": [{"delta": {"content": "I am"}}]},
            {"choices": [{"delta": {"reasoning": "Planning"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "call_456", "function": {"name": "calc", "arguments": ""}},
                            ],
                        },
                    },
                ],
            },
            {"choices": [{"delta": {"content": " searching"}}]},
        ]
        msg = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert msg.content == "I am searching"
        assert msg.reasoning_content == "Planning"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].id == "call_456"
        assert msg.tool_calls[0].function.name == "calc"
        assert usage is None


class TestCompletionType:
    """Tests for the Completion type with Usage."""

    def test_completion_with_usage(self) -> None:
        """Test Completion with usage object."""
        message = AssistantMessage(content="Hello")
        usage = Usage(tokens=100, cost=0.005)
        completion = Completion(message=message, usage=usage)

        assert completion.message == message
        assert cast(Any, completion.usage).tokens == 100
        assert cast(Any, completion.usage).cost == 0.005

    def test_completion_without_usage(self) -> None:
        """Test Completion without usage object."""
        message = AssistantMessage(content="Hello")
        completion = Completion(message=message)

        assert completion.message == message
        assert completion.usage is None

    def test_completion_with_tool_calls(self) -> None:
        """Test Completion with tool calls and usage info."""
        tool_call = ToolCall(id="test", function=FunctionCall(name="test_func", arguments="{}"))
        message = AssistantMessage(content=None, tool_calls=[tool_call])
        usage = Usage(tokens=200, cost=0.01)
        completion = Completion(message=message, usage=usage)

        assert completion.message.tool_calls == [tool_call]
        assert cast(Any, completion.usage).tokens == 200
        assert cast(Any, completion.usage).cost == 0.01

    def test_completion_with_reasoning(self) -> None:
        """Test Completion with reasoning content and usage info."""
        message = AssistantMessage(content="Answer", reasoning_content="Reasoning")
        usage = Usage(tokens=150, cost=0.0075)
        completion = Completion(message=message, usage=usage)

        assert completion.message.reasoning_content == "Reasoning"
        assert cast(Any, completion.usage).tokens == 150
        assert cast(Any, completion.usage).cost == 0.0075


class TestIntegration:
    """Integration tests for usage extraction workflow."""

    def test_full_workflow_with_usage(self) -> None:
        """Test complete workflow from chunks to Completion with usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "The answer"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": " is 42."},
                        "finish_reason": None,
                    },
                ],
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
        message = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))
        assert message.content == "The answer is 42."

        assert usage is not None
        assert cast(Any, usage).tokens == 520
        assert usage is not None
        assert cast(Any, usage).cost == 0.00052

        # Create completion
        completion = Completion(message=message, usage=usage)

        assert completion.message.content == "The answer is 42."
        assert cast(Any, completion.usage).tokens == 520
        assert cast(Any, completion.usage).cost == 0.00052

    def test_workflow_without_cost(self) -> None:
        """Test workflow when provider doesn't return cost."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": "Response"},
                        "finish_reason": None,
                    },
                ],
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

        message = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert message.content == "Response"
        assert usage is not None
        assert usage is not None
        assert cast(Any, usage).tokens == 110
        assert cast(Any, usage).cost is None  # cost is None when not provided

        completion = Completion(message=message, usage=usage)

        assert completion.usage is not None
        assert cast(Any, completion.usage).tokens == 110
        assert cast(Any, completion.usage).cost is None

    def test_workflow_without_token_count(self) -> None:
        """Test workflow when provider doesn't return a token count."""
        chunks = [
            {
                "choices": [{"delta": {}}],
                "usage": {"cost": 0.001},
            },
        ]

        usage = _extract_usage(cast(Any, chunks))

        assert usage == Usage(tokens=None, cost=0.001)

    def test_workflow_with_reasoning_and_usage(self) -> None:
        """Test workflow with reasoning content and usage."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning": "Step 1"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"reasoning": " Step 2"},
                        "finish_reason": None,
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {"content": "Final"},
                        "finish_reason": None,
                    },
                ],
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

        message = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert message.reasoning_content == "Step 1 Step 2"
        assert message.content == "Final"
        assert usage is not None
        assert cast(Any, usage).tokens == 230
        assert usage is not None
        assert cast(Any, usage).cost == 0.0023

        completion = Completion(message=message, usage=usage)

        assert completion.message.reasoning_content == "Step 1 Step 2"
        assert cast(Any, completion.usage).tokens == 230
        assert cast(Any, completion.usage).cost == 0.0023

    def test_workflow_empty_chunks(self) -> None:
        """Test workflow with empty chunk list."""
        chunks: Any = []

        message = _merge_chunks(cast(Any, chunks))
        usage = _extract_usage(cast(Any, chunks))

        assert message.content is None
        assert usage is None

        completion = Completion(message=message, usage=usage)

        assert completion.message.content is None
        assert completion.usage is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_base_url_and_api_key_openai(self) -> None:
        url, key = _get_base_url_and_api_key({"OPENAI_API_KEY": "sk-openai"})
        assert url == "https://api.openai.com/v1"
        assert key == "sk-openai"

    def test_get_base_url_and_api_key_custom(self) -> None:
        url, key = _get_base_url_and_api_key(
            {"OPENAI_BASE_URL": "https://custom.api/v1", "OPENAI_API_KEY": "sk-custom"}
        )
        assert url == "https://custom.api/v1"
        assert key == "sk-custom"

    def test_get_base_url_and_api_key_openrouter(self) -> None:
        url, key = _get_base_url_and_api_key({"OPENROUTER_API_KEY": "sk-openrouter"})
        assert url == "https://openrouter.ai/api/v1"
        assert key == "sk-openrouter"

    def test_get_base_url_and_api_key_custom_with_openrouter_key(self) -> None:
        url, key = _get_base_url_and_api_key(
            {"OPENAI_BASE_URL": "https://custom.api/v1", "OPENROUTER_API_KEY": "sk-openrouter"}
        )
        assert url == "https://custom.api/v1"
        assert key == "sk-openrouter"

    def test_prepare_messages(self) -> None:
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

    @pytest.mark.asyncio
    async def test_list_models(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/models"
            assert request.headers["Authorization"] == "Bearer test-key"
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "z-model", "reasoning": {"supported_efforts": ["high"]}},
                        {"id": "a-model"},
                    ],
                },
            )

        transport = httpx.MockTransport(handler)
        models = await openai_model.list_models(
            transport=transport,
            base_url="https://api.openai.com/v1",
            api_key="test-key",
        )
        assert models == [
            ProviderModel(id="a-model", reasoning_efforts=()),
            ProviderModel(id="z-model", reasoning_efforts=("high",)),
        ]

    @pytest.mark.parametrize(
        ("item", "expected"),
        [
            (
                {"reasoning": {"supported_efforts": ["provider-high", "provider-low"]}},
                ("provider-high", "provider-low"),
            ),
            ({}, ()),
            ({"reasoning": None}, ()),
            ({"reasoning": {}}, ()),
            ({"reasoning": {"supported_efforts": None}}, ()),
            ({"reasoning": {"supported_efforts": "high"}}, ()),
            ({"reasoning": {"supported_efforts": ["high", None]}}, ()),
        ],
    )
    def test_reasoning_efforts_from_model(self, item: dict[str, Any], expected: tuple[str, ...]) -> None:
        assert openai_model._reasoning_efforts_from_model(item) == expected


class _SSEStream(httpx.AsyncByteStream):
    def __init__(self, sse_payloads: list[str]) -> None:
        self._payloads = sse_payloads

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for payload in self._payloads:
            yield f"data: {payload}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"


def _sse_response(chunks: Sequence[Any]) -> httpx.Response:
    payloads = [json.dumps(c) for c in chunks]
    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        stream=_SSEStream(payloads),
    )


async def collect_events(
    *,
    messages: list[UserMessage],
    model: str,
    tools: Any,
    transport: httpx.AsyncBaseTransport,
    retry_delay: float | None = 0.0,
) -> list[Any]:
    return [
        event
        async for event in openai_model.stream_completion(
            messages,
            model=model,
            tools=tools,
            transport=transport,
            base_url="https://api.openai.com/v1",
            api_key="fake_key",
            retry_delay=retry_delay,
        )
    ]


class TestOpenAIComplete:
    """Integration tests for the stream_completion() function using MockTransport."""

    @pytest.mark.asyncio
    async def test_openai_complete_streaming_happy_path(self) -> None:
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [], "usage": {"total_tokens": 3, "cost": 0.001}},
        ]
        transport = httpx.MockTransport(lambda req: _sse_response(chunks))

        msgs = [UserMessage(content="Hello")]
        events = await collect_events(messages=msgs, model="gpt-4o", tools=[], transport=transport)
        assert events == [
            ContentDeltaEvent(content="Hello"),
            ContentDeltaEvent(content=" world"),
            CompletionEvent(
                completion=Completion(
                    message=AssistantMessage(content="Hello world"),
                    usage=Usage(tokens=3, cost=0.001),
                ),
            ),
        ]

    @pytest.mark.asyncio
    async def test_openai_complete_tool_calls(self) -> None:
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "function": {"name": "get_weather", "arguments": '{"location": "New York"}'},
                                },
                            ],
                        },
                    },
                ],
            },
        ]
        transport = httpx.MockTransport(lambda req: _sse_response(chunks))

        msgs = [UserMessage(content="What's the weather in New York")]
        tools: Any = []

        events = await collect_events(messages=msgs, model="gpt-4o", tools=tools, transport=transport)
        assert isinstance(events[-1], CompletionEvent)
        ret = events[-1].completion

        assert ret.message.tool_calls[0].id == "call_123"
        assert ret.message.tool_calls[0].function.name == "get_weather"
        assert ret.message.tool_calls[0].function.arguments == '{"location": "New York"}'

    @pytest.mark.asyncio
    async def test_openai_complete_with_reasoning(self) -> None:
        chunks = [
            {"choices": [{"delta": {"reasoning": "Thinking"}}]},
            {"choices": [{"delta": {"reasoning": " step by step"}}]},
            {"choices": [{"delta": {"content": "Answer"}}]},
        ]
        transport = httpx.MockTransport(lambda req: _sse_response(chunks))

        msgs = [UserMessage(content="Reason")]
        events = await collect_events(messages=msgs, model="o1-preview", tools=[], transport=transport)
        assert events[:-1] == [
            ReasoningDeltaEvent(content="Thinking"),
            ReasoningDeltaEvent(content=" step by step"),
            ContentDeltaEvent(content="Answer"),
        ]
        assert isinstance(events[-1], CompletionEvent)
        ret = events[-1].completion
        assert ret.message.content == "Answer"
        assert ret.message.reasoning_content == "Thinking step by step"

    @pytest.mark.asyncio
    async def test_openai_complete_with_reasoning_details(self) -> None:
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {
                                    "format": "anthropic-claude-v1",
                                    "index": 0,
                                    "signature": "sig1",
                                    "type": "reasoning.text",
                                },
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"index": 0, "text": "Analyzing problem", "type": "reasoning.text"},
                            ],
                        },
                    },
                ],
            },
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_details": [
                                {"index": 0, "text": " step by step", "type": "reasoning.text"},
                            ],
                        },
                    },
                ],
            },
            {"choices": [{"delta": {"content": "Solution"}}]},
        ]
        transport = httpx.MockTransport(lambda req: _sse_response(chunks))

        msgs = [UserMessage(content="Solve")]
        events = await collect_events(messages=msgs, model="claude-3-7-sonnet", tools=[], transport=transport)
        assert events[:-1] == [
            ReasoningDeltaEvent(content="Analyzing problem"),
            ReasoningDeltaEvent(content=" step by step"),
            ContentDeltaEvent(content="Solution"),
        ]
        assert isinstance(events[-1], CompletionEvent)
        ret = events[-1].completion
        assert ret.message.content == "Solution"
        assert ret.message.reasoning_content == "Analyzing problem step by step"
        assert ret.message.provider_specific_fields["reasoning_details"] == [
            {
                "format": "anthropic-claude-v1",
                "index": 0,
                "signature": "sig1",
                "text": "Analyzing problem step by step",
                "type": "reasoning.text",
            },
        ]

    @pytest.mark.asyncio
    async def test_openai_complete_with_reasoning_effort(self) -> None:
        captured_payload: dict[str, Any] | None = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal captured_payload
            captured_payload = json.loads(request.content.decode("utf-8"))
            return _sse_response([{"choices": [{"delta": {"content": "ok"}}]}])

        transport = httpx.MockTransport(handler)

        msgs = [UserMessage(content="Reason")]
        await collect_events(messages=msgs, model="o1 (high)", tools=[], transport=transport)

        assert captured_payload is not None
        assert captured_payload["model"] == "o1"
        assert captured_payload["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_openai_complete_error_retry(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            raise httpx.ReadTimeout("Timeout")

        transport = httpx.MockTransport(handler)

        with pytest.raises(httpx.ReadTimeout):
            await collect_events(
                messages=[UserMessage(content="hi")],
                model="gpt-4o",
                tools=[],
                transport=transport,
                retry_delay=0.0,
            )

        assert call_count == 5

    @pytest.mark.asyncio
    async def test_openai_complete_error_recovery(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ReadTimeout("Timeout")
            return _sse_response([{"choices": [{"delta": {"content": "Recovered"}}]}])

        transport = httpx.MockTransport(handler)

        events = await collect_events(
            messages=[UserMessage(content="hi")],
            model="gpt-4o",
            tools=[],
            transport=transport,
            retry_delay=0.0,
        )
        assert events[0] == ModelRetryEvent()
        assert isinstance(events[-1], CompletionEvent)
        assert events[-1].completion.message.content == "Recovered"
        assert call_count == 2
