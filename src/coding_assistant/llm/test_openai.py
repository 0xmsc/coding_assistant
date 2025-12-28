
from coding_assistant.llm.openai import _merge_chunks, _extract_usage
from coding_assistant.llm.types import Completion, Usage, AssistantMessage, ToolCall, FunctionCall


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

        result, usage = _merge_chunks(chunks)

        assert result.content == "Hello world"
        assert result.role == "assistant"
        assert result.reasoning_content is None
        assert usage.tokens == 0
        assert usage.cost == 0.0

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

        result, usage = _merge_chunks(chunks)

        assert result.content == "Answer"
        assert result.reasoning_content == "Thinking more thoughts"
        assert usage.tokens == 0
        assert usage.cost == 0.0

    def test_merge_chunks_with_tool_calls(self):
        """Test merging chunks with tool calls."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "tool_calls": [
                                {"index": 0, "id": "call_", "function": {"name": "test", "arguments": ""}}
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"key"'}}
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": ': "value"}'}}
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
        ]

        result, usage = _merge_chunks(chunks)

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_"
        assert result.tool_calls[0].function.name == "test"
        assert result.tool_calls[0].function.arguments == '{"key": "value"}'

    def test_merge_chunks_empty(self):
        """Test merging empty chunk list."""
        result, usage = _merge_chunks([])

        assert result.content is None
        assert result.role == "assistant"
        assert result.tool_calls == []
        assert usage.tokens == 0
        assert usage.cost == 0.0

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

        result, usage = _merge_chunks(chunks)

        assert result.content is None
        assert result.role == "assistant"
        assert usage.tokens == 0
        assert usage.cost == 0.0

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

        result, usage = _merge_chunks(chunks)

        assert result.content == "Hello"
        assert usage.tokens == 50
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
                    "completion_tokens": 10,
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
                    "completion_tokens": 20,
                    "cost": 0.0002,
                },
            },
        ]

        result, usage = _merge_chunks(chunks)

        assert result.content == "Part 1 Part 2"
        assert usage.tokens == 20
        assert usage.cost == 0.0002


class TestExtractUsage:
    """Tests for usage extraction from completion chunks."""

    def test_extract_usage_with_all_fields(self):
        """Test extracting usage when all fields are present."""
        chunks = [
            {"choices": [{"delta": {"content": "test"}}]},
            {
                "choices": [{"delta": {"content": " more"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0015,
                },
            },
        ]

        result = _extract_usage(chunks)

        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["cost"] == 0.0015

    def test_extract_usage_missing_usage(self):
        """Test extracting usage when last chunk has no usage field."""
        chunks = [
            {"choices": [{"delta": {"content": "test"}}]},
            {"choices": [{"delta": {"content": " more"}}]},
        ]

        result = _extract_usage(chunks)

        assert result == {}

    def test_extract_usage_empty_chunks(self):
        """Test extracting usage from empty chunk list."""
        result = _extract_usage([])

        assert result == {}

    def test_extract_usage_partial_fields(self):
        """Test extracting usage with only some fields present."""
        chunks = [
            {"choices": [{"delta": {"content": "test"}}]},
            {
                "choices": [{"delta": {"content": " more"}}],
                "usage": {
                    "completion_tokens": 25,
                    "total_tokens": 100,
                },
            },
        ]

        result = _extract_usage(chunks)

        assert result["completion_tokens"] == 25
        assert result["total_tokens"] == 100
        assert result.get("cost") is None
        assert result.get("prompt_tokens") is None


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

    def test_completion_with_empty_usage(self):
        """Test Completion with default usage."""
        message = AssistantMessage(content="Hello")
        usage = Usage()
        completion = Completion(message=message, usage=usage)

        assert completion.message == message
        assert completion.usage.tokens == 0
        assert completion.usage.cost == 0.0

    def test_completion_with_none_cost(self):
        """Test Completion with usage where cost might be None."""
        message = AssistantMessage(content="Hello")
        usage = Usage(tokens=50, cost=None)
        completion = Completion(message=message, usage=usage)

        assert completion.message == message
        assert completion.usage.tokens == 50
        assert completion.usage.cost is None

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
        message, usage = _merge_chunks(chunks)
        assert message.content == "The answer is 42."

        # Extract usage from last chunk
        raw_usage = _extract_usage(chunks)
        assert raw_usage["completion_tokens"] == 20
        assert raw_usage["cost"] == 0.00052

        # Verify the usage from merge matches raw usage
        assert usage.tokens == raw_usage["completion_tokens"]
        assert usage.cost == raw_usage["cost"]

        # Create completion
        completion = Completion(message=message, usage=usage)

        assert completion.message.content == "The answer is 42."
        assert completion.usage.tokens == 20
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

        message, usage = _merge_chunks(chunks)
        raw_usage = _extract_usage(chunks)

        assert message.content == "Response"
        assert usage.tokens == 10
        assert usage.cost == 0.0  # Default when not provided

        completion = Completion(message=message, usage=usage)

        assert completion.usage.tokens == 10
        assert completion.usage.cost == 0.0

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

        message, usage = _merge_chunks(chunks)
        raw_usage = _extract_usage(chunks)

        assert message.reasoning_content == "Step 1 Step 2"
        assert message.content == "Final"
        assert usage.tokens == 30
        assert usage.cost == 0.0023

        completion = Completion(message=message, usage=usage)

        assert completion.message.reasoning_content == "Step 1 Step 2"
        assert completion.usage.tokens == 30
        assert completion.usage.cost == 0.0023

    def test_workflow_empty_chunks(self):
        """Test workflow with empty chunk list."""
        chunks = []

        message, usage = _merge_chunks(chunks)
        raw_usage = _extract_usage(chunks)

        assert message.content is None
        assert usage.tokens == 0
        assert usage.cost == 0.0
        assert raw_usage == {}

        completion = Completion(message=message, usage=usage)

        assert completion.message.content is None
        assert completion.usage.tokens == 0
        assert completion.usage.cost == 0.0
