from coding_assistant.llm.openai import _merge_chunks
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

        result, usage = _merge_chunks(chunks)

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

        result, usage = _merge_chunks(chunks)

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

        result, usage = _merge_chunks(chunks)

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

        result, usage = _merge_chunks(chunks)

        assert result.content == "Part 1 Part 2"
        assert usage.tokens == 20
        assert usage.cost == 0.0002


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
        message, usage = _merge_chunks(chunks)
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

        message, usage = _merge_chunks(chunks)

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

        message, usage = _merge_chunks(chunks)

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

        message, usage = _merge_chunks(chunks)

        assert message.content is None
        assert usage is None

        completion = Completion(message=message, usage=usage)

        assert completion.message.content is None
        assert completion.usage is None
