import json

import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FunctionCall,
    FakeMessage,
    ToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.framework.types import AgentContext, TextResult, Tool, AgentOutput
from coding_assistant.llm.types import message_to_dict
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


class FakeEchoTool(Tool):
    def __init__(self):
        self.called_with = None

    def name(self) -> str:
        return "fake.echo"

    def description(self) -> str:
        return "Echo a provided text"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict) -> TextResult:
        self.called_with = parameters
        return TextResult(content=f"echo: {parameters['text']}")


@pytest.mark.asyncio
async def test_tool_selection_then_finish():
    echo_call = ToolCall("1", FunctionCall("fake.echo", json.dumps({"text": "hi"})))
    finish_call = ToolCall(
        "2",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "done", "summary": "sum"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[echo_call]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    fake_tool = FakeEchoTool()
    agent = make_test_agent(tools=[fake_tool, FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )

    assert state.output is not None
    assert state.output.result == "done"
    assert state.output.summary == "sum"
    assert fake_tool.called_with == {"text": "hi"}

    desc, state = agent
    assert [message_to_dict(m) for m in state.history[1:]] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "fake.echo",
                        "arguments": '{"text": "hi"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: hi",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "done", "summary": "sum"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]


@pytest.mark.asyncio
async def test_unknown_tool_error_then_finish(monkeypatch):
    unknown_call = ToolCall("1", FunctionCall("unknown.tool", "{}"))
    finish_call = ToolCall(
        "2",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "ok", "summary": "s"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[unknown_call]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )

    desc, state = agent
    assert [message_to_dict(m) for m in state.history[1:]] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "unknown.tool",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "unknown.tool",
            "content": "Error executing tool: Tool unknown.tool not found in agent tools.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "ok", "summary": "s"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
    assert state.output is not None
    assert state.output.result == "ok"


@pytest.mark.asyncio
async def test_assistant_message_without_tool_calls_prompts_correction(monkeypatch):
    finish_call = ToolCall(
        "2",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "r", "summary": "s"}),
        ),
    )
    completer = FakeCompleter(
        [
            FakeMessage(content="Hello"),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )

    desc, state = agent
    assert [message_to_dict(m) for m in state.history[1:]] == [
        {
            "role": "assistant",
            "content": "Hello",
        },
        {
            "role": "user",
            "content": "I detected a step from you without any tool calls. This is not allowed. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "r", "summary": "s"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
    assert state.output is not None
    assert state.output.result == "r"


@pytest.mark.asyncio
async def test_errors_if_output_already_set():
    desc, state = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])
    state.output = AgentOutput(result="r", summary="s")
    with pytest.raises(RuntimeError, match="Agent already has a result or summary."):
        await run_agent_loop(
            AgentContext(desc=desc, state=state),
            progress_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            compact_conversation_at_tokens=200_000,
            completer=FakeCompleter([FakeMessage(content="irrelevant")]),
            ui=make_ui_mock(),
        )


@pytest.mark.asyncio
async def test_feedback_ok_does_not_reloop():
    finish_call = ToolCall(
        "1",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "final", "summary": "sum"}),
        ),
    )
    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    agent = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )
    assert state.output is not None
    assert state.output.result == "final"


@pytest.mark.asyncio
async def test_multiple_tool_calls_processed_in_order():
    call1 = ToolCall("1", FunctionCall("fake.echo", json.dumps({"text": "first"})))
    call2 = ToolCall("2", FunctionCall("fake.echo", json.dumps({"text": "second"})))
    finish_call = ToolCall(
        "3",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "ok", "summary": "s"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[call1, call2]),
            FakeMessage(tool_calls=[finish_call]),
        ]
    )

    echo_tool = FakeEchoTool()
    agent = make_test_agent(tools=[echo_tool, FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )
    assert state.output is not None
    assert state.output.result == "ok"
    desc, state = agent
    actual_history = [message_to_dict(m) for m in state.history[1:4]]
    assert actual_history[0] == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "fake.echo",
                    "arguments": '{"text": "first"}',
                },
            },
            {
                "id": "2",
                "type": "function",
                "function": {
                    "name": "fake.echo",
                    "arguments": '{"text": "second"}',
                },
            },
        ],
    }
    # Verify both tool results are present, regardless of order
    tool_results = sorted(actual_history[1:3], key=lambda x: x["tool_call_id"])
    assert tool_results == [
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: first",
        },
        {
            "tool_call_id": "2",
            "role": "tool",
            "name": "fake.echo",
            "content": "echo: second",
        },
    ]

    desc, state = agent
    assert message_to_dict(state.history[4]) == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "3",
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "arguments": '{"result": "ok", "summary": "s"}',
                },
            }
        ],
    }
    desc, state = agent
    assert message_to_dict(state.history[5]) == {
        "tool_call_id": "3",
        "role": "tool",
        "name": "finish_task",
        "content": "Agent output set.",
    }


@pytest.mark.asyncio
async def test_feedback_loop_then_finish():
    finish_call_1 = ToolCall(
        "1",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "first", "summary": "s1"}),
        ),
    )

    finish_call_2 = ToolCall(
        "2",
        FunctionCall(
            "finish_task",
            json.dumps({"result": "second", "summary": "s2"}),
        ),
    )

    completer = FakeCompleter(
        [
            FakeMessage(tool_calls=[finish_call_1]),
            FakeMessage(tool_calls=[finish_call_2]),
        ]
    )

    agent = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])
    desc, state = agent

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )
    assert state.output is not None
    assert state.output.result == "first"
    assert state.output.summary == "s1"

    desc, state = agent
    assert [message_to_dict(m) for m in state.history[1:]] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {
                        "name": "finish_task",
                        "arguments": '{"result": "first", "summary": "s1"}',
                    },
                }
            ],
        },
        {
            "tool_call_id": "1",
            "role": "tool",
            "name": "finish_task",
            "content": "Agent output set.",
        },
    ]
