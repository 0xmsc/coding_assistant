import json

import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
    no_feedback,
)
from coding_assistant.agents.types import Agent, TextResult, Tool
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


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
    echo_call = FakeToolCall("1", FakeFunction("fake.echo", json.dumps({"text": "hi"})))
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
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
    agent = make_test_agent(tools=[fake_tool, FinishTaskTool(), ShortenConversation()])

    output = await run_agent_loop(
        agent,
        NullCallbacks(),
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        completer=completer,
        ui=make_ui_mock(),
    )

    assert output.result == "done"
    assert output.summary == "sum"
    assert fake_tool.called_with == {"text": "hi"}

    assert agent.history[1:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
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
    unknown_call = FakeToolCall("1", FakeFunction("unknown.tool", "{}"))
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
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

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])

    output = await run_agent_loop(
        agent,
        NullCallbacks(),
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        completer=completer,
        ui=make_ui_mock(),
    )

    assert agent.history[1:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
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
    assert output.result == "ok"


@pytest.mark.asyncio
async def test_assistant_message_without_tool_calls_prompts_correction(monkeypatch):
    finish_call = FakeToolCall(
        "2",
        FakeFunction(
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

    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])

    output = await run_agent_loop(
        agent,
        NullCallbacks(),
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        completer=completer,
        ui=make_ui_mock(),
    )

    assert agent.history[1:] == [
        {
            "role": "assistant",
            "content": "Hello",
        },
        {
            "role": "user",
            "content": "I detected a step from you without any tool calls. This is not allowed. If you want to ask the client something, please use the `ask_user` tool. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "2",
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
    assert output.result == "r"
