import json

import pytest
from unittest.mock import Mock

from coding_assistant.agents.callbacks import AgentProgressCallbacks, NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import do_single_step
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeMessage,
    FakeToolCall,
    FakeFunction,
    make_test_agent,
    make_test_context,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentDescription, AgentState, AgentContext, TextResult, Tool
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


class DummyTool(Tool):
    def name(self):
        return "dummy"

    def description(self):
        return ""

    def parameters(self):
        return {}

    async def execute(self, parameters):
        return TextResult(content="ok")


@pytest.mark.asyncio
async def test_do_single_step_adds_shorten_prompt_on_token_threshold():
    # Make the assistant respond with a tool call so the "no tool calls" warning is not added
    tool_call = FakeToolCall(id="call_1", function=FakeFunction(name="dummy", arguments="{}"))
    fake_message = FakeMessage(content=("h" * 2000), tool_calls=[tool_call])
    completer = FakeCompleter([fake_message])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), ShortenConversation()], history=[{"role": "user", "content": "start"}]
    )
    ctx = AgentContext(desc=desc, state=state)

    msg = await do_single_step(
        ctx,
        NullProgressCallbacks(),
        shorten_conversation_at_tokens=1000,
        completer=completer,
        ui=make_ui_mock(),
        tool_callbacks=NullToolCallbacks(),
    )

    assert msg.content == fake_message.content

    expected_history = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": fake_message.content,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "dummy", "arguments": "{}"},
                }
            ],
        },
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "name": "dummy",
            "content": "ok",
        },
        {
            "role": "user",
            "content": (
                "Your conversation history has grown too large. "
                "Please summarize it by using the `shorten_conversation` tool."
            ),
        },
    ]

    assert state.history == expected_history


@pytest.mark.asyncio
async def test_reasoning_is_forwarded_and_not_stored():
    # Prepare a message that includes reasoning_content and a tool call to avoid the no-tool-calls warning
    tool_call = FakeToolCall(id="call_reason", function=FakeFunction(name="dummy", arguments="{}"))
    msg = FakeMessage(content="Hello", tool_calls=[tool_call])
    msg.reasoning_content = "These are my private thoughts"

    completer = FakeCompleter([msg])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), ShortenConversation()],
        history=[{"role": "user", "content": "start"}],
    )
    ctx = AgentContext(desc=desc, state=state)

    callbacks = Mock(spec=AgentProgressCallbacks)

    await do_single_step(
        ctx,
        callbacks,
        shorten_conversation_at_tokens=100_000,
        completer=completer,
        ui=make_ui_mock(),
        tool_callbacks=NullToolCallbacks(),
    )

    # Assert reasoning was forwarded via callback
    callbacks.on_assistant_reasoning.assert_called_once_with(desc.name, "These are my private thoughts")

    # Assert reasoning is not stored in history anywhere
    for entry in state.history:
        assert "reasoning_content" not in entry


# Guard rails for do_single_step


@pytest.mark.asyncio
async def test_requires_finish_tool():
    # Missing finish_task tool should raise
    desc, state = make_test_agent(
        tools=[DummyTool(), ShortenConversation()],
        history=[{"role": "user", "content": "start"}],
    )
    ctx = AgentContext(desc=desc, state=state)
    with pytest.raises(RuntimeError, match="Agent needs to have a `finish_task` tool in order to run a step."):
        await do_single_step(
            ctx,
            NullProgressCallbacks(),
            shorten_conversation_at_tokens=1000,
            completer=FakeCompleter([FakeMessage(content="hi")]),
            ui=make_ui_mock(),
            tool_callbacks=NullToolCallbacks(),
        )


@pytest.mark.asyncio
async def test_requires_shorten_tool():
    # Missing shorten_conversation tool should raise
    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool()],
        history=[{"role": "user", "content": "start"}],
    )
    ctx = AgentContext(desc=desc, state=state)
    with pytest.raises(RuntimeError, match="Agent needs to have a `shorten_conversation` tool in order to run a step."):
        await do_single_step(
            ctx,
            NullProgressCallbacks(),
            shorten_conversation_at_tokens=1000,
            completer=FakeCompleter([FakeMessage(content="hi")]),
            ui=make_ui_mock(),
            tool_callbacks=NullToolCallbacks(),
        )


@pytest.mark.asyncio
async def test_requires_non_empty_history():
    # Empty history should raise
    desc, state = make_test_agent(tools=[DummyTool(), FinishTaskTool(), ShortenConversation()], history=[])
    ctx = AgentContext(desc=desc, state=state)
    with pytest.raises(RuntimeError, match="Agent needs to have history in order to run a step."):
        await do_single_step(
            ctx,
            NullProgressCallbacks(),
            shorten_conversation_at_tokens=1000,
            completer=FakeCompleter([FakeMessage(content="hi")]),
            ui=make_ui_mock(),
            tool_callbacks=NullToolCallbacks(),
        )
