import pytest
from unittest.mock import Mock

from coding_assistant.agents.callbacks import ProgressCallbacks, NullProgressCallbacks, NullToolCallbacks
from coding_assistant.agents.execution import do_single_step, handle_tool_calls, run_agent_loop
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeMessage,
    FakeToolCall,
    FakeFunction,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import AgentContext, TextResult, Tool
from coding_assistant.tools.tools import FinishTaskTool, CompactConversation


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
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[{"role": "user", "content": "start"}]
    )

    msg, tokens = await do_single_step(
        state.history,
        desc.model,
        desc.tools,
        NullProgressCallbacks(),
        completer=completer,
        context_name=desc.name,
    )

    assert msg.content == fake_message.content

    # Append assistant message to history (simulating loop behavior)
    from coding_assistant.agents.history import append_assistant_message

    append_assistant_message(state.history, NullProgressCallbacks(), desc.name, msg)

    # Simulate loop behavior: execute tools and then append shorten prompt due to tokens
    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
    )
    if tokens > 1000:
        state.history.append(
            {
                "role": "user",
                "content": (
                    "Your conversation history has grown too large. "
                    "Please summarize it by using the `compact_conversation` tool."
                ),
            }
        )

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
                "Please summarize it by using the `compact_conversation` tool."
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
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()],
        history=[{"role": "user", "content": "start"}],
    )

    callbacks = Mock(spec=ProgressCallbacks)

    _, _ = await do_single_step(
        state.history,
        desc.model,
        desc.tools,
        callbacks,
        completer=completer,
        context_name=desc.name,
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
        tools=[DummyTool(), CompactConversation()],
        history=[{"role": "user", "content": "start"}],
    )
    ctx = AgentContext(desc=desc, state=state)
    with pytest.raises(RuntimeError, match="Agent needs to have a `finish_task` tool in order to run."):
        await run_agent_loop(
            ctx,
            progress_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=FakeCompleter([FakeMessage(content="hi")]),
            ui=make_ui_mock(),
            compact_conversation_at_tokens=1000,
        )


@pytest.mark.asyncio
async def test_requires_shorten_tool():
    # Missing compact_conversation tool should raise
    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool()],
        history=[{"role": "user", "content": "start"}],
    )
    ctx = AgentContext(desc=desc, state=state)
    with pytest.raises(RuntimeError, match="Agent needs to have a `compact_conversation` tool in order to run."):
        await run_agent_loop(
            ctx,
            progress_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=FakeCompleter([FakeMessage(content="hi")]),
            ui=make_ui_mock(),
            compact_conversation_at_tokens=1000,
        )


@pytest.mark.asyncio
async def test_requires_non_empty_history():
    # Empty history should raise
    desc, state = make_test_agent(tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[])
    with pytest.raises(RuntimeError, match="History is required in order to run a step."):
        await do_single_step(
            state.history,
            desc.model,
            desc.tools,
            NullProgressCallbacks(),
            completer=FakeCompleter([FakeMessage(content="hi")]),
            context_name=desc.name,
        )
