import pytest
from unittest.mock import Mock

from coding_assistant.framework.callbacks import ProgressCallbacks, NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.execution import do_single_step, handle_tool_calls
from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from coding_assistant.framework.types import AgentContext, TextResult, Tool
from coding_assistant.framework.history import append_assistant_message
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


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
    tool_call = ToolCall(id="call_1", function=FunctionCall(name="dummy", arguments="{}"))
    fake_message = AssistantMessage(content=("h" * 2000), tool_calls=[tool_call])
    completer = FakeCompleter([fake_message])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()],
        history=[UserMessage(content="start")],
    )

    msg, usage = await do_single_step(
        state.history,
        desc.model,
        desc.tools,
        NullProgressCallbacks(),
        completer=completer,
        context_name=desc.name,
    )

    assert msg.content == fake_message.content

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
    if usage is not None and usage.tokens > 1000:
        state.history.append(
            UserMessage(
                content=(
                    "Your conversation history has grown too large. "
                    "Please summarize it by using the `compact_conversation` tool."
                ),
            )
        )

    expected_history = [
        UserMessage(content="start"),
        AssistantMessage(
            content=fake_message.content,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=FunctionCall(name="dummy", arguments="{}"),
                )
            ],
        ),
        ToolMessage(
            tool_call_id="call_1",
            name="dummy",
            content="ok",
        ),
        UserMessage(
            content=(
                "Your conversation history has grown too large. "
                "Please summarize it by using the `compact_conversation` tool."
            ),
        ),
    ]

    assert state.history == expected_history


@pytest.mark.asyncio
async def test_reasoning_is_forwarded_and_not_stored():
    # Prepare a message that includes reasoning_content and a tool call to avoid the no-tool-calls warning
    tool_call = ToolCall(id="call_reason", function=FunctionCall(name="dummy", arguments="{}"))
    msg = AssistantMessage(
        content="Hello",
        tool_calls=[tool_call],
        reasoning_content="These are my private thoughts",
    )

    completer = FakeCompleter([msg])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()],
        history=[UserMessage(content="start")],
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
        assert getattr(entry, "reasoning_content", None) is None


# Guard rails for do_single_step


@pytest.mark.asyncio
async def test_auto_inject_builtin_tools():
    # Tools are empty initially
    desc, state = make_test_agent(
        tools=[],
        history=[UserMessage(content="start")],
    )
    ctx = AgentContext(desc=desc, state=state)

    # We need a completer that will eventually allow the loop to terminate
    # First message: no tool calls -> warning
    # Second message: finish_task -> stop
    completer = FakeCompleter(
        [
            AssistantMessage(content="no tools yet"),
            AssistantMessage(
                content="done",
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(name="finish_task", arguments='{"result": "ok", "summary": "done"}'),
                    )
                ],
            ),
        ]
    )

    await run_agent_loop(
        ctx,
        progress_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=make_ui_mock(),
        compact_conversation_at_tokens=1000,
    )

    assert state.output is not None
    assert state.output.result == "ok"


@pytest.mark.asyncio
async def test_requires_non_empty_history():
    desc, state = make_test_agent(tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[])
    with pytest.raises(RuntimeError, match="History is required in order to run a step."):
        await do_single_step(
            state.history,
            desc.model,
            desc.tools,
            NullProgressCallbacks(),
            completer=FakeCompleter([AssistantMessage(content="hi")]),
            context_name=desc.name,
        )
