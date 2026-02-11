from typing import Any
import pytest
from unittest.mock import Mock

from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    agent_actor_scope,
    append_tool_call_results_to_history,
    execute_tool_calls_via_messages,
    make_test_agent,
    make_ui_mock,
    system_actor_scope_for_tests,
    tool_call_actor_scope,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    FunctionCall,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from coding_assistant.framework.types import AgentContext
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.history import append_assistant_message
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


class DummyTool(Tool):
    def name(self) -> str:
        return "dummy"

    def description(self) -> str:
        return ""

    def parameters(self) -> dict[str, Any]:
        return {}

    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        return TextResult(content="ok")


@pytest.mark.asyncio
async def test_do_single_step_adds_shorten_prompt_on_token_threshold() -> None:
    # Make the assistant respond with a tool call so the "no tool calls" warning is not added
    tool_call = ToolCall(id="call_1", function=FunctionCall(name="dummy", arguments="{}"))
    fake_message = AssistantMessage(content=("h" * 2000), tool_calls=[tool_call])
    completer = FakeCompleter([fake_message])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[UserMessage(content="start")]
    )

    async with agent_actor_scope(context_name=desc.name) as agent_actor:
        msg, usage = await agent_actor.do_single_step(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            completer=completer,
            context_name=desc.name,
            progress_callbacks=NullProgressCallbacks(),
        )

    assert msg.content == fake_message.content

    append_assistant_message(state.history, context_name=desc.name, message=msg)

    # Simulate loop behavior: execute tools and then append shorten prompt due to tokens
    async with tool_call_actor_scope(tools=desc.tools, ui=make_ui_mock(), context_name=desc.name) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=state.history,
            execution_results=response.results,
            context_name=desc.name,
            progress_callbacks=NullProgressCallbacks(),
        )
    if usage is not None and usage.tokens > 1000:
        state.history.append(
            UserMessage(
                content=(
                    "Your conversation history has grown too large. "
                    "Please summarize it by using the `compact_conversation` tool."
                )
            )
        )

    expected_history = [
        UserMessage(content="start"),
        AssistantMessage(
            content=fake_message.content,
            tool_calls=[ToolCall(id="call_1", function=FunctionCall(name="dummy", arguments="{}"))],
        ),
        ToolMessage(tool_call_id="call_1", name="dummy", content="ok"),
        UserMessage(
            content=(
                "Your conversation history has grown too large. "
                "Please summarize it by using the `compact_conversation` tool."
            )
        ),
    ]

    assert state.history == expected_history


@pytest.mark.asyncio
async def test_reasoning_is_forwarded_and_not_stored() -> None:
    # Prepare a message that includes reasoning_content and a tool call to avoid the no-tool-calls warning
    tool_call = ToolCall(id="call_reason", function=FunctionCall(name="dummy", arguments="{}"))
    msg = AssistantMessage(content="Hello", tool_calls=[tool_call], reasoning_content="These are my private thoughts")

    completer = FakeCompleter([msg])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[UserMessage(content="start")]
    )

    callbacks = Mock(spec=ProgressCallbacks)

    async with agent_actor_scope(context_name=desc.name) as agent_actor:
        _, _ = await agent_actor.do_single_step(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            progress_callbacks=callbacks,
            completer=completer,
            context_name=desc.name,
        )

    # Assert reasoning is not stored in history anywhere
    for entry in state.history:
        assert getattr(entry, "reasoning_content", None) is None


# Guard rails for do_single_step


@pytest.mark.asyncio
async def test_auto_inject_builtin_tools() -> None:
    # Tools are empty initially
    desc, state = make_test_agent(tools=[], history=[UserMessage(content="start")])
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

    ui = make_ui_mock()
    tools_with_meta = list(ctx.desc.tools)
    if not any(tool.name() == "finish_task" for tool in tools_with_meta):
        tools_with_meta.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversation())
    async with system_actor_scope_for_tests(tools=tools_with_meta, ui=ui, context_name=ctx.desc.name) as actors:
        await actors.agent_actor.run_agent_loop(
            ctx,
            tools=tools_with_meta,
            progress_callbacks=NullProgressCallbacks(),
            completer=completer,
            compact_conversation_at_tokens=1000,
            tool_call_actor_uri=actors.tool_call_actor_uri,
        )

    assert state.output is not None
    assert state.output.result == "ok"


@pytest.mark.asyncio
async def test_requires_non_empty_history() -> None:
    desc, state = make_test_agent(tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[])
    with pytest.raises(RuntimeError, match="History is required in order to run a step."):
        async with agent_actor_scope(context_name=desc.name) as agent_actor:
            await agent_actor.do_single_step(
                history=state.history,
                model=desc.model,
                tools=desc.tools,
                completer=FakeCompleter([AssistantMessage(content="hi")]),
                context_name=desc.name,
                progress_callbacks=NullProgressCallbacks(),
            )
