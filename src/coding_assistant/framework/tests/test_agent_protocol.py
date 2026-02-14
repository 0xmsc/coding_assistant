from typing import Any
import pytest
from unittest.mock import Mock

from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    agent_actor_scope,
    make_test_agent,
    make_ui_mock,
    run_agent_via_messages,
    system_actor_scope_for_tests,
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
    finish_call = ToolCall(
        id="finish_1",
        function=FunctionCall(name="finish_task", arguments='{"result": "ok", "summary": "done"}'),
    )
    completer = FakeCompleter([fake_message, AssistantMessage(tool_calls=[finish_call])])

    desc, state = make_test_agent(
        tools=[DummyTool(), FinishTaskTool(), CompactConversation()], history=[UserMessage(content="start")]
    )
    ctx = AgentContext(desc=desc, state=state)

    async with system_actor_scope_for_tests(tools=desc.tools, ui=make_ui_mock(), context_name=desc.name) as actors:
        await run_agent_via_messages(
            actors,
            ctx=ctx,
            tools=desc.tools,
            completer=completer,
            progress_callbacks=NullProgressCallbacks(),
            compact_conversation_at_tokens=1000,
        )

    assert state.output is not None
    assert state.output.result == "ok"
    assert any(isinstance(entry, ToolMessage) and entry.tool_call_id == "call_1" for entry in state.history)
    assert any(
        isinstance(entry, UserMessage)
        and isinstance(entry.content, str)
        and "Your conversation history has grown too large." in entry.content
        for entry in state.history
    )


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
        await run_agent_via_messages(
            actors,
            ctx=ctx,
            tools=tools_with_meta,
            progress_callbacks=NullProgressCallbacks(),
            completer=completer,
            compact_conversation_at_tokens=1000,
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
