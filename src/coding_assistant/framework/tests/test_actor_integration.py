import json
from typing import Any

import pytest

from coding_assistant.framework.builtin_tools import CompactConversationTool as CompactConversation
from coding_assistant.framework.builtin_tools import FinishTaskTool
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FakeMessage,
    FunctionCall,
    ToolCall,
    make_test_agent,
    make_ui_mock,
    system_actor_scope_for_tests,
)
from coding_assistant.framework.types import AgentContext
from coding_assistant.framework.results import TextResult
from coding_assistant.llm.types import BaseMessage, NullProgressCallbacks, Tool, UserMessage
from coding_assistant.tools.tools import AgentTool
from coding_assistant.ui import NullUI


class FakeEchoTool(Tool):
    def __init__(self) -> None:
        self.called_with: Any = None

    def name(self) -> str:
        return "fake.echo"

    def description(self) -> str:
        return "Echo a provided text"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        self.called_with = parameters
        return TextResult(content=f"echo: {parameters['text']}")


@pytest.mark.asyncio
async def test_system_actor_chat_loop_executes_tool_then_exits() -> None:
    echo_call = ToolCall("1", FunctionCall("fake.echo", json.dumps({"text": "hi"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[echo_call]), FakeMessage(content="done")])

    history: list[BaseMessage] = [UserMessage(content="start")]
    model = "test-model"
    instructions = None
    callbacks = NullProgressCallbacks()

    echo_tool = FakeEchoTool()
    ui = make_ui_mock(ask_sequence=[("> ", "run"), ("> ", "/exit")])

    async with system_actor_scope_for_tests(
        tools=[echo_tool],
        ui=ui,
        context_name="test",
        progress_callbacks=callbacks,
    ) as actors:
        await actors.agent_actor.run_chat_loop(
            history=history,
            model=model,
            tools=[echo_tool],
            instructions=instructions,
            callbacks=callbacks,
            completer=completer,
            context_name="test",
            user_actor=actors.user_actor,
            tool_call_actor=actors.tool_call_actor,
        )

    assert echo_tool.called_with == {"text": "hi"}


@pytest.mark.asyncio
async def test_system_actor_agent_loop_finishes() -> None:
    finish_call = ToolCall("f1", FunctionCall("finish_task", json.dumps({"result": "ok", "summary": "sum"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])
    desc, state = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])

    callbacks = NullProgressCallbacks()
    ui = make_ui_mock()

    async with system_actor_scope_for_tests(
        tools=desc.tools,
        ui=ui,
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actors:
        await actors.agent_actor.run_agent_loop(
            AgentContext(desc=desc, state=state),
            tools=desc.tools,
            progress_callbacks=callbacks,
            completer=completer,
            compact_conversation_at_tokens=200_000,
            tool_call_actor=actors.tool_call_actor,
        )
    assert state.output is not None
    assert state.output.result == "ok"


def test_agent_tool_requires_actor_dependencies() -> None:
    with pytest.raises(RuntimeError, match="AgentTool requires actor-backed dependencies"):
        AgentTool(
            model="test-model",
            expert_model="test-expert-model",
            compact_conversation_at_tokens=200_000,
            enable_ask_user=False,
            tools=[],
            ui=NullUI(),
            agent_actor=None,  # type: ignore[arg-type]
            tool_call_actor=None,  # type: ignore[arg-type]
            user_actor=None,  # type: ignore[arg-type]
        )
