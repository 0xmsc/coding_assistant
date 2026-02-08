from typing import Any
import json
from unittest.mock import Mock

import pytest

from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    make_test_agent,
    make_ui_mock,
    system_actor_scope_for_tests,
)
from coding_assistant.llm.types import AssistantMessage, ToolCall, FunctionCall, ToolMessage, Tool
from coding_assistant.framework.types import AgentContext
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


class EchoTool(Tool):
    def name(self) -> str:
        return "echo"

    def description(self) -> str:
        return "echo"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        return TextResult(content=parameters["text"])


@pytest.mark.asyncio
async def test_agent_loop_runs_successfully() -> None:
    callbacks = Mock()
    finish = ToolCall("f1", FunctionCall("finish_task", json.dumps({"result": "r", "summary": "s"})))
    completer = FakeCompleter([AssistantMessage(tool_calls=[finish])])
    desc, state = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])

    ui = make_ui_mock()
    async with system_actor_scope_for_tests(
        tools=desc.tools,
        ui=ui,
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actors:
        await run_agent_loop(
            AgentContext(desc=desc, state=state),
            progress_callbacks=callbacks,
            compact_conversation_at_tokens=200_000,
            completer=completer,
            ui=actors.user_actor,
            system_actors=actors,
        )

    assert state.output is not None
    assert state.output.result == "r"


@pytest.mark.asyncio
async def test_on_tool_message_called_with_arguments_and_result() -> None:
    callbacks = Mock()
    call = ToolCall("1", FunctionCall("echo", json.dumps({"text": "hello"})))
    finish = ToolCall("2", FunctionCall("finish_task", json.dumps({"result": "ok", "summary": "s"})))
    completer = FakeCompleter([AssistantMessage(tool_calls=[call]), AssistantMessage(tool_calls=[finish])])
    desc, state = make_test_agent(tools=[EchoTool(), FinishTaskTool(), CompactConversation()])

    ui = make_ui_mock()
    async with system_actor_scope_for_tests(
        tools=desc.tools,
        ui=ui,
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actors:
        await run_agent_loop(
            AgentContext(desc=desc, state=state),
            progress_callbacks=callbacks,
            compact_conversation_at_tokens=200_000,
            completer=completer,
            ui=actors.user_actor,
            system_actors=actors,
        )

    found = False
    for call_args in callbacks.on_tool_message.call_args_list:
        # on_tool_message is called positionally in code; args tuple
        args = call_args[0]
        if (
            len(args) == 4
            and args[0] == desc.name
            and isinstance(args[1], ToolMessage)
            and args[1].tool_call_id == "1"
            and args[1].content == "hello"
            and args[2] == "echo"
            and args[3] == {"text": "hello"}
        ):
            found = True
            break
    assert found, "Expected on_tool_message to be called with echo arguments and result"
