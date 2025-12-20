import json
from unittest.mock import Mock

import pytest

from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.callbacks import NullToolCallbacks
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.framework.types import TextResult, Tool, AgentContext
from coding_assistant.tools.tools import FinishTaskTool, CompactConversation


class EchoTool(Tool):
    def name(self) -> str:
        return "echo"

    def description(self) -> str:
        return "echo"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content=parameters["text"])


@pytest.mark.asyncio
async def test_agent_loop_runs_successfully():
    callbacks = Mock()
    finish = FakeToolCall("f1", FakeFunction("finish_task", json.dumps({"result": "r", "summary": "s"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[finish])])
    desc, state = make_test_agent(tools=[FinishTaskTool(), CompactConversation()])

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=callbacks,
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )

    assert state.output.result == "r"


@pytest.mark.asyncio
async def test_on_tool_message_called_with_arguments_and_result():
    callbacks = Mock()
    call = FakeToolCall("1", FakeFunction("echo", json.dumps({"text": "hello"})))
    finish = FakeToolCall("2", FakeFunction("finish_task", json.dumps({"result": "ok", "summary": "s"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[call]), FakeMessage(tool_calls=[finish])])
    desc, state = make_test_agent(tools=[EchoTool(), FinishTaskTool(), CompactConversation()])

    await run_agent_loop(
        AgentContext(desc=desc, state=state),
        progress_callbacks=callbacks,
        tool_callbacks=NullToolCallbacks(),
        compact_conversation_at_tokens=200_000,
        completer=completer,
        ui=make_ui_mock(),
    )

    # Verify at least one on_tool_message call with expected payload
    found = False
    for call_args in callbacks.on_tool_message.call_args_list:
        # on_tool_message is called positionally in code; args tuple
        args = call_args[0]
        if (
            len(args) == 5
            and args[0] == desc.name
            and args[1] == "1"
            and args[2] == "echo"
            and args[3] == {"text": "hello"}
            and args[4] == "hello"
        ):
            found = True
            break
    assert found, "Expected on_tool_message to be called with echo arguments and result"
