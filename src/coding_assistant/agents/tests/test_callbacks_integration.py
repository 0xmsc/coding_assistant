import json
from unittest.mock import Mock

import pytest

from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.types import TextResult, Tool
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


class EchoTool(Tool):
    def name(self) -> str: return "echo"
    def description(self) -> str: return "echo"
    def parameters(self) -> dict: return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    async def execute(self, parameters: dict) -> TextResult: return TextResult(content=parameters["text"])


@pytest.mark.asyncio
async def test_on_agent_start_end_called_with_expected_args():
    callbacks = Mock()
    finish = FakeToolCall("f1", FakeFunction("finish_task", json.dumps({"result": "r", "summary": "s"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[finish])])
    agent = make_test_agent(tools=[FinishTaskTool(), ShortenConversation()])

    await run_agent_loop(
        agent,
        callbacks,
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    callbacks.on_agent_start.assert_called_once()
    callbacks.on_agent_end.assert_called_once_with(agent.name, "r", "s")


@pytest.mark.asyncio
async def test_on_tool_message_called_with_arguments_and_result():
    callbacks = Mock()
    call = FakeToolCall("1", FakeFunction("echo", json.dumps({"text": "hello"})))
    finish = FakeToolCall("2", FakeFunction("finish_task", json.dumps({"result": "ok", "summary": "s"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[call]), FakeMessage(tool_calls=[finish])])
    agent = make_test_agent(tools=[EchoTool(), FinishTaskTool(), ShortenConversation()])

    await run_agent_loop(
        agent,
        callbacks,
        shorten_conversation_at_tokens=200_000,
        no_truncate_tools=set(),
        enable_user_feedback=False,
        completer=completer,
        ui=make_ui_mock(),
    )

    # Verify at least one on_tool_message call with expected payload
    found = False
    for call_args in callbacks.on_tool_message.call_args_list:
        # on_tool_message is called positionally in code; args tuple
        args = call_args[0]
        if len(args) == 4 and args[0] == agent.name and args[1] == "echo" and args[2] == {"text": "hello"} and args[3] == "hello":
            found = True
            break
    assert found, "Expected on_tool_message to be called with echo arguments and result"
