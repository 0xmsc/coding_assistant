import json
import pytest

from coding_assistant.agents.tests.helpers import (
    FakeCompleter,
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    make_test_agent,
    make_ui_mock,
)
from coding_assistant.agents.execution import run_chat_loop
from coding_assistant.agents.types import Tool, TextResult, AgentContext
from coding_assistant.agents.callbacks import NullProgressCallbacks, NullToolCallbacks


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
async def test_chat_step_prompts_user_on_no_tool_calls_once():
    # Assistant emits no tool calls -> in chat mode we should prompt the user once and append reply
    completer = FakeCompleter([FakeMessage(content="Hello")])
    desc, state = make_test_agent(tools=[], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(ask_sequence=[("> ", "User reply"), ("> ", "User reply 2")])

    # Run a single chat-loop iteration by exhausting the completer after one step
    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            AgentContext(desc=desc, state=state),
            agent_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Should prompt first, then assistant responds, then prompt again
    roles = [m.get("role") for m in state.history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_step_executes_tools_without_prompt():
    echo_call = FakeToolCall("1", FakeFunction("fake.echo", json.dumps({"text": "hi"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[echo_call])])

    echo_tool = FakeEchoTool()
    desc, state = make_test_agent(tools=[echo_tool], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(ask_sequence=[("> ", "Hi")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            AgentContext(desc=desc, state=state),
            agent_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Tool must have executed
    assert echo_tool.called_with == {"text": "hi"}


@pytest.mark.asyncio
async def test_chat_mode_does_not_require_finish_task_tool():
    # No finish_task tool; chat mode should still allow a step
    completer = FakeCompleter([FakeMessage(content="Hi there")])
    desc, state = make_test_agent(tools=[], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(ask_sequence=[("> ", "Ack"), ("> ", "Ack 2")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            AgentContext(desc=desc, state=state),
            agent_callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Should be assistant followed by next user prompt
    roles = [m.get("role") for m in state.history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_exit_command_stops_loop_without_appending_command():
    # Assistant sends a normal message, user replies with /exit which should stop the loop
    completer = FakeCompleter([FakeMessage(content="Hello chat")])
    desc, state = make_test_agent(tools=[], history=[{"role": "user", "content": "start"}])

    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    # Should return cleanly without exhausting the completer further
    await run_chat_loop(
        AgentContext(desc=desc, state=state),
        agent_callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    # Verify that '/exit' was not appended to history
    assert not any(m.get("role") == "user" and (m.get("content") or "").strip() == "/exit" for m in state.history)
    # No assistant step should have happened; last message remains the start message
    assert state.history[-1]["role"] == "user"
