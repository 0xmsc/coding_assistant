import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import do_single_step, handle_tool_call
from coding_assistant.agents.tests.helpers import FakeFunction, FakeToolCall, make_test_agent, make_ui_mock
from coding_assistant.agents.types import Agent, TextResult, Tool, ToolResult
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


class FakeBigOutputTool(Tool):
    def name(self) -> str:
        return "fake_tool.big_output"

    def description(self) -> str:
        return "Return a very large text payload for testing truncation behavior."

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content="X" * 60_000)


@pytest.mark.asyncio
async def test_no_truncate_blocks_large_output_by_default():
    agent = make_test_agent(model="TestModel", tools=[FakeBigOutputTool()])

    tool_call = FakeToolCall(id="1", function=FakeFunction(name="fake_tool.big_output", arguments="{}"))
    await handle_tool_call(tool_call, agent, NullCallbacks(), no_truncate_tools=set(), ui=make_ui_mock())

    assert agent.history, "Expected a tool message to be appended to history"
    assert (
        agent.history[-1]["content"]
        == "System error: Tool call result too long. Please use a tool or arguments that return shorter results."
    )


@pytest.mark.asyncio
async def test_no_truncate_allows_large_output_for_matching_tools():
    agent = make_test_agent(model="TestModel", tools=[FakeBigOutputTool()])

    tool_call = FakeToolCall(id="1", function=FakeFunction(name="fake_tool.big_output", arguments="{}"))
    await handle_tool_call(
        tool_call, agent, NullCallbacks(), no_truncate_tools={r"^fake_tool\.big_"}, ui=make_ui_mock()
    )

    assert agent.history, "Expected a tool message to be appended to history"
    assert agent.history[-1]["content"] == "X" * 60_000


class FakeConfirmTool(Tool):
    def __init__(self):
        self.calls: list[dict] = []

    def name(self) -> str:
        return "execute_shell_command"

    def description(self) -> str:
        return "Pretend to execute a shell command"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}

    async def execute(self, parameters: dict) -> TextResult:
        self.calls.append(parameters)
        return TextResult(content=f"ran: {parameters['cmd']}")


@pytest.mark.asyncio
async def test_tool_confirmation_denied_and_allowed():
    tool = FakeConfirmTool()
    agent = make_test_agent(
        tools=[tool],
        tool_confirmation_patterns=[r"^execute_shell_command"],
    )

    # Arguments will be parsed and shown as a Python dict in the confirm prompt
    args_json = '{"cmd": "echo 123"}'
    expected_prompt = "Execute tool `execute_shell_command` with arguments `{'cmd': 'echo 123'}`?"

    ui = make_ui_mock(confirm_sequence=[(expected_prompt, False), (expected_prompt, True)])

    # First: denied
    call1 = FakeToolCall(id="1", function=FakeFunction(name="execute_shell_command", arguments=args_json))
    await handle_tool_call(call1, agent, NullCallbacks(), no_truncate_tools=set(), ui=ui)

    assert tool.calls == []  # should not run
    assert agent.history[-1] == {
        "tool_call_id": "1",
        "role": "tool",
        "name": "execute_shell_command",
        "content": "Tool execution denied.",
    }

    # Second: allowed
    call2 = FakeToolCall(id="2", function=FakeFunction(name="execute_shell_command", arguments=args_json))
    await handle_tool_call(call2, agent, NullCallbacks(), no_truncate_tools=set(), ui=ui)

    assert tool.calls == [{"cmd": "echo 123"}]
    assert agent.history[-1] == {
        "tool_call_id": "2",
        "role": "tool",
        "name": "execute_shell_command",
        "content": "ran: echo 123",
    }



@pytest.mark.asyncio
async def test_unknown_result_type_raises():
    class WeirdResult(ToolResult):
        pass

    class WeirdTool(Tool):
        def name(self) -> str: return "weird"
        def description(self) -> str: return ""
        def parameters(self) -> dict: return {}
        async def execute(self, parameters: dict) -> ToolResult: return WeirdResult()

    agent = make_test_agent(model="TestModel", tools=[WeirdTool()])
    tool_call = FakeToolCall(id="1", function=FakeFunction(name="weird", arguments="{}"))
    with pytest.raises(TypeError, match=r"Unknown tool result type"):  # from handle_tool_call
        await handle_tool_call(tool_call, agent, NullCallbacks(), no_truncate_tools=set(), ui=make_ui_mock())

