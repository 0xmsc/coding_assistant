import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import handle_tool_call
from coding_assistant.agents.tests.helpers import FakeFunction, FakeToolCall, make_test_agent, make_ui_mock
from coding_assistant.agents.types import Agent, TextResult, Tool


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
