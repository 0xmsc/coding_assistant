import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import handle_tool_call
from coding_assistant.agents.types import Agent, TextResult, Tool

TEST_MODEL = "openai/gpt-5-mini"


class FakeBigOutputTool(Tool):
    def name(self) -> str:
        return "fake_tool.big_output"

    def description(self) -> str:
        return "Return a very large text payload for testing truncation behavior."

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content="X" * 60_000)


class TestFunction:
    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class TestToolCall:
    def __init__(self, id: str, function: TestFunction):
        self.id = id
        self.function = function


@pytest.mark.asyncio
async def test_no_truncate_blocks_large_output_by_default():
    agent = Agent(
        name="TestAgent",
        model=TEST_MODEL,
        description="",
        parameters=[],
        feedback_function=lambda agent: None,
        tools=[FakeBigOutputTool()],
        mcp_servers=[],
        tool_confirmation_patterns=[],
        history=[],
    )

    tool_call = TestToolCall(id="1", function=TestFunction(name="fake_tool.big_output", arguments="{}"))
    await handle_tool_call(tool_call, agent, NullCallbacks(), no_truncate_tools=set())

    assert agent.history, "Expected a tool message to be appended to history"
    assert (
        agent.history[-1]["content"]
        == "System error: Tool call result too long. Please use a tool or arguments that return shorter results."
    )


@pytest.mark.asyncio
async def test_no_truncate_allows_large_output_for_matching_tools():
    agent = Agent(
        name="TestAgent",
        model=TEST_MODEL,
        description="",
        parameters=[],
        feedback_function=lambda agent: None,
        tools=[FakeBigOutputTool()],
        mcp_servers=[],
        tool_confirmation_patterns=[],
        history=[],
    )

    tool_call = TestToolCall(id="1", function=TestFunction(name="fake_tool.big_output", arguments="{}"))
    await handle_tool_call(tool_call, agent, NullCallbacks(), no_truncate_tools={r"^fake_tool\.big_"})

    assert agent.history, "Expected a tool message to be appended to history"
    assert agent.history[-1]["content"] == "X" * 60_000
