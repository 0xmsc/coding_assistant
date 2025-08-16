import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import handle_tool_call
from coding_assistant.agents.types import Agent, TextResult, Tool
from coding_assistant.config import Config
from coding_assistant.tools.tools import FeedbackTool

TEST_MODEL = "openai/gpt-5-mini"


def create_test_config() -> Config:
    """Helper function to create a test Config with all required parameters."""
    return Config(
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        enable_feedback_agent=True,
        enable_user_feedback=False,
        shorten_conversation_at_tokens=200_000,
        enable_ask_user=False,
        shell_confirmation_patterns=[],
        tool_confirmation_patterns=[],
        no_truncate_tools=set(),
    )


@pytest.mark.asyncio
async def test_feedback_tool_execute_ok():
    config = create_test_config()
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "4",
        }
    )
    assert result.content == "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_wrong():
    config = create_test_config()
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "5",
        }
    )
    assert result.content != "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_no_result():
    config = create_test_config()
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "I calculated the result of 2 + 2 and gave it to the user.",
        }
    )
    assert result.content != "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_after_feedback():
    config = create_test_config()
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "5",
            "summary": "I calculated the result of '2 + 3'. I did not calculate the answer to the original question since the client gave me the feedback that he made a mistake while asking the question. What he wanted to ask was 'what is 2 + 3?'. He confirmed that he wants me to give an answer to the updated question.",
        }
    )
    assert result.content == "Ok"


class FakeBigOutputTool(Tool):
    def name(self) -> str:
        return "fake_tool.big_output"

    def description(self) -> str:
        return "Return a very large text payload for testing truncation behavior."

    def parameters(self) -> dict:
        return {}

    async def execute(self, parameters: dict) -> TextResult:
        return TextResult(content="X" * 60_000)


class _Fn:
    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, function: _Fn):
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

    tool_call = _ToolCall(id="1", function=_Fn(name="fake_tool.big_output", arguments="{}"))
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

    tool_call = _ToolCall(id="1", function=_Fn(name="fake_tool.big_output", arguments="{}"))
    await handle_tool_call(tool_call, agent, NullCallbacks(), no_truncate_tools={r"^fake_tool\.big_"})

    assert agent.history, "Expected a tool message to be appended to history"
    assert agent.history[-1]["content"] == "X" * 60_000
