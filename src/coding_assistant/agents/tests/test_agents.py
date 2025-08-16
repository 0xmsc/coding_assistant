from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import handle_tool_call
from coding_assistant.agents.types import Agent, AgentOutput, TextResult, Tool
from coding_assistant.config import Config
from coding_assistant.tools.tools import FeedbackTool, OrchestratorTool

TEST_MODEL = "openai/gpt-5-nano"


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

    # Make deterministic: the feedback agent should return Ok after rework
    with patch(
        "coding_assistant.tools.tools.run_agent_loop", new=AsyncMock(return_value=AgentOutput(result="Ok", summary=""))
    ):
        result = await tool.execute(
            parameters={
                "description": "The agent will only give correct answers",
                "parameters": "What is 2 + 2?",
                "result": "5",
                "summary": "I calculated the result of '2 + 3'. I did not calculate the answer to the original question since the client gave me the feedback that he made a mistake while asking the question. What he wanted to ask was 'what is 2 + 3?'. He confirmed that he wants me to give an answer to the updated question.",
            }
        )
    assert result.content == "Ok"


@pytest.mark.long
@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = OrchestratorTool(config=config)
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.long
@pytest.mark.asyncio
async def test_orchestrator_tool_resume():
    config = create_test_config()
    first = OrchestratorTool(config=config)

    result = await first.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"

    second = OrchestratorTool(config=config, history=first.history)
    result = await second.execute(
        parameters={"task": "Re-do your previous task, just translate your output to German."}
    )
    assert result.content == "Hallo, Welt!"


@pytest.mark.long
@pytest.mark.asyncio
async def test_orchestrator_tool_instructions():
    config = create_test_config()
    tool = OrchestratorTool(config=config)
    result = await tool.execute(
        parameters={
            "task": "Say 'Hello, World!'",
            "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
        }
    )
    assert result.content == "Servus, World!"


def _create_confirmation_orchestrator():
    config = create_test_config()
    config.shell_confirmation_patterns = ["^echo"]
    tool = OrchestratorTool(config=config)
    parameters = {
        "task": "Execute shell command 'echo Hello World' and verbatim output the stdout output. If the command execution is denied, output 'Command execution denied.'",
    }
    return tool, parameters


@pytest.mark.asyncio
async def test_shell_confirmation_positive():
    tool, parameters = _create_confirmation_orchestrator()

    with patch("coding_assistant.tools.tools.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = True
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Hello World"


@pytest.mark.asyncio
async def test_shell_confirmation_negative():
    tool, parameters = _create_confirmation_orchestrator()

    with patch("coding_assistant.tools.tools.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = False
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Command execution denied."


def _create_tool_confirmation_orchestrator():
    config = create_test_config()
    config.tool_confirmation_patterns = ["^execute_shell_command"]
    tool = OrchestratorTool(config=config)
    parameters = {
        "task": "Use the execute_shell_command to echo 'Hello, World!'. If the tool execution is denied, output 'Tool execution denied.'",
    }
    return tool, parameters


@pytest.mark.asyncio
async def test_tool_confirmation_positive():
    tool, parameters = _create_tool_confirmation_orchestrator()

    with patch("coding_assistant.agents.execution.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = True
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert "Hello, World!" in result.content


@pytest.mark.asyncio
async def test_tool_confirmation_negative():
    tool, parameters = _create_tool_confirmation_orchestrator()

    with patch("coding_assistant.agents.execution.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = False
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Tool execution denied."


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


@pytest.mark.asyncio
async def test_no_truncate_plumbs_through_orchestrator(monkeypatch):
    config = create_test_config()
    config.no_truncate_tools = {r"^mcp_context7_get-library-docs"}

    captured = {}

    async def fake_run_agent_loop(agent, agent_callbacks, shorten_conversation_at_tokens, no_truncate_tools):
        # Capture arguments for assertion outside
        captured["no_truncate_tools"] = no_truncate_tools
        return AgentOutput(result="ok", summary="sum")

    with patch("coding_assistant.tools.tools.run_agent_loop", new=AsyncMock(side_effect=fake_run_agent_loop)):
        tool = OrchestratorTool(config=config)
        result = await tool.execute(parameters={"task": "noop"})
        assert result.content == "ok"

    assert captured.get("no_truncate_tools") == config.no_truncate_tools
