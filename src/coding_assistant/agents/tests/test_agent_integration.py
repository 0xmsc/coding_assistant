from unittest.mock import AsyncMock, patch

import pytest

from coding_assistant.config import Config
from coding_assistant.tools.tools import OrchestratorTool

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


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = OrchestratorTool(config=config)
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.slow
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


@pytest.mark.slow
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
        "task": "Execute shell command 'echo Hello World' and verbatim output stdout. If the command execution is denied, output 'Command execution denied.'",
    }
    return tool, parameters


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shell_confirmation_positive():
    tool, parameters = _create_confirmation_orchestrator()

    with patch("coding_assistant.tools.tools.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = True
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Hello World"


@pytest.mark.slow
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
        "task": "Use the execute_shell_command to echo 'Hello, World!' and verbatim output stdout. If the tool execution is denied, output 'Tool execution denied.'",
    }
    return tool, parameters


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_confirmation_positive():
    tool, parameters = _create_tool_confirmation_orchestrator()

    with patch("coding_assistant.agents.execution.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = True
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_confirmation_negative():
    tool, parameters = _create_tool_confirmation_orchestrator()

    with patch("coding_assistant.agents.execution.create_confirm_session") as mock_create_confirm:
        mock_session = AsyncMock()
        mock_session.prompt_async.return_value = False
        mock_create_confirm.return_value = mock_session

        result = await tool.execute(parameters=parameters)
        assert result.content.strip() == "Tool execution denied."
