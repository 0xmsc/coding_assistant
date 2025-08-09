from unittest.mock import AsyncMock, patch

import pytest

from coding_assistant.config import Config
from coding_assistant.tools.tools import AskClientTool, ExecuteShellCommandTool, _get_feedback
from coding_assistant.agents.types import Agent, AgentOutput


@pytest.mark.asyncio
async def test_execute_shell_command_tool_timeout():
    tool = ExecuteShellCommandTool()
    result = await tool.execute({"command": "sleep 2", "timeout": 1})
    assert "timed out" in result.content


@pytest.mark.asyncio
@patch("coding_assistant.tools.tools.create_confirm_session")
async def test_execute_shell_command_tool_confirmation_yes(mock_create_confirm):
    mock_session = AsyncMock()
    mock_session.prompt_async.return_value = True
    mock_create_confirm.return_value = mock_session

    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"])
    result = await tool.execute({"command": "echo 'Hello'"})
    assert result.content == "Hello\n"


@pytest.mark.asyncio
@patch("coding_assistant.tools.tools.create_confirm_session")
async def test_execute_shell_command_tool_confirmation_yes_with_whitespace(mock_create_confirm):
    mock_session = AsyncMock()
    mock_session.prompt_async.return_value = False
    mock_create_confirm.return_value = mock_session

    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"])
    result = await tool.execute({"command": "  echo 'Hello'"})
    assert result.content == "Command execution denied."


@pytest.mark.asyncio
@patch("coding_assistant.tools.tools.create_confirm_session")
async def test_execute_shell_command_tool_confirmation_no(mock_create_confirm):
    mock_session = AsyncMock()
    mock_session.prompt_async.return_value = False
    mock_create_confirm.return_value = mock_session

    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"])
    result = await tool.execute({"command": "echo 'Hello'"})
    assert result.content == "Command execution denied."


@pytest.mark.asyncio
@patch("coding_assistant.tools.tools.create_confirm_session")
async def test_execute_shell_command_tool_no_match(mock_create_confirm):
    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["ls"])
    result = await tool.execute({"command": "echo 'Hello'"})
    assert result.content == "Hello\n"
    mock_create_confirm.assert_not_called()


@pytest.mark.asyncio
@patch("coding_assistant.tools.tools.PromptSession")
async def test_ask_client_tool_enabled(mock_prompt_session_class):
    mock_session = AsyncMock()
    mock_session.prompt_async.return_value = "yes"
    mock_prompt_session_class.return_value = mock_session
    
    tool = AskClientTool(enabled=True)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert result.content == "yes"
    mock_session.prompt_async.assert_called_once_with("Do you want to proceed? ", default="no")


@pytest.mark.asyncio
async def test_ask_client_tool_disabled():
    tool = AskClientTool(enabled=False)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert (
        result.content
        == "Client input is disabled for this session. Please continue as if the client had given the most sensible answer to your question."
    )
