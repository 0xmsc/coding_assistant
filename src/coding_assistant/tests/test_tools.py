import pytest
from coding_assistant.config import Config
from coding_assistant.tools.tools import AskClientTool, ExecuteShellCommandTool
from unittest.mock import patch


@pytest.mark.asyncio
async def test_execute_shell_command_tool_timeout():
    tool = ExecuteShellCommandTool()
    result = await tool.execute({"command": "sleep 2", "timeout": 1})
    assert "timed out" in result.content


@pytest.mark.asyncio
@patch("rich.prompt.Prompt.ask")
async def test_execute_shell_command_tool_confirmation_yes(mock_ask):
    mock_ask.return_value = "y"
    tool = ExecuteShellCommandTool(ask_shell_confirmation_patterns=[r"^\s*rm"])
    result = await tool.execute({"command": "rm -rf /"})
    assert "denied" not in result.content.lower()


@pytest.mark.asyncio
@patch("rich.prompt.Prompt.ask")
async def test_execute_shell_command_tool_confirmation_no(mock_ask):
    mock_ask.return_value = "n"
    tool = ExecuteShellCommandTool(ask_shell_confirmation_patterns=[r"^\s*rm"])
    result = await tool.execute({"command": "rm -rf /"})
    assert "denied" in result.content.lower()


@pytest.mark.asyncio
@patch("rich.prompt.Prompt.ask")
async def test_execute_shell_command_tool_no_match(mock_ask):
    tool = ExecuteShellCommandTool(ask_shell_confirmation_patterns=[r"^\s*rm"])
    result = await tool.execute({"command": "ls -l"})
    assert "denied" not in result.content.lower()
    mock_ask.assert_not_called()


@pytest.mark.asyncio
@patch('rich.prompt.Prompt.ask')
async def test_ask_client_tool_enabled(mock_ask):
    mock_ask.return_value = "yes"
    tool = AskClientTool(enabled=True)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert result.content == "yes"
    mock_ask.assert_called_once_with("Do you want to proceed?", default="no")

@pytest.mark.asyncio
async def test_ask_client_tool_disabled():
    tool = AskClientTool(enabled=False)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert "disabled" in result.content.lower()
