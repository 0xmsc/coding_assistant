import json

import pytest

from coding_assistant.agents.tests.helpers import make_ui_mock
from coding_assistant.tools.tools import AskClientTool, ExecuteShellCommandTool
from coding_assistant.ui import NullUI


@pytest.mark.asyncio
async def test_execute_shell_command_tool_timeout():
    tool = ExecuteShellCommandTool()
    result = await tool.execute({"command": "sleep 2", "timeout": 1})
    assert "timed out" in result.content


@pytest.mark.asyncio
async def test_execute_shell_command_tool_confirmation_yes():
    ui = make_ui_mock(confirm_sequence=[("Execute `echo 'Hello'`?", True)])
    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"], ui=ui)
    result = await tool.execute({"command": "echo 'Hello'"})

    expected = {"stdout": "Hello\n", "stderr": "", "returncode": 0}
    assert json.loads(result.content) == expected


@pytest.mark.asyncio
async def test_execute_shell_command_tool_confirmation_yes_with_whitespace():
    ui = make_ui_mock(confirm_sequence=[("Execute `echo 'Hello'`?", False)])
    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"], ui=ui)
    result = await tool.execute({"command": "  echo 'Hello'"})
    assert result.content == "Command execution denied."


@pytest.mark.asyncio
async def test_execute_shell_command_tool_confirmation_no():
    ui = make_ui_mock(confirm_sequence=[("Execute `echo 'Hello'`?", False)])
    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["echo"], ui=ui)
    result = await tool.execute({"command": "echo 'Hello'"})
    assert result.content == "Command execution denied."


@pytest.mark.asyncio
async def test_execute_shell_command_tool_no_match():
    ui = make_ui_mock()
    tool = ExecuteShellCommandTool(shell_confirmation_patterns=["ls"], ui=ui)
    result = await tool.execute({"command": "echo 'Hello'"})

    expected = {"stdout": "Hello\n", "stderr": "", "returncode": 0}
    assert json.loads(result.content) == expected
    ui.confirm.assert_not_called()


@pytest.mark.asyncio
async def test_ask_client_tool_enabled():
    ui = make_ui_mock(ask_sequence=[("Do you want to proceed?", "yes")])
    tool = AskClientTool(enabled=True, ui=ui)
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert result.content == "yes"


@pytest.mark.asyncio
async def test_ask_client_tool_disabled():
    tool = AskClientTool(enabled=False, ui=NullUI())
    result = await tool.execute({"question": "Do you want to proceed?", "default_answer": "no"})
    assert (
        result.content
        == "Client input is disabled for this session. Please continue as if the client had given the most sensible answer to your question."
    )
