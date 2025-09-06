import pytest

from coding_assistant.agents.tests.helpers import make_ui_mock
from coding_assistant.config import Config
from coding_assistant.tools.tools import OrchestratorTool
from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.ui import NullUI

# This file contains integration tests using the real LLM API.

TEST_MODEL = "openai/gpt-5-mini"


def create_test_config() -> Config:
    """Helper function to create a test Config with all required parameters."""
    return Config(
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        enable_user_feedback=False,
        shorten_conversation_at_tokens=200_000,
        enable_ask_user=False,
        shell_confirmation_patterns=[],
        tool_confirmation_patterns=[],
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = OrchestratorTool(config=config, tools=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_resume():
    config = create_test_config()
    first = OrchestratorTool(config=config, tools=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())

    result = await first.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"

    second = OrchestratorTool(config=config, tools=[], history=first.history, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await second.execute(
        parameters={"task": "Re-do your previous task, just translate your output to German."}
    )
    assert result.content == "Hallo, Welt!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_instructions():
    config = create_test_config()
    tool = OrchestratorTool(config=config, tools=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "task": "Say 'Hello, World!'",
            "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
        }
    )
    assert result.content == "Servus, World!"


def _create_confirmation_orchestrator(confirm_value: bool):
    config = create_test_config()
    config.shell_confirmation_patterns = ["^echo"]

    ui = make_ui_mock(
        confirm_sequence=[
            ("Execute `echo Hello World`?", confirm_value),
        ]
    )
    tool = OrchestratorTool(config=config, tools=[], history=None, agent_callbacks=NullCallbacks(), ui=ui)
    parameters = {
        "task": "Execute shell command 'echo Hello World' with a timeout of 10 seconds and verbatim output stdout. If the command execution is denied, output 'Command execution denied.'",
    }
    return tool, parameters


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shell_confirmation_positive():
    tool, parameters = _create_confirmation_orchestrator(confirm_value=True)
    result = await tool.execute(parameters=parameters)
    assert result.content.strip() == "Hello World"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_shell_confirmation_negative():
    tool, parameters = _create_confirmation_orchestrator(confirm_value=False)
    result = await tool.execute(parameters=parameters)
    assert result.content.strip() == "Command execution denied."


def _create_tool_confirmation_orchestrator(confirm_value: bool):
    config = create_test_config()
    config.tool_confirmation_patterns = ["^execute_shell_command"]

    ui = make_ui_mock(
        confirm_sequence=[
            (
                "Execute tool `execute_shell_command` with arguments `{'command': \"echo 'Hello, World!'\", 'timeout': 10}`?",
                confirm_value,
            ),
        ]
    )
    tool = OrchestratorTool(config=config, tools=[], history=None, agent_callbacks=NullCallbacks(), ui=ui)
    parameters = {
        "task": "Use the execute_shell_command to echo 'Hello, World!' with a timeout of 10 seconds and verbatim output stdout. If the tool execution is denied, output 'Tool execution denied.'",
    }
    return tool, parameters


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_confirmation_positive():
    tool, parameters = _create_tool_confirmation_orchestrator(confirm_value=True)
    result = await tool.execute(parameters=parameters)
    assert result.content.strip() == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tool_confirmation_negative():
    tool, parameters = _create_tool_confirmation_orchestrator(confirm_value=False)
    result = await tool.execute(parameters=parameters)
    assert result.content.strip() == "Tool execution denied."
