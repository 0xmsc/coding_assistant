import pytest

from coding_assistant.agents.tests.helpers import make_ui_mock
from coding_assistant.config import Config
from coding_assistant.tools.tools import FeedbackTool, OrchestratorTool
from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.ui import NullUI

# This file contains integration tests using the real LLM API.

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
async def test_feedback_tool_execute_ok():
    config = create_test_config()
    tool = FeedbackTool(config=config, mcp_servers=[], agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "4",
        }
    )
    assert result.content == "Ok"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_feedback_tool_execute_wrong():
    config = create_test_config()
    tool = FeedbackTool(config=config, mcp_servers=[], agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "5",
        }
    )
    assert result.content != "Ok"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_feedback_tool_execute_no_result():
    config = create_test_config()
    tool = FeedbackTool(config=config, mcp_servers=[], agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "I calculated the result of 2 + 2 and gave it to the user.",
        }
    )
    assert result.content != "Ok"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_feedback_tool_after_feedback():
    config = create_test_config()
    tool = FeedbackTool(config=config, mcp_servers=[], agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "5",
            "summary": "I calculated the result of '2 + 3'. I did not calculate the answer to the original question since the client gave me the feedback that he made a mistake while asking the question. What he wanted to ask was 'what is 2 + 3?'. He confirmed that he wants me to give an answer to the updated question.",
        }
    )
    assert result.content == "Ok"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = OrchestratorTool(config=config, mcp_servers=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_resume():
    config = create_test_config()
    first = OrchestratorTool(config=config, mcp_servers=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())

    result = await first.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"

    second = OrchestratorTool(config=config, mcp_servers=[], history=first.history, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await second.execute(
        parameters={"task": "Re-do your previous task, just translate your output to German."}
    )
    assert result.content == "Hallo, Welt!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_instructions():
    config = create_test_config()
    tool = OrchestratorTool(config=config, mcp_servers=[], history=None, agent_callbacks=NullCallbacks(), ui=NullUI())
    result = await tool.execute(
        parameters={
            "task": "Say 'Hello, World!'",
            "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
        }
    )
    assert result.content == "Servus, World!"


def _create_confirmation_orchestrator(confirm_value: bool):
    config = create_test_config()
    config.enable_feedback_agent = False
    config.shell_confirmation_patterns = ["^echo"]

    ui = make_ui_mock(
        confirm_sequence=[
            ("Execute `echo Hello World`?", confirm_value),
        ]
    )
    tool = OrchestratorTool(config=config, mcp_servers=[], history=None, agent_callbacks=NullCallbacks(), ui=ui)
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
    config.enable_feedback_agent = False
    config.tool_confirmation_patterns = ["^execute_shell_command"]

    ui = make_ui_mock(
        confirm_sequence=[
            (
                "Execute tool `execute_shell_command` with arguments `{'command': \"echo 'Hello, World!'\", 'timeout': 10}`?",
                confirm_value,
            ),
        ]
    )
    tool = OrchestratorTool(config=config, mcp_servers=[], history=None, agent_callbacks=NullCallbacks(), ui=ui)
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
