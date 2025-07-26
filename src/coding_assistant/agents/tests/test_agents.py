from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coding_assistant.agents.execution import create_start_message
from coding_assistant.agents.types import Agent
from coding_assistant.config import Config
from coding_assistant.tools.tools import FeedbackTool, OrchestratorTool

TEST_MODEL = "gemini/gemini-2.5-flash"


def create_test_config() -> Config:
    """Helper function to create a test Config with all required parameters."""
    return Config(
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        enable_feedback_agent=True,
        enable_user_feedback=False,
        instructions=None,
        readable_sandbox_directories=[],
        writable_sandbox_directories=[],
        mcp_servers=[],
        shorten_conversation_at_tokens=200_000,
        enable_ask_user=False,
        print_chunks=False,
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
            "feedback": "The client made a mistake while asking the question, he meant 'what is 2 + 3?'. He wanted me to give an answer to the updated question.",
        }
    )
    assert result.content == "Ok"


@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = OrchestratorTool(config=config)
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.asyncio
async def test_orchestrator_tool_resume():
    config = create_test_config()
    first = OrchestratorTool(config=config)

    result = await first.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"

    # Mock the Prompt.ask to return a default answer for translation question
    with patch("rich.prompt.Prompt.ask", return_value="Hallo, Welt!"):
        second = OrchestratorTool(config=config, history=first.history)
        result = await second.execute(
            parameters={"task": "Re-do your previous task, just translate your output to German."}
        )
        assert result.content == "Hallo, Welt!"


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
