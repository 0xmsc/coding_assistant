import pytest
from unittest.mock import patch, MagicMock

from coding_assistant.agents.logic import Agent
from coding_assistant.agents.tools import FeedbackTool, OrchestratorTool
from coding_assistant.config import Config
from coding_assistant.tools import Tools

TEST_MODEL = "gpt-4.1"


@pytest.mark.asyncio
async def test_feedback_tool_execute_ok(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = FeedbackTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "output": "4",
        }
    )
    assert result == "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_wrong(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = FeedbackTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "output": "5",
        }
    )
    assert result != "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_no_result(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = FeedbackTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "output": "I calculated the result of 2 + 2 and gave it to the user.",
        }
    )
    assert result != "Ok"


@pytest.mark.asyncio
async def test_orchestrator_tool(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = OrchestratorTool(config=config, tools=Tools())
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result == "Hello, World!"
