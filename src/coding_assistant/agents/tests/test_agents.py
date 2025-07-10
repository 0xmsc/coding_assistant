from unittest.mock import MagicMock, patch

import pytest

from coding_assistant.agents.agents import FeedbackTool, OrchestratorTool
from coding_assistant.agents.logic import Agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

TEST_MODEL = "gemini/gemini-2.5-flash"


@pytest.mark.asyncio
async def test_feedback_tool_execute_ok(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = FeedbackTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "4",
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
            "result": "5",
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
            "result": "I calculated the result of 2 + 2 and gave it to the user.",
        }
    )
    assert result != "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_after_feedback(tmp_path):
    config = Config(working_directory=tmp_path, model=TEST_MODEL, disable_user_feedback=True)
    tool = FeedbackTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "description": "The agent will only give correct answers",
            "parameters": "What is 2 + 2?",
            "result": "5",
            "feedback": "The client made a mistake while asking the question, he meant 'what is 2 + 3? He wanted me to give the question to the updated answer.'",
        }
    )
    assert result == "Ok"


@pytest.mark.asyncio
async def test_orchestrator_tool(tmp_path):
    config = Config(
        working_directory=tmp_path,
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        disable_user_feedback=True,
    )
    tool = OrchestratorTool(config=config, tools=Tools())
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result == "Hello, World!"


@pytest.mark.asyncio
async def test_orchestrator_tool_instructions(tmp_path):
    config = Config(
        working_directory=tmp_path,
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        disable_user_feedback=True,
    )
    tool = OrchestratorTool(config=config, tools=Tools())
    result = await tool.execute(
        parameters={
            "task": "Say 'Hello, World!'",
            "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
        }
    )
    assert result == "Servus, World!"
