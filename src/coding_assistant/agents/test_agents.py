import pytest
from unittest.mock import patch, MagicMock

from coding_assistant.agents.logic import Agent, FeedbackTool
from coding_assistant.config import Config


@pytest.mark.asyncio
async def test_feedback_tool_execute_ok(tmp_path):
    config = Config(working_directory=tmp_path, model="o4-mini")
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={"task": "Give me the result of 2 + 2", "output": "4"}
    )
    assert result == "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_wrong(tmp_path):
    config = Config(working_directory=tmp_path, model="o4-mini")
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={"task": "Give me the result of 2 + 2", "output": "3"}
    )
    assert result != "Ok"


@pytest.mark.asyncio
async def test_feedback_tool_execute_no_result(tmp_path):
    config = Config(working_directory=tmp_path, model="o4-mini")
    tool = FeedbackTool(config=config)
    result = await tool.execute(
        parameters={
            "task": "Give me the result of 2 + 2",
            "output": "I calculated the result of 2 + 2 and gave it to the user.",
        }
    )
    assert result != "Ok"
