import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.config import Config
from coding_assistant.tools.tools import AgentTool
from coding_assistant.ui import NullUI

# This file contains integration tests using the real LLM API.

TEST_MODEL = "openrouter/openai/gpt-5-mini"


def create_test_config() -> Config:
    """Helper function to create a test Config with all required parameters."""
    return Config(
        model=TEST_MODEL,
        expert_model=TEST_MODEL,
        compact_conversation_at_tokens=200_000,
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool():
    config = create_test_config()
    tool = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=[],
        history=None,
        progress_callbacks=NullProgressCallbacks(),
        ui=NullUI(),
        tool_callbacks=NullToolCallbacks(),
    )
    result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_resume():
    config = create_test_config()
    first = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=[],
        history=None,
        progress_callbacks=NullProgressCallbacks(),
        ui=NullUI(),
        tool_callbacks=NullToolCallbacks(),
    )

    result = await first.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"

    second = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=[],
        history=first.history,
        progress_callbacks=NullProgressCallbacks(),
        ui=NullUI(),
        tool_callbacks=NullToolCallbacks(),
    )
    result = await second.execute(
        parameters={"task": "Re-do your previous task, just translate your output to German."}
    )
    assert result.content == "Hallo, Welt!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_instructions():
    config = create_test_config()
    tool = AgentTool(
        model=config.model,
        expert_model=config.expert_model,
        compact_conversation_at_tokens=config.compact_conversation_at_tokens,
        enable_ask_user=config.enable_ask_user,
        tools=[],
        history=None,
        progress_callbacks=NullProgressCallbacks(),
        ui=NullUI(),
        tool_callbacks=NullToolCallbacks(),
    )
    result = await tool.execute(
        parameters={
            "task": "Say 'Hello, World!'",
            "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
        }
    )
    assert result.content == "Servus, World!"
