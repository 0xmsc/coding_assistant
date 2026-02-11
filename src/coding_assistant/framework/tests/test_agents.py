import pytest

from coding_assistant.config import Config
from coding_assistant.framework.tests.helpers import system_actor_scope_for_tests
from coding_assistant.tools.tools import AgentTool
from coding_assistant.ui import NullUI

# This file contains integration tests using the real LLM API.

TEST_MODEL = "openai/gpt-5-mini"


def create_test_config() -> Config:
    """Helper function to create a test Config with all required parameters."""
    return Config(model=TEST_MODEL, expert_model=TEST_MODEL, compact_conversation_at_tokens=200_000)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool() -> None:
    config = create_test_config()
    ui = NullUI()
    async with system_actor_scope_for_tests(tools=[], ui=ui, context_name="test") as actors:
        tool = AgentTool(
            model=config.model,
            expert_model=config.expert_model,
            compact_conversation_at_tokens=config.compact_conversation_at_tokens,
            enable_ask_user=config.enable_ask_user,
            tools=[],
            history=None,
            ui=ui,
            agent_actor=actors.agent_actor,
            tool_call_actor_uri=actors.tool_call_actor_uri,
            user_actor_uri=actors.user_actor_uri,
        )
        result = await tool.execute(parameters={"task": "Say 'Hello, World!'"})
    assert result.content == "Hello, World!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_resume() -> None:
    config = create_test_config()
    ui = NullUI()
    async with system_actor_scope_for_tests(tools=[], ui=ui, context_name="test") as actors:
        first = AgentTool(
            model=config.model,
            expert_model=config.expert_model,
            compact_conversation_at_tokens=config.compact_conversation_at_tokens,
            enable_ask_user=config.enable_ask_user,
            tools=[],
            history=None,
            ui=ui,
            agent_actor=actors.agent_actor,
            tool_call_actor_uri=actors.tool_call_actor_uri,
            user_actor_uri=actors.user_actor_uri,
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
            ui=ui,
            agent_actor=actors.agent_actor,
            tool_call_actor_uri=actors.tool_call_actor_uri,
            user_actor_uri=actors.user_actor_uri,
        )
        result = await second.execute(
            parameters={"task": "Re-do your previous task, just translate your output to German."}
        )
    assert result.content == "Hallo, Welt!"


@pytest.mark.slow
@pytest.mark.asyncio
async def test_orchestrator_tool_instructions() -> None:
    config = create_test_config()
    ui = NullUI()
    async with system_actor_scope_for_tests(tools=[], ui=ui, context_name="test") as actors:
        tool = AgentTool(
            model=config.model,
            expert_model=config.expert_model,
            compact_conversation_at_tokens=config.compact_conversation_at_tokens,
            enable_ask_user=config.enable_ask_user,
            tools=[],
            history=None,
            ui=ui,
            agent_actor=actors.agent_actor,
            tool_call_actor_uri=actors.tool_call_actor_uri,
            user_actor_uri=actors.user_actor_uri,
        )
        result = await tool.execute(
            parameters={
                "task": "Say 'Hello, World!'",
                "instructions": "When you are told to say 'Hello', actually say 'Servus', do not specifically mention that you have replaced 'Hello' with 'Servus'.",
            }
        )
    assert result.content == "Servus, World!"
