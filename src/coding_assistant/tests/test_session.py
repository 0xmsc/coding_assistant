import pytest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from coding_assistant.config import Config
from coding_assistant.session import Session
from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.llm.types import ProgressCallbacks, Tool
from coding_assistant.framework.types import AgentOutput


@pytest.fixture
def session_args() -> dict[str, Any]:
    mock_config = Config(
        model="test-model",
        expert_model="test-expert",
        compact_conversation_at_tokens=200000,
        enable_chat_mode=True,
        enable_ask_user=True,
    )
    mock_callbacks = MagicMock(spec=ProgressCallbacks)
    mock_tool_callbacks = MagicMock(spec=ToolCallbacks)
    mock_ui = MagicMock()

    # Ensure mock_ui methods are awaitable for chat loop
    async def mock_prompt(*args: Any, **kwargs: Any) -> str:
        return "/exit"

    mock_ui.prompt.side_effect = mock_prompt

    return {
        "config": mock_config,
        "ui": mock_ui,
        "callbacks": mock_callbacks,
        "tool_callbacks": mock_tool_callbacks,
        "working_directory": Path("/tmp/work"),
        "coding_assistant_root": Path("/tmp/root"),
        "mcp_server_configs": [],
    }


@pytest.mark.asyncio
async def test_session_init(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    assert session.config.model == "test-model"
    assert session.working_directory == Path("/tmp/work")


@pytest.mark.asyncio
async def test_session_run_agent(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    mock_tool = MagicMock(spec=Tool)
    session.tools = [mock_tool]
    session.instructions = "test instructions"

    # Patch all possible entry points for run_agent_loop
    with (
        patch("coding_assistant.framework.agent.run_agent_loop") as mock_run,
        patch("coding_assistant.session.save_orchestrator_history") as mock_save,
    ):

        async def mock_impl(ctx: Any, **kwargs: Any) -> None:
            ctx.state.output = AgentOutput(result="test result", summary="test summary")

        mock_run.side_effect = mock_impl

        # Patch local name in session module and tools module
        import coding_assistant.session as s_mod
        import coding_assistant.tools.tools as t_mod

        setattr(s_mod, "run_agent_loop", mock_run)
        setattr(t_mod, "run_agent_loop", mock_run)

        result = await session.run_agent("test task")

        assert result is not None
        # AgentTool.execute returns a TextResult containing the result string
        from coding_assistant.framework.results import TextResult

        assert isinstance(result, TextResult)
        assert result.content == "test result"
        assert mock_run.called
        assert mock_save.called


@pytest.mark.asyncio
async def test_session_run_chat(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    session.tools = []
    session.instructions = "test instructions"

    with (
        patch("coding_assistant.framework.chat.run_chat_loop") as mock_run,
        patch("coding_assistant.session.save_orchestrator_history") as mock_save,
    ):
        import coding_assistant.session as s_mod

        setattr(s_mod, "run_chat_loop", mock_run)

        await session.run_chat()
        assert mock_run.called
        assert mock_save.called
