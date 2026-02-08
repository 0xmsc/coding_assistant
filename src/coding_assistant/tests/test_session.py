import pytest
from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch
from coding_assistant.session import Session
from coding_assistant.config import Config
from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.llm.types import StatusLevel, ProgressCallbacks, Tool
from coding_assistant.tools.mcp import MCPServer
from coding_assistant.tools.mcp_manager import MCPServerBundle
from coding_assistant.tools.tools import RedirectToolCallTool
from coding_assistant.ui import UI


@pytest.fixture
def mock_config() -> Config:
    return Config(model="test-model", expert_model="test-expert", compact_conversation_at_tokens=200000)


@pytest.fixture
def mock_ui() -> MagicMock:
    return MagicMock(spec=UI)


@pytest.fixture
def mock_callbacks() -> MagicMock:
    return MagicMock(spec=ProgressCallbacks)


@pytest.fixture
def mock_tool_callbacks() -> MagicMock:
    return MagicMock(spec=ToolCallbacks)


@pytest.fixture
def session_args(
    mock_config: Config, mock_ui: MagicMock, mock_callbacks: MagicMock, mock_tool_callbacks: MagicMock
) -> dict[str, Any]:
    return {
        "config": mock_config,
        "ui": mock_ui,
        "callbacks": mock_callbacks,
        "tool_callbacks": mock_tool_callbacks,
        "working_directory": Path("/tmp/work"),
        "coding_assistant_root": Path("/tmp/root"),
        "mcp_server_configs": [],
    }


def test_get_default_mcp_server_config() -> None:
    root = Path("/root")
    skills = ["skill1", "skill2"]
    config = Session.get_default_mcp_server_config(root, skills)

    assert config.name == "coding_assistant.mcp"
    assert config.args is not None
    assert "--skills-directories" in config.args
    assert "skill1" in config.args
    assert "skill2" in config.args


@pytest.mark.asyncio
async def test_session_context_manager(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)

    with (
        patch("coding_assistant.session.sandbox") as mock_sandbox,
        patch("coding_assistant.session.MCPServerManager") as mock_manager_class,
        patch("coding_assistant.session.get_instructions") as mock_get_instructions,
    ):
        mock_manager = mock_manager_class.return_value
        mock_tool = MagicMock(spec=Tool)
        mock_server = MagicMock(spec=MCPServer)
        mock_manager.initialize = AsyncMock(return_value=MCPServerBundle(servers=[mock_server], tools=[mock_tool]))
        mock_manager.shutdown = AsyncMock()
        mock_get_instructions.return_value = "test instructions"

        async with session:
            assert len(session.tools) == 2
            assert session.tools[0] == mock_tool
            # The second tool should be RedirectToolCallTool
            assert isinstance(session.tools[1], RedirectToolCallTool)
            assert session.instructions == "test instructions"
            assert session.mcp_servers == [mock_server]
            mock_sandbox.assert_called_once()
            mock_manager.start.assert_called_once()

        # Verify exit calls
        mock_manager.shutdown.assert_called_once()
        session_args["callbacks"].on_status_message.assert_any_call("Session closed.", level=StatusLevel.INFO)


@pytest.mark.asyncio
async def test_session_run_chat(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    mock_tool = MagicMock(spec=Tool)
    session.tools = [mock_tool]
    session.instructions = "test instructions"

    with (
        patch("coding_assistant.session.run_chat_loop", new_callable=AsyncMock) as mock_chat_loop,
        patch("coding_assistant.session.history_manager_scope") as mock_history_scope,
    ):
        mock_history_manager = MagicMock()
        mock_history_manager.save_orchestrator_history = AsyncMock()

        @asynccontextmanager
        async def history_scope(*, context_name: str) -> AsyncIterator[MagicMock]:
            yield mock_history_manager

        mock_history_scope.side_effect = history_scope

        await session.run_chat(history=[])

        mock_chat_loop.assert_called_once()
        mock_history_manager.save_orchestrator_history.assert_called_once_with(
            working_directory=session.working_directory, history=[]
        )


@pytest.mark.asyncio
async def test_session_run_agent(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    mock_tool = MagicMock(spec=Tool)
    session.tools = [mock_tool]
    session.instructions = "test instructions"

    with (
        patch("coding_assistant.session.AgentTool") as mock_agent_tool_class,
        patch("coding_assistant.session.history_manager_scope") as mock_history_scope,
    ):
        mock_history_manager = MagicMock()
        mock_history_manager.save_orchestrator_history = AsyncMock()

        @asynccontextmanager
        async def history_scope(*, context_name: str) -> AsyncIterator[MagicMock]:
            yield mock_history_manager

        mock_history_scope.side_effect = history_scope

        mock_tool_instance = mock_agent_tool_class.return_value
        mock_tool_instance.execute = AsyncMock(return_value=MagicMock(content="result"))
        mock_tool_instance.history = ["msg1"]

        result = await session.run_agent(task="test task")

        assert result.content == "result"
        mock_tool_instance.execute.assert_called_once()
        mock_history_manager.save_orchestrator_history.assert_called_once_with(
            working_directory=session.working_directory, history=["msg1"]
        )
