import pytest
import asyncio
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from coding_assistant.session import Session, stop_mcp_server
from coding_assistant.config import Config
from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.llm.types import StatusLevel, ProgressCallbacks
from coding_assistant.framework.types import Tool
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
    config = Session.get_default_mcp_server_config(root, skills, mcp_url="http://localhost:1234/mcp")

    assert config.name == "coding_assistant.mcp"
    assert config.args is not None
    assert "--skills-directories" in config.args
    assert "skill1" in config.args
    assert "skill2" in config.args
    assert "--mcp-url" in config.args
    assert "http://localhost:1234/mcp" in config.args


@pytest.mark.asyncio
async def test_stop_mcp_server() -> None:
    callbacks = MagicMock(spec=ProgressCallbacks)
    mcp_task: asyncio.Future[Any] = asyncio.Future()
    mcp_task.set_result(None)
    mcp_task.cancel = MagicMock()  # type: ignore

    await stop_mcp_server(mcp_task, callbacks)  # type: ignore

    mcp_task.cancel.assert_called_once()
    callbacks.on_status_message.assert_called_with("Shutting down external MCP server...", level=StatusLevel.INFO)


@pytest.mark.asyncio
async def test_stop_mcp_server_cancelled() -> None:
    callbacks = MagicMock(spec=ProgressCallbacks)
    mcp_task: asyncio.Future[Any] = asyncio.Future()
    mcp_task.set_exception(asyncio.CancelledError())
    mcp_task.cancel = MagicMock()  # type: ignore

    await stop_mcp_server(mcp_task, callbacks)  # type: ignore
    assert mcp_task.done()


@pytest.mark.asyncio
async def test_session_context_manager(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)

    with (
        patch("coding_assistant.session.sandbox") as mock_sandbox,
        patch("coding_assistant.session.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.session.get_mcp_wrapped_tools", new_callable=AsyncMock) as mock_get_tools,
        patch("coding_assistant.session.get_instructions") as mock_get_instructions,
    ):
        # Setup mocks
        mock_mcp_cm = AsyncMock()
        mock_mcp_cm.__aenter__.return_value = ["mock_server"]
        mock_get_mcp.return_value = mock_mcp_cm
        mock_tool = MagicMock(spec=Tool)
        mock_get_tools.return_value = [mock_tool]
        mock_get_instructions.return_value = "test instructions"

        async with session:
            assert session.tools == [mock_tool]
            assert session.instructions == "test instructions"
            assert session.mcp_servers == ["mock_server"]
            mock_sandbox.assert_called_once()
            mock_get_mcp.assert_called_once()

        # Verify exit calls
        mock_mcp_cm.__aexit__.assert_called_once()
        session_args["callbacks"].on_status_message.assert_any_call("Session closed.", level=StatusLevel.INFO)


@pytest.mark.asyncio
async def test_session_mcp_server_port(session_args: dict[str, Any]) -> None:
    session_args["mcp_server_port"] = 8000
    session = Session(**session_args)

    with (
        patch("coding_assistant.session.sandbox"),
        patch("coding_assistant.session.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.session.get_mcp_wrapped_tools", new_callable=AsyncMock),
        patch("coding_assistant.session.start_mcp_server", new_callable=AsyncMock) as mock_start_server,
        patch("coding_assistant.session.get_instructions"),
    ):
        mock_mcp_cm = AsyncMock()
        mock_mcp_cm.__aenter__.return_value = ["mock_server"]
        mock_get_mcp.return_value = mock_mcp_cm

        # Return a future that is already done to avoid issues in stop_mcp_server
        mock_task: asyncio.Future[Any] = asyncio.Future()
        mock_task.set_result(None)
        mock_task.cancel = MagicMock()  # type: ignore
        mock_start_server.return_value = mock_task

        async with session:
            mock_start_server.assert_called_once()


@pytest.mark.asyncio
async def test_session_run_chat(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    mock_tool = MagicMock(spec=Tool)
    session.tools = [mock_tool]
    session.instructions = "test instructions"

    with (
        patch("coding_assistant.session.run_chat_loop", new_callable=AsyncMock) as mock_chat_loop,
        patch("coding_assistant.session.save_orchestrator_history") as mock_save,
    ):
        await session.run_chat(history=[])

        mock_chat_loop.assert_called_once()
        mock_save.assert_called_once_with(session.working_directory, [])


@pytest.mark.asyncio
async def test_session_run_agent(session_args: dict[str, Any]) -> None:
    session = Session(**session_args)
    mock_tool = MagicMock(spec=Tool)
    session.tools = [mock_tool]
    session.instructions = "test instructions"

    with (
        patch("coding_assistant.session.AgentTool") as mock_agent_tool_class,
        patch("coding_assistant.session.save_orchestrator_history") as mock_save,
    ):
        mock_tool_instance = mock_agent_tool_class.return_value
        mock_tool_instance.execute = AsyncMock(return_value=MagicMock(content="result"))
        mock_tool_instance.history = ["msg1"]

        result = await session.run_agent(task="test task")

        assert result.content == "result"
        mock_tool_instance.execute.assert_called_once()
        mock_save.assert_called_once_with(session.working_directory, ["msg1"])
