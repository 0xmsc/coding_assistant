import pytest
from typing import Any
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch
from coding_assistant.session import Session
from coding_assistant.config import Config
from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.common.messages import RunAgentRequest, RunChatRequest, RunCompleted
from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.framework.types import AgentOutput
from coding_assistant.llm.types import StatusLevel, ProgressCallbacks, Tool, ToolResult
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
        patch("coding_assistant.session.MCPServerManagerActor") as mock_manager_class,
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
    class _NoopTool(Tool):
        def name(self) -> str:
            return "noop"

        def description(self) -> str:
            return "noop"

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> ToolResult:
            raise RuntimeError("not expected")

    class _ChatActor:
        def __init__(self, actor_directory: ActorDirectory) -> None:
            self.actor_directory = actor_directory
            self.received: RunChatRequest | None = None

        async def send_message(self, message: object) -> None:
            assert isinstance(message, RunChatRequest)
            self.received = message
            await self.actor_directory.send_message(
                uri=message.reply_to_uri or "",
                message=RunCompleted(request_id=message.request_id),
            )

    session = Session(**session_args)
    actor_directory = ActorDirectory()
    chat_uri = "actor://test/chat"
    chat_actor = _ChatActor(actor_directory)
    actor_directory.register(uri=chat_uri, actor=chat_actor)
    session.tools = [_NoopTool()]
    session.instructions = "test instructions"
    session._actor_directory = actor_directory
    session._chat_actor_uri = chat_uri
    session._tool_call_actor_uri = "actor://test/tool-call"
    session._user_actor = MagicMock(spec=UI)
    session._user_actor_uri = "actor://test/user"
    history: list[Any] = []

    with patch("coding_assistant.session.history_manager_scope") as mock_history_scope:
        mock_history_manager = MagicMock()
        mock_history_manager.save_orchestrator_history = AsyncMock()

        @asynccontextmanager
        async def history_scope(*, context_name: str) -> AsyncIterator[MagicMock]:
            yield mock_history_manager

        mock_history_scope.side_effect = history_scope

        await session.run_chat(history=history)

        assert chat_actor.received is not None
        assert chat_actor.received.reply_to_uri is not None
        assert chat_actor.received.tool_call_actor_uri == session._tool_call_actor_uri
        assert chat_actor.received.user_actor_uri == session._user_actor_uri
        assert chat_actor.received.instructions == "test instructions"
        mock_history_manager.save_orchestrator_history.assert_called_once_with(
            working_directory=session.working_directory, history=history
        )


@pytest.mark.asyncio
async def test_session_run_agent(session_args: dict[str, Any]) -> None:
    class _NoopTool(Tool):
        def name(self) -> str:
            return "noop"

        def description(self) -> str:
            return "noop"

        def parameters(self) -> dict[str, Any]:
            return {}

        async def execute(self, parameters: dict[str, Any]) -> ToolResult:
            raise RuntimeError("not expected")

    class _AgentActor:
        def __init__(self, actor_directory: ActorDirectory) -> None:
            self.actor_directory = actor_directory
            self.received: RunAgentRequest | None = None

        async def send_message(self, message: object) -> None:
            assert isinstance(message, RunAgentRequest)
            self.received = message
            message.ctx.state.output = AgentOutput(result="result", summary="summary")
            await self.actor_directory.send_message(
                uri=message.reply_to_uri or "",
                message=RunCompleted(request_id=message.request_id),
            )

    session = Session(**session_args)
    actor_directory = ActorDirectory()
    agent_uri = "actor://test/agent"
    agent_actor = _AgentActor(actor_directory)
    actor_directory.register(uri=agent_uri, actor=agent_actor)
    session.tools = [_NoopTool()]
    session.instructions = "test instructions"
    session._actor_directory = actor_directory
    session._agent_actor_uri = agent_uri
    session._tool_call_actor_uri = "actor://test/tool-call"
    session._user_actor = MagicMock(spec=UI)
    session._user_actor_uri = "actor://test/user"

    with patch("coding_assistant.session.history_manager_scope") as mock_history_scope:
        mock_history_manager = MagicMock()
        mock_history_manager.save_orchestrator_history = AsyncMock()

        @asynccontextmanager
        async def history_scope(*, context_name: str) -> AsyncIterator[MagicMock]:
            yield mock_history_manager

        mock_history_scope.side_effect = history_scope

        result = await session.run_agent(task="test task")

        assert result.content == "result"
        assert agent_actor.received is not None
        assert agent_actor.received.reply_to_uri is not None
        assert agent_actor.received.tool_call_actor_uri == session._tool_call_actor_uri
        saved_history = mock_history_manager.save_orchestrator_history.call_args.kwargs["history"]
        assert isinstance(saved_history, list)
        mock_history_manager.save_orchestrator_history.assert_called_once_with(
            working_directory=session.working_directory, history=saved_history
        )
