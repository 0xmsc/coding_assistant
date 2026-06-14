from pathlib import Path
from types import TracebackType
from typing import Any, ClassVar

import pytest

from coding_assistant.tools.mcp_manager import MCPServerConfig, MCPServerManager


class FakeTool:
    """Minimal MCP tool object for manager tests."""

    def __init__(self, *, name: str, description: str | None = None) -> None:
        self.name = name
        self.description = description


class FakeTextContent:
    """Minimal text content object for MCP call results."""

    def __init__(self, text: str) -> None:
        self.text = text


class FakeCallResult:
    """Minimal MCP call result object."""

    def __init__(self, content: list[FakeTextContent]) -> None:
        self.content = content


class FakeClient:
    """Fake FastMCP client that errors if used outside its async context."""

    instances: ClassVar[list["FakeClient"]] = []
    fail_list_tools: ClassVar[bool] = False

    def __init__(self, transport: object, name: str) -> None:
        del transport
        self.name = name
        self.connected = False
        self.enter_count = 0
        self.exit_count = 0
        self.initialize_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakeClient.instances.append(self)

    async def __aenter__(self) -> "FakeClient":
        self.connected = True
        self.enter_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.connected = False
        self.exit_count += 1

    async def initialize(self) -> None:
        self._require_connected()
        self.initialize_count += 1

    async def list_tools(self) -> list[FakeTool]:
        self._require_connected()
        if FakeClient.fail_list_tools:
            raise RuntimeError("list failed")
        return [
            FakeTool(name="lookup", description="Finds things"),
            FakeTool(name="empty_description"),
        ]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> FakeCallResult:
        self._require_connected()
        self.calls.append((tool_name, arguments))
        return FakeCallResult(content=[FakeTextContent(f"called {tool_name}")])

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("client disconnected")


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> type[FakeClient]:
    """Patch MCP manager to use a fake client."""
    FakeClient.instances.clear()
    FakeClient.fail_list_tools = False
    monkeypatch.setattr("coding_assistant.tools.mcp_manager.Client", FakeClient)
    return FakeClient


def test_mcp_server_config_validation_command_only() -> None:
    """Config with only command is valid."""
    config = MCPServerConfig(name="test", command="ls")
    assert config.name == "test"
    assert config.command == "ls"


def test_mcp_server_config_validation_url_only() -> None:
    """Config with only URL is valid."""
    config = MCPServerConfig(name="test", url="http://localhost:8000")
    assert config.name == "test"
    assert config.url == "http://localhost:8000"


def test_mcp_server_config_validation_neither() -> None:
    """Config with neither command nor URL is invalid."""
    with pytest.raises(ValueError, match="must have either a command or a url"):
        MCPServerConfig(name="test")


def test_mcp_server_config_validation_both() -> None:
    """Config with both command and URL is invalid."""
    with pytest.raises(ValueError, match="cannot have both a command and a url"):
        MCPServerConfig(name="test", command="ls", url="http://localhost")


class TestMCPServerManager:
    """Tests for MCPServerManager."""

    def test_available_servers_empty(self) -> None:
        """Manager with no configs has no available servers."""
        manager = MCPServerManager(configs=[], working_directory=Path("/tmp"))
        assert manager.available_servers == []

    def test_available_servers_with_configs(self) -> None:
        """Manager exposes configured server names."""
        configs = [
            MCPServerConfig(name="server1", command="cmd1"),
            MCPServerConfig(name="server2", url="http://example.com"),
        ]
        manager = MCPServerManager(configs=configs, working_directory=Path("/tmp"))
        assert manager.available_servers == ["server1", "server2"]

    def test_running_servers_empty_initially(self) -> None:
        """No servers running at start."""
        configs = [MCPServerConfig(name="test", command="ls")]
        manager = MCPServerManager(configs=configs, working_directory=Path("/tmp"))
        assert manager.running_servers == []

    @pytest.mark.asyncio
    async def test_start_unknown_server(self) -> None:
        """Starting unknown server returns error."""
        manager = MCPServerManager(configs=[], working_directory=Path("/tmp"))
        result = await manager.start("unknown")
        assert "Unknown server 'unknown'" in result
        assert "Available:" in result

    @pytest.mark.asyncio
    async def test_stop_not_running_server(self) -> None:
        """Stopping server that's not running returns error."""
        configs = [MCPServerConfig(name="test", command="ls")]
        manager = MCPServerManager(configs=configs, working_directory=Path("/tmp"))
        result = await manager.stop("test")
        assert "is not running" in result

    @pytest.mark.asyncio
    async def test_list_tools_not_running(self) -> None:
        """Listing tools for non-running server returns error."""
        configs = [MCPServerConfig(name="test", command="ls")]
        manager = MCPServerManager(configs=configs, working_directory=Path("/tmp"))
        result = await manager.list_tools("test")
        assert "is not running" in result
        assert "Use mcp_start" in result

    @pytest.mark.asyncio
    async def test_call_not_running(self) -> None:
        """Calling tool on non-running server returns error."""
        configs = [MCPServerConfig(name="test", command="ls")]
        manager = MCPServerManager(configs=configs, working_directory=Path("/tmp"))
        result = await manager.call("test", "some_tool", {})
        assert "is not running" in result

    @pytest.mark.asyncio
    async def test_start_keeps_client_context_open_for_calls(
        self,
        tmp_path: Path,
        fake_client: type[FakeClient],
    ) -> None:
        """Started clients stay connected for later tool calls."""
        configs = [MCPServerConfig(name="test", command="cmd")]
        manager = MCPServerManager(configs=configs, working_directory=tmp_path)

        start_result = await manager.start("test")
        assert start_result == "Started 'test' with 2 tools: lookup, empty_description"

        client = fake_client.instances[0]
        assert client.connected
        assert client.enter_count == 1
        assert client.exit_count == 0

        tools_result = await manager.list_tools("test")
        assert "lookup: Finds things" in tools_result
        assert "empty_description: (no description)" in tools_result

        call_result = await manager.call("test", "lookup", {"query": "abc"})
        assert call_result == "called lookup"
        assert client.calls == [("lookup", {"query": "abc"})]

    @pytest.mark.asyncio
    async def test_stop_closes_running_client_context(
        self,
        tmp_path: Path,
        fake_client: type[FakeClient],
    ) -> None:
        """Stopping a server exits the stored client context."""
        configs = [MCPServerConfig(name="test", command="cmd")]
        manager = MCPServerManager(configs=configs, working_directory=tmp_path)

        await manager.start("test")
        client = fake_client.instances[0]

        stop_result = await manager.stop("test")
        assert stop_result == "Stopped 'test'."
        assert manager.running_servers == []
        assert not client.connected
        assert client.exit_count == 1

    @pytest.mark.asyncio
    async def test_start_failure_closes_entered_client_context(
        self,
        tmp_path: Path,
        fake_client: type[FakeClient],
    ) -> None:
        """A failed start does not leave an entered client context running."""
        fake_client.fail_list_tools = True
        configs = [MCPServerConfig(name="test", command="cmd")]
        manager = MCPServerManager(configs=configs, working_directory=tmp_path)

        result = await manager.start("test")

        client = fake_client.instances[0]
        assert result == "Failed to start 'test': list failed"
        assert manager.running_servers == []
        assert not client.connected
        assert client.exit_count == 1
