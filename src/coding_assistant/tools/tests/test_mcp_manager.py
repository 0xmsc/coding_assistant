from pathlib import Path

import pytest

from coding_assistant.tools.mcp_manager import MCPServerConfig, MCPServerManager


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
