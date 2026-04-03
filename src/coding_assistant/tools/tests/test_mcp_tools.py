from unittest.mock import AsyncMock, MagicMock

import pytest

from coding_assistant.tools.mcp_manager import MCPServerManager
from coding_assistant.tools.mcp_tools import (
    MCPCallTool,
    MCPListToolsTool,
    MCPStartTool,
    MCPStopTool,
    create_mcp_tools,
)


class TestMCPStartTool:
    """Tests for MCPStartTool."""

    def test_name(self) -> None:
        """Tool has correct name."""
        manager = MagicMock(spec=MCPServerManager)
        tool = MCPStartTool(manager)
        assert tool.name() == "mcp_start"

    def test_description_includes_available_servers(self) -> None:
        """Description shows available servers."""
        manager = MagicMock()
        manager.available_servers = ["server1", "server2"]
        tool = MCPStartTool(manager)
        assert "server1" in tool.description()
        assert "server2" in tool.description()

    def test_description_empty_when_no_servers(self) -> None:
        """Description shows 'none' when no servers configured."""
        manager = MagicMock()
        manager.available_servers = []
        tool = MCPStartTool(manager)
        assert "none" in tool.description()

    def test_parameters_has_enum(self) -> None:
        """Parameters include enum of available servers."""
        manager = MagicMock()
        manager.available_servers = ["s1", "s2"]
        tool = MCPStartTool(manager)
        params = tool.parameters()
        assert params["properties"]["server"]["enum"] == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_execute_calls_manager_start(self) -> None:
        """Execute delegates to manager.start()."""
        manager = MagicMock(spec=MCPServerManager)
        manager.start = AsyncMock(return_value="Started")
        tool = MCPStartTool(manager)
        result = await tool.execute({"server": "test"})
        manager.start.assert_called_once_with("test")
        assert result == "Started"


class TestMCPStopTool:
    """Tests for MCPStopTool."""

    def test_name(self) -> None:
        """Tool has correct name."""
        manager = MagicMock(spec=MCPServerManager)
        tool = MCPStopTool(manager)
        assert tool.name() == "mcp_stop"

    @pytest.mark.asyncio
    async def test_execute_calls_manager_stop(self) -> None:
        """Execute delegates to manager.stop()."""
        manager = MagicMock(spec=MCPServerManager)
        manager.stop = AsyncMock(return_value="Stopped")
        tool = MCPStopTool(manager)
        result = await tool.execute({"server": "test"})
        manager.stop.assert_called_once_with("test")
        assert result == "Stopped"


class TestMCPCallTool:
    """Tests for MCPCallTool."""

    def test_name(self) -> None:
        """Tool has correct name."""
        manager = MagicMock(spec=MCPServerManager)
        tool = MCPCallTool(manager)
        assert tool.name() == "mcp_call"

    def test_parameters_includes_all_fields(self) -> None:
        """Parameters include server, tool, and arguments."""
        tool = MCPCallTool(MagicMock())
        params = tool.parameters()
        assert "server" in params["properties"]
        assert "tool" in params["properties"]
        assert "arguments" in params["properties"]

    @pytest.mark.asyncio
    async def test_execute_calls_manager_call(self) -> None:
        """Execute delegates to manager.call()."""
        manager = MagicMock(spec=MCPServerManager)
        manager.call = AsyncMock(return_value="result")
        tool = MCPCallTool(manager)
        result = await tool.execute(
            {
                "server": "myserver",
                "tool": "mytool",
                "arguments": {"arg1": "value1"},
            }
        )
        manager.call.assert_called_once_with("myserver", "mytool", {"arg1": "value1"})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_handles_missing_arguments(self) -> None:
        """Execute handles missing arguments gracefully."""
        manager = MagicMock(spec=MCPServerManager)
        manager.call = AsyncMock(return_value="result")
        tool = MCPCallTool(manager)
        await tool.execute({"server": "s", "tool": "t"})
        manager.call.assert_called_once_with("s", "t", {})


class TestMCPListToolsTool:
    """Tests for MCPListToolsTool."""

    def test_name(self) -> None:
        """Tool has correct name."""
        manager = MagicMock(spec=MCPServerManager)
        tool = MCPListToolsTool(manager)
        assert tool.name() == "mcp_list_tools"

    @pytest.mark.asyncio
    async def test_execute_calls_manager_list_tools(self) -> None:
        """Execute delegates to manager.list_tools()."""
        manager = MagicMock(spec=MCPServerManager)
        manager.list_tools = AsyncMock(return_value="Tools:\n- tool1")
        tool = MCPListToolsTool(manager)
        result = await tool.execute({"server": "test"})
        manager.list_tools.assert_called_once_with("test")
        assert result == "Tools:\n- tool1"


class TestCreateMCPTools:
    """Tests for create_mcp_tools factory."""

    def test_returns_all_four_tools(self) -> None:
        """Factory returns all MCP management tools."""
        manager = MagicMock(spec=MCPServerManager)
        tools = create_mcp_tools(manager)
        assert len(tools) == 4
        names = {t.name() for t in tools}
        assert names == {"mcp_start", "mcp_stop", "mcp_call", "mcp_list_tools"}
