from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coding_assistant.config import MCPServerConfig
from coding_assistant.llm.types import Tool
from coding_assistant.tools.mcp import MCPServer
from coding_assistant.tools.mcp_manager import MCPServerManager, MCPServerManagerActor


@pytest.mark.asyncio
async def test_mcp_manager_actor_initialize_and_shutdown() -> None:
    config = [MCPServerConfig(name="test", command="cmd", args=[])]
    manager = MCPServerManagerActor(context_name="test")
    mock_cm = AsyncMock()
    mock_server = MagicMock(spec=MCPServer)
    mock_cm.__aenter__.return_value = [mock_server]

    with (
        patch("coding_assistant.tools.mcp_manager.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.tools.mcp_manager.get_mcp_wrapped_tools", new_callable=AsyncMock) as mock_get_tools,
    ):
        mock_get_mcp.return_value = mock_cm
        mock_tool = MagicMock(spec=Tool)
        mock_get_tools.return_value = [mock_tool]

        manager.start()
        bundle = await manager.initialize(config_servers=config, working_directory=Path("/tmp"))
        assert bundle.servers == [mock_server]
        assert bundle.tools == [mock_tool]

        await manager.shutdown()
        mock_cm.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_manager_actor_cleanup_on_initialize_error() -> None:
    config = [MCPServerConfig(name="test", command="cmd", args=[])]
    manager = MCPServerManagerActor(context_name="test")
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = [MagicMock(spec=MCPServer)]

    with (
        patch("coding_assistant.tools.mcp_manager.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.tools.mcp_manager.get_mcp_wrapped_tools", new_callable=AsyncMock) as mock_get_tools,
    ):
        mock_get_mcp.return_value = mock_cm
        mock_get_tools.side_effect = RuntimeError("boom")

        manager.start()
        with pytest.raises(RuntimeError, match="boom"):
            await manager.initialize(config_servers=config, working_directory=Path("/tmp"))

        mock_cm.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_plain_mcp_manager_rejects_double_initialize() -> None:
    config = [MCPServerConfig(name="test", command="cmd", args=[])]
    manager = MCPServerManager()
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = [MagicMock(spec=MCPServer)]

    with (
        patch("coding_assistant.tools.mcp_manager.get_mcp_servers_from_config") as mock_get_mcp,
        patch("coding_assistant.tools.mcp_manager.get_mcp_wrapped_tools", new_callable=AsyncMock) as mock_get_tools,
    ):
        mock_get_mcp.return_value = mock_cm
        mock_tool = MagicMock(spec=Tool)
        mock_get_tools.return_value = [mock_tool]

        bundle = await manager.initialize(config_servers=config, working_directory=Path("/tmp"))
        assert bundle.tools == [mock_tool]

        with pytest.raises(RuntimeError, match="already initialized"):
            await manager.initialize(config_servers=config, working_directory=Path("/tmp"))
