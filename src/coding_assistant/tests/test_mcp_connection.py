import pytest
from pydantic import ValidationError
from pathlib import Path
from unittest.mock import patch, MagicMock
from coding_assistant.config import MCPServerConfig
from coding_assistant.tools.mcp import get_mcp_servers_from_config
from fastmcp.mcp_config import StdioMCPServer, RemoteMCPServer


@pytest.mark.asyncio
async def test_get_mcp_servers_from_config_stdio():
    config = [MCPServerConfig(name="test-stdio", command="test-cmd", args=["--arg1"])]
    working_dir = Path("/tmp")

    with patch("coding_assistant.tools.mcp._get_mcp_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.name = "test-stdio"

        mock_get_server.return_value.__aenter__.return_value = mock_server

        async with get_mcp_servers_from_config(config, working_dir) as servers:
            assert len(servers) == 1
            assert servers[0].name == "test-stdio"
            mock_get_server.assert_called_once()
            args, kwargs = mock_get_server.call_args
            config_arg = kwargs["config"]
            assert isinstance(config_arg, StdioMCPServer)
            assert config_arg.command == "test-cmd"
            assert config_arg.args == ["--arg1"]


@pytest.mark.asyncio
async def test_get_mcp_servers_from_config_sse():
    config = [MCPServerConfig(name="test-sse", url="http://localhost:8000/sse")]
    working_dir = Path("/tmp")

    with patch("coding_assistant.tools.mcp._get_mcp_server") as mock_get_server:
        mock_server = MagicMock()
        mock_server.name = "test-sse"
        mock_get_server.return_value.__aenter__.return_value = mock_server

        async with get_mcp_servers_from_config(config, working_dir) as servers:
            assert len(servers) == 1
            assert servers[0].name == "test-sse"
            mock_get_server.assert_called_once()
            args, kwargs = mock_get_server.call_args
            config_arg = kwargs["config"]
            assert isinstance(config_arg, RemoteMCPServer)
            assert config_arg.url == "http://localhost:8000/sse"


@pytest.mark.asyncio
async def test_mcp_server_config_validation():
    with pytest.raises(ValidationError, match="must have either a command or a url"):
        MCPServerConfig(name="invalid")

    with pytest.raises(ValidationError, match="cannot have both a command and a url"):
        MCPServerConfig(name="too-much", command="ls", url="http://localhost")
