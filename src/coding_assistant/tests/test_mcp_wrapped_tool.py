import pytest
from unittest.mock import MagicMock, AsyncMock
from coding_assistant.tools.mcp import MCPWrappedTool, get_mcp_wrapped_tools, MCPServer


@pytest.mark.asyncio
async def test_mcp_wrapped_tool_name_without_prefix():
    # Setup
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "my_tool"

    wrapped = MCPWrappedTool(client=mock_client, server_name="test_server", tool=mock_tool, prefix=None)

    # Assert
    assert wrapped.name() == "my_tool"


@pytest.mark.asyncio
async def test_mcp_wrapped_tool_name_with_prefix():
    # Setup
    mock_client = MagicMock()
    mock_tool = MagicMock()
    mock_tool.name = "my_tool"

    wrapped = MCPWrappedTool(client=mock_client, server_name="test_server", tool=mock_tool, prefix="pre_")

    # Assert
    assert wrapped.name() == "pre_my_tool"


@pytest.mark.asyncio
async def test_get_mcp_wrapped_tools_applies_prefix():
    # Setup
    mock_client = MagicMock(spec=["list_tools"])
    mock_tool = MagicMock()
    mock_tool.name = "tool1"

    mock_client.list_tools = AsyncMock(return_value=[mock_tool])

    servers = [MCPServer(name="server1", client=mock_client, instructions=None, prefix="s1_")]

    # Execute
    tools = await get_mcp_wrapped_tools(servers)

    # Assert
    assert len(tools) == 1
    assert tools[0].name() == "s1_tool1"


@pytest.mark.asyncio
async def test_get_mcp_wrapped_tools_collision_detection():
    # Setup
    mock_client1 = MagicMock(spec=["list_tools"])
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool"
    mock_client1.list_tools = AsyncMock(return_value=[mock_tool1])

    mock_client2 = MagicMock(spec=["list_tools"])
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool"
    mock_client2.list_tools = AsyncMock(return_value=[mock_tool2])

    # Two servers with no prefixes and same tool name should collide
    servers = [
        MCPServer(name="server1", client=mock_client1, instructions=None, prefix=None),
        MCPServer(name="server2", client=mock_client2, instructions=None, prefix=None),
    ]

    # Execute & Assert
    with pytest.raises(ValueError, match="MCP tool name collision detected: tool"):
        await get_mcp_wrapped_tools(servers)


@pytest.mark.asyncio
async def test_get_mcp_wrapped_tools_prefix_prevents_collision():
    # Setup
    mock_client1 = MagicMock(spec=["list_tools"])
    mock_tool1 = MagicMock()
    mock_tool1.name = "tool"
    mock_client1.list_tools = AsyncMock(return_value=[mock_tool1])

    mock_client2 = MagicMock(spec=["list_tools"])
    mock_tool2 = MagicMock()
    mock_tool2.name = "tool"
    mock_client2.list_tools = AsyncMock(return_value=[mock_tool2])

    # Same tool names but different prefixes
    servers = [
        MCPServer(name="server1", client=mock_client1, instructions=None, prefix="a_"),
        MCPServer(name="server2", client=mock_client2, instructions=None, prefix="b_"),
    ]

    # Execute
    tools = await get_mcp_wrapped_tools(servers)

    # Assert
    assert len(tools) == 2
    names = {t.name() for t in tools}
    assert names == {"a_tool", "b_tool"}
