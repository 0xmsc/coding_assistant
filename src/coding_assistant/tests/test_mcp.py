from typing import cast
import pytest
from unittest.mock import MagicMock

from coding_assistant.tools.mcp import MCPServer, get_default_env, handle_mcp_tool_call


class _FakeClient:
    def __init__(self, name: str, responses: dict[str, str]):
        self._name = name
        self._responses = responses

    async def call_tool(self, tool_name: str, arguments: dict):
        if tool_name not in self._responses:
            return None  # Or simulate what FastMCP does for empty
        return self._responses[tool_name]


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_happy_path():
    client = _FakeClient(
        name="server1",
        responses={"echo": "hello"},
    )
    servers = [MCPServer(name="server1", client=cast(any, client), instructions=None)]

    content = await handle_mcp_tool_call("mcp_server1_echo", {"msg": "ignored"}, servers)
    assert content == "hello"


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_no_content_returns_message():
    client = _FakeClient(name="server1", responses={"empty": "None"})
    servers = [MCPServer(name="server1", client=cast(any, client), instructions=None)]

    content = await handle_mcp_tool_call("mcp_server1_empty", {}, servers)
    assert content == "None"


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_server_not_found():
    servers = [MCPServer(name="serverA", client=MagicMock(), instructions=None)]
    with pytest.raises(RuntimeError, match="Server serverB not found"):
        await handle_mcp_tool_call("mcp_serverB_echo", {}, servers)


def test_get_default_env_includes_https_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    env = get_default_env()
    assert env.get("HTTPS_PROXY") == "http://proxy:8080"
