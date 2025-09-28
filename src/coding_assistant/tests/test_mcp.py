from typing import cast
import pytest

from mcp import ClientSession
from coding_assistant.tools.mcp import MCPServer, get_default_env, handle_mcp_tool_call


class _ResultContent:
    def __init__(self, text: str):
        self.text = text


class _CallToolResult:
    def __init__(self, content: list[_ResultContent] | None):
        self.content = content


class _FakeSession:
    def __init__(self, name: str, responses: dict[str, _CallToolResult]):
        self._name = name
        self._responses = responses

    async def call_tool(self, tool_name: str, arguments: dict):
        return self._responses.get(tool_name, _CallToolResult(content=None))


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_happy_path():
    session = _FakeSession(
        name="server1",
        responses={"echo": _CallToolResult([_ResultContent("hello")])},
    )
    servers = [MCPServer(name="server1", session=cast(ClientSession, session), instructions=None)]

    content = await handle_mcp_tool_call("mcp_server1_echo", {"msg": "ignored"}, servers)
    assert content == "hello"


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_no_content_returns_message():
    session = _FakeSession(name="server1", responses={"empty": _CallToolResult(content=None)})
    servers = [MCPServer(name="server1", session=cast(ClientSession, session), instructions=None)]

    content = await handle_mcp_tool_call("mcp_server1_empty", {}, servers)
    assert content == "MCP server did not return any content."


@pytest.mark.asyncio
async def test_handle_mcp_tool_call_server_not_found():
    servers = [MCPServer(name="serverA", session=cast(ClientSession, _FakeSession("serverA", {})), instructions=None)]
    with pytest.raises(RuntimeError, match="Server serverB not found"):
        await handle_mcp_tool_call("mcp_serverB_echo", {}, servers)


def test_get_default_env_includes_https_proxy(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    env = get_default_env()
    assert env.get("HTTPS_PROXY") == "http://proxy:8080"
