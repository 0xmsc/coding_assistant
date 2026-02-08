from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coding_assistant.config import MCPServerConfig
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.llm.types import Tool
from coding_assistant.tools.mcp import MCPServer, get_mcp_servers_from_config, get_mcp_wrapped_tools


@dataclass(slots=True)
class MCPServerBundle:
    servers: list[MCPServer]
    tools: list[Tool]


@dataclass(slots=True)
class _Initialize:
    config_servers: list[MCPServerConfig]
    working_directory: Path


@dataclass(slots=True)
class _Shutdown:
    exc_type: type[BaseException] | None
    exc_val: BaseException | None
    exc_tb: Any


_Message = _Initialize | _Shutdown


class MCPServerManager:
    def __init__(self, *, context_name: str = "mcp-manager") -> None:
        self._context_name = context_name
        self._actor: Actor[_Message, MCPServerBundle | None] = Actor(
            name=f"{context_name}.mcp-manager", handler=self._handle_message
        )
        self._cm: Any | None = None
        self._servers: list[MCPServer] | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def initialize(self, *, config_servers: list[MCPServerConfig], working_directory: Path) -> MCPServerBundle:
        self.start()
        result = await self._actor.ask(_Initialize(config_servers=config_servers, working_directory=working_directory))
        if result is None:
            raise RuntimeError("MCP server initialization failed.")
        return result

    async def shutdown(
        self, *, exc_type: type[BaseException] | None = None, exc_val: BaseException | None = None, exc_tb: Any = None
    ) -> None:
        if not self._started:
            return
        await self._actor.ask(_Shutdown(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb))
        await self._actor.stop()
        self._started = False

    async def _handle_message(self, message: _Message) -> MCPServerBundle | None:
        if isinstance(message, _Initialize):
            if self._cm is not None:
                raise RuntimeError("MCP servers already initialized.")
            self._cm = get_mcp_servers_from_config(message.config_servers, working_directory=message.working_directory)
            try:
                self._servers = await self._cm.__aenter__()
                tools = await get_mcp_wrapped_tools(self._servers)
            except Exception:
                await self._cleanup()
                raise
            return MCPServerBundle(servers=list(self._servers), tools=tools)

        if isinstance(message, _Shutdown):
            await self._cleanup(exc_type=message.exc_type, exc_val=message.exc_val, exc_tb=message.exc_tb)
            return None

        raise RuntimeError(f"Unknown MCP manager message: {message!r}")

    async def _cleanup(
        self,
        *,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: Any = None,
    ) -> None:
        if self._cm is not None:
            await self._cm.__aexit__(exc_type, exc_val, exc_tb)
        self._cm = None
        self._servers = None
