from __future__ import annotations

import asyncio
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


class MCPServerManager:
    def __init__(self) -> None:
        self._cm: Any | None = None
        self._servers: list[MCPServer] | None = None

    async def initialize(self, *, config_servers: list[MCPServerConfig], working_directory: Path) -> MCPServerBundle:
        if self._cm is not None:
            raise RuntimeError("MCP servers already initialized.")
        self._cm = get_mcp_servers_from_config(config_servers, working_directory=working_directory)
        try:
            self._servers = await self._cm.__aenter__()
            tools = await get_mcp_wrapped_tools(self._servers)
            return MCPServerBundle(servers=list(self._servers), tools=tools)
        except BaseException:
            await self._cleanup()
            raise

    async def shutdown(
        self, *, exc_type: type[BaseException] | None = None, exc_val: BaseException | None = None, exc_tb: Any = None
    ) -> None:
        await self._cleanup(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

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


@dataclass(slots=True)
class _InitializeRequest:
    config_servers: list[MCPServerConfig]
    working_directory: Path
    future: asyncio.Future[MCPServerBundle]


@dataclass(slots=True)
class _ShutdownRequest:
    exc_type: type[BaseException] | None
    exc_val: BaseException | None
    exc_tb: Any
    future: asyncio.Future[None]


_Message = _InitializeRequest | _ShutdownRequest


class MCPServerManagerActor:
    def __init__(self, *, context_name: str = "mcp-manager") -> None:
        self._actor: Actor[_Message] = Actor(name=f"{context_name}.mcp-manager", handler=self._handle_message)
        self._manager = MCPServerManager()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def initialize(self, *, config_servers: list[MCPServerConfig], working_directory: Path) -> MCPServerBundle:
        self.start()
        future: asyncio.Future[MCPServerBundle] = asyncio.get_running_loop().create_future()
        await self._actor.send(
            _InitializeRequest(
                config_servers=config_servers,
                working_directory=working_directory,
                future=future,
            )
        )
        return await future

    async def shutdown(
        self, *, exc_type: type[BaseException] | None = None, exc_val: BaseException | None = None, exc_tb: Any = None
    ) -> None:
        if not self._started:
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._actor.send(
            _ShutdownRequest(
                exc_type=exc_type,
                exc_val=exc_val,
                exc_tb=exc_tb,
                future=future,
            )
        )
        await future
        await self._actor.stop()
        self._started = False

    async def _handle_message(self, message: _Message) -> None:
        if isinstance(message, _InitializeRequest):
            try:
                bundle = await self._manager.initialize(
                    config_servers=message.config_servers,
                    working_directory=message.working_directory,
                )
            except BaseException as exc:
                if not message.future.done():
                    message.future.set_exception(exc)
                return None
            if not message.future.done():
                message.future.set_result(bundle)
            return None

        if isinstance(message, _ShutdownRequest):
            try:
                await self._manager.shutdown(exc_type=message.exc_type, exc_val=message.exc_val, exc_tb=message.exc_tb)
            except BaseException as exc:
                if not message.future.done():
                    message.future.set_exception(exc)
                return None
            if not message.future.done():
                message.future.set_result(None)
            return None

        raise RuntimeError(f"Unknown MCP manager actor message: {message!r}")
