from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_assistant.config import MCPServerConfig
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.llm.types import Tool
from coding_assistant.tools.mcp import MCPServer, get_mcp_servers_from_config, get_mcp_wrapped_tools


@dataclass(slots=True)
class MCPServerBundle:
    servers: list[MCPServer]
    tools: list[Tool]


@dataclass(slots=True)
class _InitializeRequest:
    request_id: str
    config_servers: list[MCPServerConfig]
    working_directory: Path


@dataclass(slots=True)
class _InitializeDone:
    request_id: str
    bundle: MCPServerBundle | None
    error: BaseException | None = None


@dataclass(slots=True)
class _ShutdownRequest:
    request_id: str
    exc_type: type[BaseException] | None
    exc_val: BaseException | None
    exc_tb: Any


@dataclass(slots=True)
class _ShutdownDone:
    request_id: str
    error: BaseException | None = None


_Message = _InitializeRequest | _InitializeDone | _ShutdownRequest | _ShutdownDone


class MCPServerManager:
    def __init__(self, *, context_name: str = "mcp-manager") -> None:
        self._context_name = context_name
        self._actor: Actor[_Message] = Actor(name=f"{context_name}.mcp-manager", handler=self._handle_message)
        self._cm: Any | None = None
        self._servers: list[MCPServer] | None = None
        self._started = False
        self._pending_init: dict[str, asyncio.Future[MCPServerBundle]] = {}
        self._pending_shutdown: dict[str, asyncio.Future[None]] = {}

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def initialize(self, *, config_servers: list[MCPServerConfig], working_directory: Path) -> MCPServerBundle:
        self.start()
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[MCPServerBundle] = loop.create_future()
        self._pending_init[request_id] = future
        await self._actor.send(
            _InitializeRequest(
                request_id=request_id,
                config_servers=config_servers,
                working_directory=working_directory,
            )
        )
        return await future

    async def shutdown(
        self, *, exc_type: type[BaseException] | None = None, exc_val: BaseException | None = None, exc_tb: Any = None
    ) -> None:
        if not self._started:
            return
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        self._pending_shutdown[request_id] = future
        await self._actor.send(
            _ShutdownRequest(
                request_id=request_id,
                exc_type=exc_type,
                exc_val=exc_val,
                exc_tb=exc_tb,
            )
        )
        await future
        await self._actor.stop()
        for init_pending in self._pending_init.values():
            if not init_pending.done():
                init_pending.set_exception(RuntimeError("MCPServerManager stopped before initialize reply."))
        self._pending_init.clear()
        for shutdown_pending in self._pending_shutdown.values():
            if not shutdown_pending.done():
                shutdown_pending.set_exception(RuntimeError("MCPServerManager stopped before shutdown reply."))
        self._pending_shutdown.clear()
        self._started = False

    async def _handle_message(self, message: _Message) -> None:
        if isinstance(message, _InitializeRequest):
            if self._cm is not None:
                raise RuntimeError("MCP servers already initialized.")
            self._cm = get_mcp_servers_from_config(message.config_servers, working_directory=message.working_directory)
            try:
                self._servers = await self._cm.__aenter__()
                tools = await get_mcp_wrapped_tools(self._servers)
            except Exception as exc:
                await self._cleanup()
                await self._actor.send(_InitializeDone(request_id=message.request_id, bundle=None, error=exc))
                return None
            await self._actor.send(
                _InitializeDone(
                    request_id=message.request_id,
                    bundle=MCPServerBundle(servers=list(self._servers), tools=tools),
                )
            )
            return None

        if isinstance(message, _InitializeDone):
            init_future = self._pending_init.pop(message.request_id, None)
            if init_future is None:
                raise RuntimeError(f"Unknown initialize response id: {message.request_id}")
            if message.error is not None:
                init_future.set_exception(message.error)
            else:
                if message.bundle is None:
                    init_future.set_exception(RuntimeError("Initialize response missing bundle."))
                else:
                    init_future.set_result(message.bundle)
            return None

        if isinstance(message, _ShutdownRequest):
            error: BaseException | None = None
            try:
                await self._cleanup(exc_type=message.exc_type, exc_val=message.exc_val, exc_tb=message.exc_tb)
            except BaseException as exc:
                error = exc
            await self._actor.send(_ShutdownDone(request_id=message.request_id, error=error))
            return None

        if isinstance(message, _ShutdownDone):
            shutdown_future = self._pending_shutdown.pop(message.request_id, None)
            if shutdown_future is None:
                raise RuntimeError(f"Unknown shutdown response id: {message.request_id}")
            if message.error is not None:
                shutdown_future.set_exception(message.error)
            else:
                shutdown_future.set_result(None)
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
