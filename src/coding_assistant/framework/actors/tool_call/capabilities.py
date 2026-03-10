from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, cast
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import ToolCapability
from coding_assistant.llm.types import Tool, ToolResult


@dataclass(slots=True)
class ExecuteToolRequest:
    request_id: str
    arguments: dict[str, object]
    reply_to_uri: str


@dataclass(slots=True)
class CancelToolExecutionRequest:
    request_id: str


@dataclass(slots=True)
class ExecuteToolResponse:
    request_id: str
    result: ToolResult | None
    error: BaseException | None = None


@dataclass(slots=True)
class _ToolExecuteReplyWaiter:
    request_id: str
    future: asyncio.Future[ToolResult]

    async def send_message(self, message: object) -> None:
        if self.future.done():
            return
        if not isinstance(message, ExecuteToolResponse):
            self.future.set_exception(RuntimeError(f"Unexpected tool response type: {type(message).__name__}"))
            return
        if message.request_id != self.request_id:
            self.future.set_exception(RuntimeError(f"Mismatched tool response id: {message.request_id}"))
            return
        if message.error is not None:
            self.future.set_exception(message.error)
            return
        if message.result is None:
            self.future.set_exception(RuntimeError("Tool execution response missing result."))
            return
        self.future.set_result(message.result)


class ToolExecutionBoundary:
    async def invoke(self, tool: Tool, arguments: dict[str, object]) -> ToolResult:
        execute_method = cast(Callable[[dict[str, object]], Awaitable[ToolResult]], getattr(tool, "execute"))
        return await execute_method(arguments)


class ToolCapabilityActor:
    def __init__(
        self,
        *,
        actor_directory: ActorDirectory,
        capability: ToolCapability,
        tool: Tool,
        boundary: ToolExecutionBoundary | None = None,
    ) -> None:
        self.capability = capability
        self._actor_directory = actor_directory
        self._tool = tool
        self._boundary = boundary or ToolExecutionBoundary()
        self._actor = Actor[ExecuteToolRequest | CancelToolExecutionRequest](
            name=f"tool-capability:{capability.name}", handler=self._handle_message
        )
        self._started = False
        self._inflight: dict[str, asyncio.Task[None]] = {}

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        for task in list(self._inflight.values()):
            task.cancel()
        if self._inflight:
            await asyncio.gather(*list(self._inflight.values()), return_exceptions=True)
            self._inflight.clear()
        await self._actor.stop()
        self._started = False

    async def send_message(self, message: ExecuteToolRequest | CancelToolExecutionRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: ExecuteToolRequest | CancelToolExecutionRequest) -> None:
        if isinstance(message, CancelToolExecutionRequest):
            task = self._inflight.pop(message.request_id, None)
            if task is not None:
                task.cancel()
            return None

        async def _run() -> None:
            try:
                result = await self._boundary.invoke(self._tool, message.arguments)
                response = ExecuteToolResponse(request_id=message.request_id, result=result)
            except BaseException as exc:
                response = ExecuteToolResponse(request_id=message.request_id, result=None, error=exc)
            finally:
                self._inflight.pop(message.request_id, None)
            await self._actor_directory.send_message(uri=message.reply_to_uri, message=response)

        task = asyncio.create_task(_run(), name=f"tool-capability-run:{message.request_id}")
        self._inflight[message.request_id] = task
        return None


def register_tool_capabilities(
    *,
    actor_directory: ActorDirectory,
    tools: tuple[Tool, ...],
    context_name: str,
    uri_prefix: str,
) -> tuple[tuple[ToolCapability, ...], tuple[ToolCapabilityActor, ...]]:
    capabilities: list[ToolCapability] = []
    actors: list[ToolCapabilityActor] = []
    seen_names: set[str] = set()
    for tool in tools:
        name = tool.name()
        if name in seen_names:
            continue
        seen_names.add(name)
        capability = ToolCapability(
            name=name,
            uri=f"{uri_prefix}/{name}/{uuid4().hex}",
            description=tool.description(),
            parameters=tool.parameters(),
        )
        capability_actor = ToolCapabilityActor(
            actor_directory=actor_directory,
            capability=capability,
            tool=tool,
        )
        actor_directory.register(uri=capability.uri, actor=capability_actor)
        capability_actor.start()
        capabilities.append(capability)
        actors.append(capability_actor)
    return tuple(capabilities), tuple(actors)


async def invoke_tool_capability(
    *,
    actor_directory: ActorDirectory,
    capability: ToolCapability,
    arguments: dict[str, object],
    context_name: str,
) -> ToolResult:
    request_id = uuid4().hex
    reply_uri = f"actor://{context_name}/tool-capability-reply/{request_id}"
    loop = asyncio.get_running_loop()
    future: asyncio.Future[ToolResult] = loop.create_future()
    actor_directory.register(uri=reply_uri, actor=_ToolExecuteReplyWaiter(request_id=request_id, future=future))
    try:
        await actor_directory.send_message(
            uri=capability.uri,
            message=ExecuteToolRequest(request_id=request_id, arguments=arguments, reply_to_uri=reply_uri),
        )
        return await future
    except asyncio.CancelledError:
        await actor_directory.send_message(
            uri=capability.uri,
            message=CancelToolExecutionRequest(request_id=request_id),
        )
        try:
            await asyncio.shield(future)
        except BaseException:
            pass
        raise
    finally:
        actor_directory.unregister(uri=reply_uri)
