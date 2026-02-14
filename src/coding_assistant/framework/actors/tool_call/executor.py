from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Sequence
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.results import TextResult
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.llm.types import Tool, ToolCall, ToolResult
from coding_assistant.trace import trace_json
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ToolCallRequest:
    request_id: str
    tool_call: ToolCall
    tools: tuple[Tool, ...]


@dataclass(slots=True)
class _ToolCallSpawned:
    request_id: str
    task: asyncio.Task[ToolResult] | None
    error: BaseException | None = None


@dataclass(slots=True)
class _ExecuteToolRequest:
    request_id: str
    function_args: dict[str, Any]
    reply_to_uri: str


@dataclass(slots=True)
class _ExecuteToolResponse:
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
        if not isinstance(message, _ExecuteToolResponse):
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


class _ToolCapabilityActor:
    def __init__(self, *, name: str, tool: Tool, actor_directory: ActorDirectory) -> None:
        self._tool = tool
        self._actor = Actor[_ExecuteToolRequest](name=name, handler=self._handle_message)
        self._actor_directory = actor_directory
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        self._started = False

    async def send_message(self, message: _ExecuteToolRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: _ExecuteToolRequest) -> None:
        try:
            result = await self._tool.execute(message.function_args)
            response = _ExecuteToolResponse(request_id=message.request_id, result=result)
        except BaseException as exc:
            response = _ExecuteToolResponse(request_id=message.request_id, result=None, error=exc)
        await self._actor_directory.send_message(uri=message.reply_to_uri, message=response)


@dataclass(slots=True)
class ToolExecutor:
    tools: Sequence[Tool]
    ui: UI
    context_name: str = "tool-executor"
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks()
    tool_callbacks: ToolCallbacks = NullToolCallbacks()
    _actor: Actor["_ToolMessage"] = field(init=False)
    _pending: dict[str, asyncio.Future[asyncio.Task[ToolResult]]] = field(init=False, default_factory=dict)
    _tool_capability_directory: ActorDirectory = field(init=False)
    _tool_capability_uris: dict[int, str] = field(init=False, default_factory=dict)
    _tool_capability_actors: dict[str, _ToolCapabilityActor] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._actor = Actor(name=f"{self.context_name}.tool-executor", handler=self._handle_message)
        self._tool_capability_directory = ActorDirectory()

    def start(self) -> None:
        self._actor.start()

    async def stop(self) -> None:
        await self._actor.stop()
        for uri, capability in list(self._tool_capability_actors.items()):
            await capability.stop()
            self._tool_capability_directory.unregister(uri=uri)
        self._tool_capability_actors.clear()
        self._tool_capability_uris.clear()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("ToolExecutor stopped before reply."))
        self._pending.clear()

    async def submit(self, tool_call: ToolCall, *, tools: Sequence[Tool] | None = None) -> asyncio.Task[ToolResult]:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[asyncio.Task[ToolResult]] = loop.create_future()
        self._pending[request_id] = future
        await self._actor.send(
            _ToolCallRequest(
                request_id=request_id,
                tool_call=tool_call,
                tools=tuple(tools) if tools is not None else tuple(self.tools),
            )
        )
        return await future

    async def _handle_message(self, message: "_ToolMessage") -> None:
        if isinstance(message, _ToolCallRequest):
            try:
                task = asyncio.create_task(
                    self._execute_tool_call(message.tool_call, message.tools),
                    name=f"{message.tool_call.function.name} ({message.tool_call.id})",
                )
                await self._actor.send(_ToolCallSpawned(request_id=message.request_id, task=task))
            except BaseException as exc:
                await self._actor.send(_ToolCallSpawned(request_id=message.request_id, task=None, error=exc))
            return None
        if isinstance(message, _ToolCallSpawned):
            future = self._pending.pop(message.request_id, None)
            if future is None:
                raise RuntimeError(f"Unknown tool spawn response id: {message.request_id}")
            if message.error is not None:
                future.set_exception(message.error)
            else:
                if message.task is None:
                    future.set_exception(RuntimeError("Tool spawn response missing task."))
                else:
                    future.set_result(message.task)
            return None
        raise RuntimeError(f"Unknown tool executor message: {message!r}")

    async def _execute_tool_call(self, tool_call: ToolCall, tools: Sequence[Tool]) -> ToolResult:
        function_name = tool_call.function.name
        if not function_name:
            raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

        args_str = tool_call.function.arguments

        try:
            function_args = json.loads(args_str)
        except JSONDecodeError as e:
            logger.error(
                f"[{self.context_name}] [{tool_call.id}] Failed to parse tool '{function_name}' arguments as JSON: {e} | raw: {args_str}"
            )
            return TextResult(content=f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}")

        logger.debug(
            f"[{tool_call.id}] [{self.context_name}] Calling tool '{function_name}' with arguments {function_args}"
        )

        self.progress_callbacks.on_tool_start(self.context_name, tool_call, function_args)

        function_call_result: ToolResult
        try:
            if callback_result := await self.tool_callbacks.before_tool_execution(
                self.context_name,
                tool_call.id,
                function_name,
                function_args,
                ui=self.ui,
            ):
                logger.info(
                    f"[{tool_call.id}] [{self.context_name}] Tool '{function_name}' execution was prevented via callback."
                )
                function_call_result = callback_result
            else:
                function_call_result = await self._execute_tool_by_name(
                    function_name=function_name,
                    function_args=function_args,
                    tools=tools,
                )
        except Exception as e:
            function_call_result = TextResult(content=f"Error executing tool: {e}")

        trace_json(
            f"tool_result_{function_name}.json",
            {
                "tool_call_id": tool_call.id,
                "function_name": function_name,
                "function_args": function_args,
                "result": function_call_result.to_dict(),
            },
        )

        return function_call_result

    def _ensure_tool_capability(self, *, tool: Tool) -> str:
        tool_id = id(tool)
        existing_uri = self._tool_capability_uris.get(tool_id)
        if existing_uri is not None:
            return existing_uri
        capability_uri = f"actor://{self.context_name}/tool-capability/{tool.name()}/{tool_id}"
        capability_actor = _ToolCapabilityActor(
            name=f"{self.context_name}.tool-capability.{tool.name()}",
            tool=tool,
            actor_directory=self._tool_capability_directory,
        )
        self._tool_capability_directory.register(uri=capability_uri, actor=capability_actor)
        capability_actor.start()
        self._tool_capability_uris[tool_id] = capability_uri
        self._tool_capability_actors[capability_uri] = capability_actor
        return capability_uri

    async def _execute_tool_by_name(
        self, *, function_name: str, function_args: dict[str, Any], tools: Sequence[Tool]
    ) -> ToolResult:
        target_tool = next((tool for tool in tools if tool.name() == function_name), None)
        if target_tool is None:
            raise ValueError(f"Tool {function_name} not found in agent tools.")
        capability_uri = self._ensure_tool_capability(tool=target_tool)
        request_id = uuid4().hex
        reply_uri = f"actor://{self.context_name}/tool-reply/{request_id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ToolResult] = loop.create_future()
        self._tool_capability_directory.register(
            uri=reply_uri,
            actor=_ToolExecuteReplyWaiter(request_id=request_id, future=future),
        )
        try:
            await self._tool_capability_directory.send_message(
                uri=capability_uri,
                message=_ExecuteToolRequest(
                    request_id=request_id,
                    function_args=function_args,
                    reply_to_uri=reply_uri,
                ),
            )
            return await future
        finally:
            self._tool_capability_directory.unregister(uri=reply_uri)


_ToolMessage = _ToolCallRequest | _ToolCallSpawned
