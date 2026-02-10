from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from json import JSONDecodeError
from typing import Any, Sequence
from uuid import uuid4

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


@dataclass(slots=True)
class _ToolCallSpawned:
    request_id: str
    task: asyncio.Task[ToolResult] | None
    error: BaseException | None = None


@dataclass(slots=True)
class ToolExecutor:
    tools: Sequence[Tool]
    ui: UI
    context_name: str = "tool-executor"
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks()
    tool_callbacks: ToolCallbacks = NullToolCallbacks()
    _actor: Actor["_ToolMessage"] = field(init=False)
    _pending: dict[str, asyncio.Future[asyncio.Task[ToolResult]]] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._actor = Actor(name=f"{self.context_name}.tool-executor", handler=self._handle_message)

    def start(self) -> None:
        self._actor.start()

    async def stop(self) -> None:
        await self._actor.stop()
        for future in self._pending.values():
            if not future.done():
                future.set_exception(RuntimeError("ToolExecutor stopped before reply."))
        self._pending.clear()

    def set_tools(self, tools: Sequence[Tool]) -> None:
        self.tools = list(tools)

    async def submit(self, tool_call: ToolCall) -> asyncio.Task[ToolResult]:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[asyncio.Task[ToolResult]] = loop.create_future()
        self._pending[request_id] = future
        await self._actor.send(_ToolCallRequest(request_id=request_id, tool_call=tool_call))
        return await future

    async def _handle_message(self, message: "_ToolMessage") -> None:
        if isinstance(message, _ToolCallRequest):
            try:
                task = asyncio.create_task(
                    self._execute_tool_call(message.tool_call),
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

    async def _execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
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
                    function_name=function_name, function_args=function_args
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

    async def _execute_tool_by_name(self, *, function_name: str, function_args: dict[str, Any]) -> ToolResult:
        for tool in self.tools:
            if tool.name() == function_name:
                return await tool.execute(function_args)
        raise ValueError(f"Tool {function_name} not found in agent tools.")


_ToolMessage = _ToolCallRequest | _ToolCallSpawned
