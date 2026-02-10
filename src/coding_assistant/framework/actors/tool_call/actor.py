from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any
from uuid import uuid4

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import HandleToolCallsRequest, HandleToolCallsResponse
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.history import append_tool_message
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.tool_executor import ToolExecutor
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    ToolMessage,
    ToolResult,
)
from coding_assistant.ui import UI


class ToolCallActor:
    def __init__(
        self,
        *,
        tools: Sequence[Tool],
        ui: UI,
        context_name: str,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    ) -> None:
        self._context_name = context_name
        self._progress_callbacks = progress_callbacks
        self._executor = ToolExecutor(
            tools=tools,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
            ui=ui,
            context_name=context_name,
        )
        self._actor: Actor[HandleToolCallsRequest] = Actor(
            name=f"{context_name}.tool-calls", handler=self._handle_message
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._executor.start()
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        await self._executor.stop()
        self._started = False

    def set_tools(self, tools: Sequence[Tool]) -> None:
        self._executor.set_tools(tools)

    async def handle_tool_calls(
        self,
        message: AssistantMessage,
        *,
        history: list[BaseMessage],
        task_created_callback: Callable[[str, asyncio.Task[object]], None] | None = None,
        handle_tool_result: Callable[[ToolResult], str] | None = None,
    ) -> None:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        @dataclass(slots=True)
        class _ReplySink:
            async def send_message(self, message: HandleToolCallsResponse) -> None:
                if message.request_id != request_id:
                    future.set_exception(RuntimeError(f"Mismatched tool response id: {message.request_id}"))
                    return
                if message.error is not None:
                    future.set_exception(message.error)
                    return
                future.set_result(None)

        await self.send_message(
            HandleToolCallsRequest(
                request_id=request_id,
                message=message,
                history=history,
                task_created_callback=task_created_callback,
                handle_tool_result=handle_tool_result,
                reply_to=_ReplySink(),
            )
        )
        await future

    async def send_message(self, message: HandleToolCallsRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: HandleToolCallsRequest) -> None:
        try:
            await self._handle_tool_calls(
                message.message,
                history=message.history,
                task_created_callback=message.task_created_callback,
                handle_tool_result=message.handle_tool_result,
            )
            await message.reply_to.send_message(HandleToolCallsResponse(request_id=message.request_id))
        except BaseException as exc:
            await message.reply_to.send_message(HandleToolCallsResponse(request_id=message.request_id, error=exc))

    async def _handle_tool_calls(
        self,
        message: AssistantMessage,
        *,
        history: list[BaseMessage],
        task_created_callback: Callable[[str, asyncio.Task[object]], None] | None = None,
        handle_tool_result: Callable[[ToolResult], str] | None = None,
    ) -> None:
        tool_calls = message.tool_calls
        if not tool_calls:
            return

        tasks_with_calls: dict[asyncio.Task[ToolResult], Any] = {}
        for tool_call in tool_calls:
            task = await self._executor.submit(tool_call)
            if task_created_callback is not None:
                task_created_callback(tool_call.id, task)
            tasks_with_calls[task] = tool_call

        any_cancelled = False
        pending = set(tasks_with_calls.keys())
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                tool_call = tasks_with_calls[task]
                try:
                    result: ToolResult = await task
                except asyncio.CancelledError:
                    result = TextResult(content="Tool execution was cancelled.")
                    any_cancelled = True

                if handle_tool_result:
                    result_summary = handle_tool_result(result)
                else:
                    if isinstance(result, TextResult):
                        result_summary = result.content
                    else:
                        result_summary = f"Tool produced result of type {type(result).__name__}"

                if result_summary is None:
                    raise RuntimeError(f"Tool call {tool_call.id} produced empty result summary.")

                try:
                    function_args = json.loads(tool_call.function.arguments)
                except JSONDecodeError:
                    function_args = {}

                tool_message = ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    content=result_summary,
                )
                append_tool_message(
                    history,
                    callbacks=self._progress_callbacks,
                    context_name=self._context_name,
                    message=tool_message,
                    arguments=function_args,
                )

        if any_cancelled:
            raise asyncio.CancelledError()
