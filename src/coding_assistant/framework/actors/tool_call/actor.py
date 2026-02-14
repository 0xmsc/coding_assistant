from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from json import JSONDecodeError
from typing import Any

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    CancelToolCallsRequest,
    HandleToolCallsRequest,
    HandleToolCallsResponse,
    ToolCapability,
    ToolCallExecutionResult,
)
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.actors.tool_call.executor import ToolExecutor
from coding_assistant.llm.types import (
    AssistantMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    ToolResult,
)
from coding_assistant.ui import UI


class ToolCallActor:
    def __init__(
        self,
        *,
        default_tool_capabilities: tuple[ToolCapability, ...],
        ui: UI,
        context_name: str,
        actor_directory: ActorDirectory | None = None,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    ) -> None:
        if actor_directory is None:
            raise RuntimeError("ToolCallActor requires an actor directory for URI capability routing.")
        self._executor = ToolExecutor(
            actor_directory=actor_directory,
            default_tool_capabilities=default_tool_capabilities,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
            ui=ui,
            context_name=context_name,
        )
        self._actor: Actor[HandleToolCallsRequest | CancelToolCallsRequest] = Actor(
            name=f"{context_name}.tool-calls", handler=self._handle_message
        )
        self._actor_directory = actor_directory
        self._inflight_tasks_by_request: dict[str, set[asyncio.Task[ToolResult]]] = {}
        self._inflight_requests: dict[str, asyncio.Task[None]] = {}
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
        for task in self._inflight_requests.values():
            task.cancel()
        if self._inflight_requests:
            await asyncio.gather(*self._inflight_requests.values(), return_exceptions=True)
        self._inflight_requests.clear()
        await self._actor.stop()
        await self._executor.stop()
        self._started = False

    async def send_message(self, message: HandleToolCallsRequest | CancelToolCallsRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: HandleToolCallsRequest | CancelToolCallsRequest) -> None:
        if isinstance(message, CancelToolCallsRequest):
            request_task = self._inflight_requests.pop(message.request_id, None)
            if request_task is not None:
                request_task.cancel()
            for task in self._inflight_tasks_by_request.pop(message.request_id, set()):
                task.cancel()
            return None
        if message.request_id in self._inflight_requests:
            await self._send_response(
                message,
                HandleToolCallsResponse(
                    request_id=message.request_id,
                    results=[],
                    error=RuntimeError(f"Duplicate tool-call request id: {message.request_id}"),
                ),
            )
            return None

        async def _run_request() -> None:
            try:
                results, cancelled = await self._handle_tool_calls(
                    message.request_id,
                    message.message,
                    tool_capabilities=message.tool_capabilities,
                )
                await self._send_response(
                    message,
                    HandleToolCallsResponse(request_id=message.request_id, results=results, cancelled=cancelled),
                )
            except asyncio.CancelledError:
                await self._send_response(
                    message, HandleToolCallsResponse(request_id=message.request_id, results=[], cancelled=True)
                )
            except BaseException as exc:
                await self._send_response(
                    message, HandleToolCallsResponse(request_id=message.request_id, results=[], error=exc)
                )
            finally:
                self._inflight_requests.pop(message.request_id, None)

        request_task = asyncio.create_task(_run_request(), name=f"tool-calls:{message.request_id}")
        self._inflight_requests[message.request_id] = request_task
        return None

    async def _handle_tool_calls(
        self,
        request_id: str,
        message: AssistantMessage,
        tool_capabilities: tuple[ToolCapability, ...] | None,
        on_result: Callable[[ToolCallExecutionResult], None] | None = None,
    ) -> tuple[list[ToolCallExecutionResult], bool]:
        tool_calls = message.tool_calls
        if not tool_calls:
            return [], False

        tasks_with_calls: dict[asyncio.Task[ToolResult], Any] = {}
        effective_capabilities = (
            tool_capabilities if tool_capabilities is not None else self._executor.default_tool_capabilities
        )
        for tool_call in tool_calls:
            task = await self._executor.submit(tool_call, tool_capabilities=effective_capabilities)
            tasks_with_calls[task] = tool_call
        self._inflight_tasks_by_request[request_id] = set(tasks_with_calls.keys())

        any_cancelled = False
        execution_results: list[ToolCallExecutionResult] = []
        pending = set(tasks_with_calls.keys())
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    tool_call = tasks_with_calls[task]
                    cancelled = False
                    try:
                        result: ToolResult = await task
                    except asyncio.CancelledError:
                        result = TextResult(content="Tool execution was cancelled.")
                        any_cancelled = True
                        cancelled = True

                    try:
                        function_args = json.loads(tool_call.function.arguments)
                    except JSONDecodeError:
                        function_args = {}
                    if not isinstance(function_args, dict):
                        function_args = {}

                    execution_results.append(
                        ToolCallExecutionResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=function_args,
                            result=result,
                            cancelled=cancelled,
                        )
                    )
                    if on_result is not None:
                        on_result(execution_results[-1])
        finally:
            self._inflight_tasks_by_request.pop(request_id, None)

        return execution_results, any_cancelled

    async def _send_response(self, request: HandleToolCallsRequest, response: HandleToolCallsResponse) -> None:
        if self._actor_directory is None:
            raise RuntimeError("ToolCallActor cannot send by URI without actor directory.")
        await self._actor_directory.send_message(uri=request.reply_to_uri, message=response)
