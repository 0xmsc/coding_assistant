import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.history import append_tool_message
from coding_assistant.framework.tool_executor import ToolExecutor
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, ToolMessage, ToolResult, Usage
from coding_assistant.framework.results import TextResult
from coding_assistant.ui import UI, ui_actor_scope


@dataclass(slots=True)
class _HandleToolCalls:
    message: AssistantMessage
    history: list[BaseMessage]
    task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None
    handle_tool_result: Callable[[ToolResult], str] | None


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
        self._ui = ui
        self._context_name = context_name
        self._progress_callbacks = progress_callbacks
        self._tool_callbacks = tool_callbacks
        self._executor = ToolExecutor(
            tools=tools,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
            ui=ui,
            context_name=context_name,
        )
        self._actor: Actor[_HandleToolCalls, None] = Actor(
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

    async def handle_tool_calls(
        self,
        message: AssistantMessage,
        *,
        history: list[BaseMessage],
        task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None = None,
        handle_tool_result: Callable[[ToolResult], str] | None = None,
    ) -> None:
        self.start()
        await self._actor.ask(
            _HandleToolCalls(
                message=message,
                history=history,
                task_created_callback=task_created_callback,
                handle_tool_result=handle_tool_result,
            )
        )

    async def _handle_message(self, message: _HandleToolCalls) -> None:
        if not isinstance(message, _HandleToolCalls):
            raise RuntimeError(f"Unknown tool call message: {message!r}")
        await _handle_tool_calls_impl(
            message.message,
            history=message.history,
            executor=self._executor,
            progress_callbacks=self._progress_callbacks,
            context_name=self._context_name,
            task_created_callback=message.task_created_callback,
            handle_tool_result=message.handle_tool_result,
        )


async def _handle_tool_calls_impl(
    message: AssistantMessage,
    *,
    history: list[BaseMessage],
    executor: ToolExecutor,
    progress_callbacks: ProgressCallbacks,
    context_name: str,
    task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None = None,
    handle_tool_result: Callable[[ToolResult], str] | None = None,
) -> None:
    tool_calls = message.tool_calls

    if not tool_calls:
        return

    tasks_with_calls = {}
    for tool_call in tool_calls:
        task = await executor.submit(tool_call)
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
                callbacks=progress_callbacks,
                context_name=context_name,
                message=tool_message,
                arguments=function_args,
            )

    if any_cancelled:
        raise asyncio.CancelledError()


async def handle_tool_calls(
    message: AssistantMessage,
    *,
    history: list[BaseMessage],
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    ui: UI,
    context_name: str,
    task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None = None,
    handle_tool_result: Callable[[ToolResult], str] | None = None,
    tool_call_actor: ToolCallActor | None = None,
) -> None:
    if not message.tool_calls:
        return

    if tool_call_actor is not None:
        await tool_call_actor.handle_tool_calls(
            message,
            history=history,
            task_created_callback=task_created_callback,
            handle_tool_result=handle_tool_result,
        )
        return

    async with ui_actor_scope(ui, context_name=context_name) as actor_ui:
        actor = ToolCallActor(
            tools=tools,
            ui=actor_ui,
            context_name=context_name,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
        )
        actor.start()
        try:
            await actor.handle_tool_calls(
                message,
                history=history,
                task_created_callback=task_created_callback,
                handle_tool_result=handle_tool_result,
            )
        finally:
            await actor.stop()


async def do_single_step(
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    completer: Completer,
    context_name: str,
    agent_actor: "AgentActor | None" = None,
) -> tuple[AssistantMessage, Usage | None]:
    if agent_actor is None:
        actor = AgentActor(context_name=context_name)
        actor.start()
        try:
            return await actor.do_single_step(
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                context_name=context_name,
            )
        finally:
            await actor.stop()
    return await agent_actor.do_single_step(
        history=history,
        model=model,
        tools=tools,
        progress_callbacks=progress_callbacks,
        completer=completer,
        context_name=context_name,
    )


@dataclass(slots=True)
class _DoSingleStep:
    history: list[BaseMessage]
    model: str
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    context_name: str


class AgentActor:
    def __init__(self, *, context_name: str = "agent") -> None:
        self._actor: Actor[_DoSingleStep, tuple[AssistantMessage, Usage | None]] = Actor(
            name=f"{context_name}.single-step", handler=self._handle_message
        )
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

    async def do_single_step(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
        context_name: str,
    ) -> tuple[AssistantMessage, Usage | None]:
        self.start()
        return await self._actor.ask(
            _DoSingleStep(
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                context_name=context_name,
            )
        )

    async def _handle_message(self, message: _DoSingleStep) -> tuple[AssistantMessage, Usage | None]:
        if not isinstance(message, _DoSingleStep):
            raise RuntimeError(f"Unknown single-step message: {message!r}")
        return await _do_single_step_impl(
            history=message.history,
            model=message.model,
            tools=message.tools,
            progress_callbacks=message.progress_callbacks,
            completer=message.completer,
            context_name=message.context_name,
        )


async def _do_single_step_impl(
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    completer: Completer,
    context_name: str,
) -> tuple[AssistantMessage, Usage | None]:
    if not history:
        raise RuntimeError("History is required in order to run a step.")

    completion = await completer(
        history,
        model=model,
        tools=tools,
        callbacks=progress_callbacks,
    )
    message = completion.message

    return message, completion.usage
