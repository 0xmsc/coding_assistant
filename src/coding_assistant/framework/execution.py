import asyncio
import json
from collections.abc import Callable, Sequence
from json import JSONDecodeError
from typing import Any

from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.history import append_tool_message
from coding_assistant.framework.tool_executor import ToolExecutor
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, ToolMessage, ToolResult, Usage
from coding_assistant.framework.results import TextResult
from coding_assistant.ui import UI


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
) -> None:
    tool_calls = message.tool_calls

    if not tool_calls:
        return

    executor = ToolExecutor(
        tools=tools,
        progress_callbacks=progress_callbacks,
        tool_callbacks=tool_callbacks,
        ui=ui,
        context_name=context_name,
    )
    executor.start()

    try:
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
    finally:
        await executor.stop()


async def do_single_step(
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
