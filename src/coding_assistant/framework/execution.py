import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from json import JSONDecodeError
from typing import cast

from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.history import append_tool_message
from coding_assistant.llm.types import AssistantMessage, LLMMessage, ToolCall
from coding_assistant.framework.types import Tool, ToolResult
from coding_assistant.framework.types import (
    Completer,
    TextResult,
)
from coding_assistant.llm.adapters import execute_tool_call, get_tools
from coding_assistant.trace import trace_data
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


async def handle_tool_call(
    tool_call: ToolCall,
    tools: Sequence[Tool],
    history: list[LLMMessage],
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    *,
    ui: UI,
    context_name: str,
) -> ToolResult:
    """Execute a single tool call and return ToolResult."""
    function_name = tool_call.function.name
    if not function_name:
        raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

    args_str = tool_call.function.arguments

    try:
        function_args = json.loads(args_str)
    except JSONDecodeError as e:
        logger.error(
            f"[{context_name}] [{tool_call.id}] Failed to parse tool '{function_name}' arguments as JSON: {e} | raw: {args_str}"
        )
        return TextResult(content=f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}")

    logger.debug(f"[{tool_call.id}] [{context_name}] Calling tool '{function_name}' with arguments {function_args}")

    # Notify callbacks that tool is starting
    progress_callbacks.on_tool_start(context_name, tool_call.id, function_name, function_args)

    function_call_result: ToolResult
    try:
        if callback_result := await tool_callbacks.before_tool_execution(
            context_name,
            tool_call.id,
            function_name,
            function_args,
            ui=ui,
        ):
            logger.info(
                f"[{tool_call.id}] [{context_name}] Tool '{function_name}' execution was prevented via callback."
            )
            function_call_result = callback_result
        else:
            # We cast here because execute_tool_call returns the LLM Protocol version of ToolResult,
            # but we know that since we passed in Framework Tool objects, it will return
            # Framework ToolResult objects.

            function_call_result = cast(ToolResult, await execute_tool_call(function_name, function_args, tools))
    except Exception as e:
        function_call_result = TextResult(content=f"Error executing tool: {e}")

    trace_data(
        f"tool_result_{function_name}.json",
        json.dumps(
            {
                "tool_call_id": tool_call.id,
                "function_name": function_name,
                "function_args": function_args,
                "result": function_call_result.to_dict(),
            },
            indent=2,
        ),
    )

    return function_call_result


async def handle_tool_calls(
    message: LLMMessage,
    tools: Sequence[Tool],
    history: list[LLMMessage],
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    *,
    ui: UI,
    context_name: str,
    task_created_callback: Callable[[str, asyncio.Task], None] | None = None,
    handle_tool_result: Callable[[ToolResult], str] | None = None,
):
    if isinstance(message, AssistantMessage):
        tool_calls = message.tool_calls
    else:
        tool_calls = []

    if not tool_calls:
        return

    tasks_with_calls = []
    loop = asyncio.get_running_loop()
    for tool_call in tool_calls:
        task = loop.create_task(
            handle_tool_call(
                tool_call,
                tools,
                history,
                progress_callbacks,
                tool_callbacks,
                ui=ui,
                context_name=context_name,
            ),
            name=f"{tool_call.function.name} ({tool_call.id})",
        )
        if task_created_callback is not None:
            task_created_callback(tool_call.id, task)
        tasks_with_calls.append((tool_call, task))

    done, pending = await asyncio.wait([task for _, task in tasks_with_calls])
    assert len(pending) == 0

    # Process results and append tool messages
    any_cancelled = False
    for tool_call, task in tasks_with_calls:
        try:
            result: ToolResult = await task
        except asyncio.CancelledError:
            # Tool was cancelled
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

        # Parse arguments from tool_call
        try:
            function_args = json.loads(tool_call.function.arguments)
        except JSONDecodeError:
            function_args = {}

        append_tool_message(
            history,
            progress_callbacks,
            context_name,
            tool_call.id,
            tool_call.function.name,
            function_args,
            result_summary,
        )

    # Re-raise CancelledError if any tool was cancelled
    if any_cancelled:
        raise asyncio.CancelledError()


async def do_single_step(
    history: list[LLMMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks,
    *,
    completer: Completer,
    context_name: str,
):
    wrapped_tools = await get_tools(tools)

    if not history:
        raise RuntimeError("History is required in order to run a step.")

    completion = await completer(
        history,
        model=model,
        tools=wrapped_tools,
        callbacks=progress_callbacks,
    )
    message = completion.message

    if isinstance(message, AssistantMessage) and message.reasoning_content:
        progress_callbacks.on_assistant_reasoning(context_name, message.reasoning_content)

    return message, completion.tokens
