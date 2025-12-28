import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from json import JSONDecodeError

from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.history import append_tool_message
from coding_assistant.llm.types import AssistantMessage, BaseMessage, ToolCall, Usage
from coding_assistant.framework.types import Tool, Completer
from coding_assistant.framework.results import ToolResult, TextResult
from coding_assistant.trace import trace_json
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


async def execute_tool_call(function_name: str, function_args: dict, tools: Sequence[Tool]) -> ToolResult:
    """Execute a tool call by finding the matching tool and calling its execute method."""
    for tool in tools:
        if tool.name() == function_name:
            return await tool.execute(function_args)
    raise ValueError(f"Tool {function_name} not found in agent tools.")


async def handle_tool_call(
    tool_call: ToolCall,
    tools: Sequence[Tool],
    history: list[BaseMessage],
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
            function_call_result = await execute_tool_call(function_name, function_args, tools)
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


async def handle_tool_calls(
    message: BaseMessage,
    tools: Sequence[Tool],
    history: list[BaseMessage],
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

    tasks_with_calls = {}
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

            append_tool_message(
                history,
                progress_callbacks,
                context_name,
                tool_call.id,
                tool_call.function.name,
                function_args,
                result_summary,
            )

    if any_cancelled:
        raise asyncio.CancelledError()


async def do_single_step(
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks,
    *,
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

    if isinstance(message, AssistantMessage) and message.reasoning_content:
        progress_callbacks.on_assistant_reasoning(context_name, message.reasoning_content)

    return message, completion.usage
