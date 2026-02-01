import asyncio
import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

from coding_assistant.framework.callbacks import (
    NullToolCallbacks,
    ToolCallbacks,
)
from coding_assistant.ui import UI
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    ToolResult,
    Usage,
    ToolMessage,
)
from coding_assistant.actors.system import ActorSystem
from coding_assistant.framework.types import Completer
from coding_assistant.messaging.envelopes import Envelope
from coding_assistant.messaging.messages import ActorMessage, ExecuteTool, ToolResult as ToolResultMessage, Error
from coding_assistant.framework.history import append_tool_message
from coding_assistant.framework.results import TextResult, result_from_dict


logger = logging.getLogger(__name__)


async def execute_tool_call(*, function_name: str, function_args: dict[str, Any], tools: Sequence[Tool]) -> ToolResult:
    """Execute a tool call by finding the matching tool and calling its execute method."""
    for tool in tools:
        if tool.name() == function_name:
            return await tool.execute(function_args)
    raise ValueError(f"Tool {function_name} not found in agent tools.")


async def do_single_step(
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    completer: Completer,
    context_name: str,
) -> tuple[AssistantMessage, Usage | None]:
    """Legacy bridge to Actor behavior"""
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
    actor_system: ActorSystem | None = None,
) -> None:
    """Legacy bridge to Actor behavior"""
    from coding_assistant.actors.tool_worker import ToolWorkerActor

    if not message.tool_calls:
        return

    # Use existing actor system or create transient one for the call
    system = actor_system or ActorSystem()

    # We must register a worker if it doesn't exist
    worker_addr = f"transient_worker_{context_name}"
    worker = ToolWorkerActor(worker_addr, system, tools)
    try:
        system.register(worker)
    except ValueError:
        pass  # Already registered

    await worker.start()

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        args_str = tool_call.function.arguments
        res_val: ToolResult

        # 1. Handle JSON parsing errors locally (legacy behavior)
        try:
            function_args = json.loads(args_str)
        except Exception as e:
            res_val = TextResult(content=f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}")
            summary = handle_tool_result(res_val) if handle_tool_result else res_val.content
            append_tool_message(
                history,
                callbacks=progress_callbacks,
                context_name=context_name,
                message=ToolMessage(tool_call_id=tool_call.id, name=tool_name, content=summary),
                arguments={},
            )
            continue

        # 2. Progress callback
        progress_callbacks.on_tool_start(context_name, tool_call, function_args)

        # 3. Handle Denial (legacy behavior)
        deny_result = await tool_callbacks.before_tool_execution(
            context_name, tool_call.id, tool_name, function_args, ui=ui
        )

        res_val
        if deny_result:
            res_val = deny_result
        else:
            # 4. Actual execution via Actor
            env: Envelope[ActorMessage] = Envelope(
                sender="legacy_bridge",
                recipient=worker_addr,
                correlation_id=tool_call.id,
                payload=ExecuteTool(tool_name=tool_name, arguments=function_args),
            )

            response = await system.ask(env)
            if isinstance(response.payload, ToolResultMessage):
                res_val = result_from_dict(response.payload.result)
            elif isinstance(response.payload, Error):
                res_val = TextResult(content=f"Error executing tool: {response.payload.message}")
            else:
                res_val = TextResult(content=f"Error: Unexpected response type {type(response.payload)}")

        # 5. Summary and History
        if handle_tool_result:
            final_summary = handle_tool_result(res_val)
        else:
            final_summary = res_val.content if isinstance(res_val, TextResult) else str(res_val)

        tool_msg = ToolMessage(tool_call_id=tool_call.id, name=tool_name, content=final_summary)
        append_tool_message(
            history, callbacks=progress_callbacks, context_name=context_name, message=tool_msg, arguments=function_args
        )
