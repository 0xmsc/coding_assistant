import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from rich.console import Console

from coding_assistant.llm.types import Usage

from coding_assistant.framework.builtin_tools import (
    CompactConversationTool,
)
from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.execution import do_single_step, handle_tool_calls
from coding_assistant.framework.history import (
    append_assistant_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.framework.types import (
    CompactConversationResult,
    Completer,
    TextResult,
    Tool,
    ToolResult,
)
from coding_assistant.ui import UI

CHAT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you have finished your task, reply without any tool calls to return control to the user.
  - When you want to ask the user a question, create a message without any tool calls to return control to the user.

{instructions_section}
""".strip()


def _create_chat_start_message(instructions: str | None) -> str:
    instructions_section = ""
    if instructions:
        instructions_section = f"## Instructions\n\n{instructions}"
    message = CHAT_START_MESSAGE_TEMPLATE.format(
        instructions_section=instructions_section,
    )
    return message


def handle_tool_result_chat(
    result: ToolResult,
    history: list,
    callbacks: ProgressCallbacks,
    context_name: str,
) -> str:
    if isinstance(result, CompactConversationResult):
        clear_history(history)
        append_user_message(
            history,
            callbacks,
            context_name,
            f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work.",
            force=False,
        )
        return "Conversation compacted and history reset."

    if isinstance(result, TextResult):
        return result.content

    raise RuntimeError(f"Tool produced unexpected result of type {type(result).__name__}")


class ChatCommandResult(Enum):
    PROCEED_WITH_MODEL = 1
    PROCEED_WITH_PROMPT = 2
    EXIT = 3


@dataclass
class ChatCommand:
    name: str
    help: str
    execute: Callable[[], Awaitable[ChatCommandResult]]


async def run_chat_loop(
    history: list,
    model: str,
    tools: list[Tool],
    instructions: str | None,
    *,
    callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    completer: Completer,
    ui: UI,
    context_name: str,
):
    tools = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools):
        tools.append(CompactConversationTool())

    if history:
        for message in history:
            if message.role == "assistant":
                if content := message.content:
                    callbacks.on_assistant_message(context_name, content, force=True)
            elif message.role == "user":
                if content := message.content:
                    callbacks.on_user_message(context_name, content, force=True)

    need_user_input = True

    async def _exit_cmd():
        return ChatCommandResult.EXIT

    async def _compact_cmd():
        append_user_message(
            history,
            callbacks,
            context_name,
            "Immediately compact our conversation so far by using the `compact_conversation` tool.",
            force=True,
        )

        nonlocal need_user_input
        need_user_input = True

        return ChatCommandResult.PROCEED_WITH_MODEL

    async def _clear_cmd():
        clear_history(history)
        print("History cleared.")
        return ChatCommandResult.PROCEED_WITH_PROMPT

    commands = [
        ChatCommand("/exit", "Exit the chat", _exit_cmd),
        ChatCommand("/compact", "Compact the conversation history", _compact_cmd),
        ChatCommand("/clear", "Clear the conversation history", _clear_cmd),
    ]
    command_map = {cmd.name: cmd for cmd in commands}
    command_names = list(command_map.keys())

    start_message = _create_chat_start_message(instructions)
    append_user_message(history, callbacks, context_name, start_message, force=True)

    usage = Usage(0, 0.0)

    while True:
        if need_user_input:
            need_user_input = False

            Console().print(f"💰 {usage.tokens} tokens • ${usage.cost:.2f}", justify="right")
            print()
            answer = await ui.prompt(words=command_names)
            answer_strip = answer.strip()

            if tool := command_map.get(answer_strip):
                result = await tool.execute()
                if result == ChatCommandResult.EXIT:
                    break
                elif result == ChatCommandResult.PROCEED_WITH_PROMPT:
                    need_user_input = True
                    continue
                elif result == ChatCommandResult.PROCEED_WITH_MODEL:
                    pass
            else:
                append_user_message(history, callbacks, context_name, answer)

        loop = asyncio.get_running_loop()
        with InterruptController(loop) as interrupt_controller:
            try:
                do_single_step_task = loop.create_task(
                    do_single_step(
                        history,
                        model,
                        tools,
                        callbacks,
                        completer=completer,
                        context_name=context_name,
                    ),
                    name="do_single_step",
                )
                interrupt_controller.register_task("do_single_step", do_single_step_task)

                message, step_usage = await do_single_step_task
                append_assistant_message(history, callbacks, context_name, message)

                if step_usage:
                    usage = Usage(
                        tokens=step_usage.tokens,
                        cost=usage.cost + step_usage.cost,
                    )

                if getattr(message, "tool_calls", []):
                    await handle_tool_calls(
                        message,
                        tools,
                        history,
                        callbacks,
                        tool_callbacks,
                        ui=ui,
                        context_name=context_name,
                        task_created_callback=interrupt_controller.register_task,
                        handle_tool_result=lambda result: handle_tool_result_chat(
                            result, history, callbacks, context_name
                        ),
                    )
                else:
                    need_user_input = True
            except asyncio.CancelledError:
                need_user_input = True
