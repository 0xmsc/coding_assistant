import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from rich.console import Console

from coding_assistant.llm.types import BaseMessage, UserMessage, AssistantMessage, Usage

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
from coding_assistant.framework.image import get_image

logger = logging.getLogger(__name__)

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
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
) -> str:
    if isinstance(result, CompactConversationResult):
        clear_history(history)
        compact_result_msg = UserMessage(
            content=f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work."
        )
        append_user_message(
            history,
            callbacks,
            context_name,
            compact_result_msg,
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
    execute: Callable[[str | None], Awaitable[ChatCommandResult]]


async def run_chat_loop(
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    *,
    callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    completer: Completer,
    ui: UI,
    context_name: str,
) -> None:
    tools = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools):
        tools.append(CompactConversationTool())

    if history:
        for message in history:
            if isinstance(message, AssistantMessage):
                callbacks.on_assistant_message(context_name, message, force=True)
            elif isinstance(message, UserMessage):
                callbacks.on_user_message(context_name, message, force=True)

    need_user_input = True

    async def _exit_cmd(arg: str | None = None) -> ChatCommandResult:
        return ChatCommandResult.EXIT

    async def _compact_cmd(arg: str | None = None) -> ChatCommandResult:
        compact_msg = UserMessage(
            content="Immediately compact our conversation so far by using the `compact_conversation` tool."
        )
        append_user_message(
            history,
            callbacks,
            context_name,
            compact_msg,
            force=True,
        )

        nonlocal need_user_input
        need_user_input = True

        return ChatCommandResult.PROCEED_WITH_MODEL

    async def _clear_cmd(arg: str | None = None) -> ChatCommandResult:
        clear_history(history)
        print("History cleared.")
        return ChatCommandResult.PROCEED_WITH_PROMPT

    async def _image_cmd(arg: str | None = None) -> ChatCommandResult:
        if arg is None:
            logger.error("/image requires a path or URL argument.")
            return ChatCommandResult.PROCEED_WITH_PROMPT
        try:
            data_url = await get_image(arg.strip())
            image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
            user_msg = UserMessage(content=image_content)
            append_user_message(history, callbacks, context_name, user_msg)
            logger.info(f"Image added from {arg}.")
            return ChatCommandResult.PROCEED_WITH_PROMPT
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return ChatCommandResult.PROCEED_WITH_PROMPT

    async def _help_cmd(arg: str | None = None) -> ChatCommandResult:
        print("Available commands:")
        for cmd in commands:
            print(f"  {cmd.name} - {cmd.help}")
        return ChatCommandResult.PROCEED_WITH_PROMPT

    commands = [
        ChatCommand("/exit", "Exit the chat", _exit_cmd),
        ChatCommand("/compact", "Compact the conversation history", _compact_cmd),
        ChatCommand("/clear", "Clear the conversation history", _clear_cmd),
        ChatCommand("/image", "Add an image (path or URL) to history", _image_cmd),
        ChatCommand("/help", "Show this help", _help_cmd),
    ]
    command_map = {cmd.name: cmd for cmd in commands}
    command_names = list(command_map.keys())

    start_message = _create_chat_start_message(instructions)
    start_user_msg = UserMessage(content=start_message)
    append_user_message(history, callbacks, context_name, start_user_msg, force=True)

    usage = Usage(0, 0.0)

    while True:
        if need_user_input:
            need_user_input = False

            Console().print(f"💰 {usage.tokens} tokens • ${usage.cost:.2f}", justify="right")
            print()
            answer = await ui.prompt(words=command_names)
            answer_strip = answer.strip()

            # Split answer into command and argument
            parts = answer_strip.split(maxsplit=1)
            cmd = parts[0]
            arg = parts[1] if len(parts) > 1 else None
            if tool := command_map.get(cmd):
                result = await tool.execute(arg)
                if result == ChatCommandResult.EXIT:
                    break
                elif result == ChatCommandResult.PROCEED_WITH_PROMPT:
                    need_user_input = True
                    continue
                elif result == ChatCommandResult.PROCEED_WITH_MODEL:
                    pass
            else:
                user_msg = UserMessage(content=answer)
                append_user_message(history, callbacks, context_name, user_msg)

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
