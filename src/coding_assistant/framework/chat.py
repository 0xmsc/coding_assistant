from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from typing import TYPE_CHECKING

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    StatusLevel,
    Tool,
    ToolResult,
    Usage,
    UserMessage,
)
from coding_assistant.framework.execution import AgentActor, ToolCallActor
from coding_assistant.framework.history import (
    append_assistant_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.framework.types import Completer
from coding_assistant.framework.results import CompactConversationResult, TextResult
from coding_assistant.ui import UI
from coding_assistant.framework.image import get_image

if TYPE_CHECKING:
    from coding_assistant.framework.system_actors import SystemActors

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
    *,
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
            callbacks=callbacks,
            context_name=context_name,
            message=compact_result_msg,
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


@dataclass(slots=True)
class _RunChatLoop:
    history: list[BaseMessage]
    model: str
    tools: list[Tool]
    instructions: str | None
    callbacks: ProgressCallbacks
    completer: Completer
    context_name: str


class ChatLoopActor:
    def __init__(
        self,
        *,
        user_actor: UI,
        tool_call_actor: ToolCallActor,
        agent_actor: AgentActor,
        context_name: str = "chat",
    ) -> None:
        self._user_actor = user_actor
        self._tool_call_actor = tool_call_actor
        self._agent_actor = agent_actor
        self._actor: Actor[_RunChatLoop, None] = Actor(name=f"{context_name}.chat-loop", handler=self._handle_message)
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

    async def run_chat_loop(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tools: list[Tool],
        instructions: str | None,
        callbacks: ProgressCallbacks,
        completer: Completer,
        context_name: str,
    ) -> None:
        self.start()
        await self._actor.ask(
            _RunChatLoop(
                history=history,
                model=model,
                tools=tools,
                instructions=instructions,
                callbacks=callbacks,
                completer=completer,
                context_name=context_name,
            )
        )

    async def _handle_message(self, message: _RunChatLoop) -> None:
        if not isinstance(message, _RunChatLoop):
            raise RuntimeError(f"Unknown chat loop message: {message!r}")
        await self._run_chat_loop_impl(message)

    async def _run_chat_loop_impl(self, message: _RunChatLoop) -> None:
        tools = list(message.tools)
        if not any(tool.name() == "compact_conversation" for tool in tools):
            tools.append(CompactConversationTool())

        if message.history:
            for entry in message.history:
                if isinstance(entry, AssistantMessage):
                    message.callbacks.on_assistant_message(message.context_name, entry, force=True)
                elif isinstance(entry, UserMessage):
                    message.callbacks.on_user_message(message.context_name, entry, force=True)

        need_user_input = True

        async def _exit_cmd(arg: str | None = None) -> ChatCommandResult:
            return ChatCommandResult.EXIT

        async def _compact_cmd(arg: str | None = None) -> ChatCommandResult:
            compact_msg = UserMessage(
                content="Immediately compact our conversation so far by using the `compact_conversation` tool."
            )
            append_user_message(
                message.history,
                callbacks=message.callbacks,
                context_name=message.context_name,
                message=compact_msg,
                force=True,
            )

            nonlocal need_user_input
            need_user_input = True

            return ChatCommandResult.PROCEED_WITH_MODEL

        async def _clear_cmd(arg: str | None = None) -> ChatCommandResult:
            clear_history(message.history)
            message.callbacks.on_status_message("History cleared.", level=StatusLevel.SUCCESS)
            return ChatCommandResult.PROCEED_WITH_PROMPT

        async def _image_cmd(arg: str | None = None) -> ChatCommandResult:
            if arg is None:
                message.callbacks.on_status_message("/image requires a path or URL argument.", level=StatusLevel.ERROR)
                return ChatCommandResult.PROCEED_WITH_PROMPT
            try:
                data_url = await get_image(arg.strip())
                image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                user_msg = UserMessage(content=image_content)
                append_user_message(
                    message.history,
                    callbacks=message.callbacks,
                    context_name=message.context_name,
                    message=user_msg,
                )
                message.callbacks.on_status_message(f"Image added from {arg}.", level=StatusLevel.SUCCESS)
                return ChatCommandResult.PROCEED_WITH_PROMPT
            except Exception as e:
                message.callbacks.on_status_message(f"Error loading image: {e}", level=StatusLevel.ERROR)
                return ChatCommandResult.PROCEED_WITH_PROMPT

        async def _help_cmd(arg: str | None = None) -> ChatCommandResult:
            help_text = "Available commands:\n" + "\n".join(f"  {cmd.name} - {cmd.help}" for cmd in commands)
            message.callbacks.on_status_message(help_text, level=StatusLevel.INFO)
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

        start_message = _create_chat_start_message(message.instructions)
        start_user_msg = UserMessage(content=start_message)
        append_user_message(
            message.history,
            callbacks=message.callbacks,
            context_name=message.context_name,
            message=start_user_msg,
            force=True,
        )

        usage = Usage(0, 0.0)

        while True:
            if need_user_input:
                need_user_input = False

                message.callbacks.on_status_message(
                    f"💰 {usage.tokens} tokens • ${usage.cost:.2f}", level=StatusLevel.INFO
                )
                answer = await self._user_actor.prompt(words=command_names)
                answer_strip = answer.strip()

                parts = answer_strip.split(maxsplit=1)
                cmd = parts[0]
                arg = parts[1] if len(parts) > 1 else None
                if tool := command_map.get(cmd):
                    result = await tool.execute(arg)
                    if result == ChatCommandResult.EXIT:
                        break
                    if result == ChatCommandResult.PROCEED_WITH_PROMPT:
                        need_user_input = True
                        continue
                else:
                    user_msg = UserMessage(content=answer)
                    append_user_message(
                        message.history,
                        callbacks=message.callbacks,
                        context_name=message.context_name,
                        message=user_msg,
                    )

            loop = asyncio.get_running_loop()
            with InterruptController(loop) as interrupt_controller:
                try:
                    do_single_step_task = loop.create_task(
                        self._agent_actor.do_single_step(
                            history=message.history,
                            model=message.model,
                            tools=tools,
                            progress_callbacks=message.callbacks,
                            completer=message.completer,
                            context_name=message.context_name,
                        ),
                        name="do_single_step",
                    )
                    interrupt_controller.register_task("do_single_step", do_single_step_task)

                    assistant_message, step_usage = await do_single_step_task
                    append_assistant_message(
                        message.history,
                        callbacks=message.callbacks,
                        context_name=message.context_name,
                        message=assistant_message,
                    )

                    if step_usage:
                        usage = Usage(tokens=step_usage.tokens, cost=usage.cost + step_usage.cost)

                    if getattr(assistant_message, "tool_calls", []):
                        await self._tool_call_actor.handle_tool_calls(
                            assistant_message,
                            history=message.history,
                            task_created_callback=interrupt_controller.register_task,
                            handle_tool_result=lambda result: handle_tool_result_chat(
                                result,
                                history=message.history,
                                callbacks=message.callbacks,
                                context_name=message.context_name,
                            ),
                        )
                    else:
                        need_user_input = True
                except asyncio.CancelledError:
                    need_user_input = True


async def run_chat_loop(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    completer: Completer,
    ui: UI,
    context_name: str,
    system_actors: SystemActors | None = None,
) -> None:
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())
    if system_actors is None:
        from coding_assistant.framework.system_actors import system_actor_scope

        async with system_actor_scope(
            tools=tools_with_meta,
            ui=ui,
            progress_callbacks=callbacks,
            tool_callbacks=tool_callbacks,
            context_name=context_name,
        ) as actors:
            await actors.run_chat_loop(
                history=history,
                model=model,
                tools=tools_with_meta,
                instructions=instructions,
                callbacks=callbacks,
                completer=completer,
                context_name=context_name,
            )
        return

    await system_actors.run_chat_loop(
        history=history,
        model=model,
        tools=tools_with_meta,
        instructions=instructions,
        callbacks=callbacks,
        completer=completer,
        context_name=context_name,
    )
