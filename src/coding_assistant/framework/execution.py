import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from coding_assistant.framework.actor_runtime import Actor, ResponseChannel
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.chat import (
    ChatCommand,
    ChatCommandResult,
    _create_chat_start_message,
    handle_tool_result_chat,
)
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.history import (
    append_assistant_message,
    append_tool_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.parameters import format_parameters
from coding_assistant.framework.tool_executor import ToolExecutor
from coding_assistant.framework.types import (
    AgentContext,
    AgentDescription,
    AgentOutput,
    AgentState,
    Completer,
)
from coding_assistant.framework.image import get_image
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, StatusLevel
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, ToolMessage, ToolResult, Usage, UserMessage
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult, TextResult
from coding_assistant.ui import UI


@dataclass(slots=True)
class _HandleToolCalls:
    message: AssistantMessage
    history: list[BaseMessage]
    task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None
    handle_tool_result: Callable[[ToolResult], str] | None
    response_channel: ResponseChannel[None]


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
        self._actor: Actor[_HandleToolCalls] = Actor(name=f"{context_name}.tool-calls", handler=self._handle_message)
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
        response_channel: ResponseChannel[None] = ResponseChannel()
        await self._actor.send(
            _HandleToolCalls(
                message=message,
                history=history,
                task_created_callback=task_created_callback,
                handle_tool_result=handle_tool_result,
                response_channel=response_channel,
            )
        )
        await response_channel.wait()

    async def _handle_message(self, message: _HandleToolCalls) -> None:
        if not isinstance(message, _HandleToolCalls):
            raise RuntimeError(f"Unknown tool call message: {message!r}")
        await self._handle_tool_calls(
            message.message,
            history=message.history,
            task_created_callback=message.task_created_callback,
            handle_tool_result=message.handle_tool_result,
        )
        message.response_channel.send(None)

    async def _handle_tool_calls(
        self,
        message: AssistantMessage,
        *,
        history: list[BaseMessage],
        task_created_callback: Callable[[str, asyncio.Task[Any]], None] | None = None,
        handle_tool_result: Callable[[ToolResult], str] | None = None,
    ) -> None:
        tool_calls = message.tool_calls

        if not tool_calls:
            return

        tasks_with_calls = {}
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


async def do_single_step(
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    completer: Completer,
    context_name: str,
    agent_actor: "AgentActor",
) -> tuple[AssistantMessage, Usage | None]:
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
    response_channel: ResponseChannel[tuple[AssistantMessage, Usage | None]]


@dataclass(slots=True)
class _RunAgentLoop:
    ctx: AgentContext
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    compact_conversation_at_tokens: int
    tool_call_actor: ToolCallActor
    response_channel: ResponseChannel[None]


@dataclass(slots=True)
class _RunChatLoop:
    model: str
    tools: list[Tool]
    instructions: str | None
    callbacks: ProgressCallbacks
    completer: Completer
    context_name: str
    user_actor: UI
    tool_call_actor: ToolCallActor
    response_channel: ResponseChannel[None]


_AgentMessage = _DoSingleStep | _RunAgentLoop | _RunChatLoop


START_MESSAGE_TEMPLATE = """
## General

- You are an agent named `{name}`.
- You are given a set of parameters by your client, among which are your task and your description.
  - It is of the utmost importance that you try your best to fulfill the task as specified.
  - The task shall be done in a way which fits your description.
- You must use at least one tool call in every step.
  - Use the `finish_task` tool when you have fully finished your task, no questions should still be open.

## Parameters

Your client has provided the following parameters for your task:

{parameters}
""".strip()


def _create_start_message(*, desc: AgentDescription) -> str:
    parameters_str = format_parameters(desc.parameters)
    message = START_MESSAGE_TEMPLATE.format(
        name=desc.name,
        parameters=parameters_str,
    )

    return message


class AgentActor:
    def __init__(self, *, context_name: str = "agent") -> None:
        self._actor: Actor[_AgentMessage] = Actor(name=f"{context_name}.agent-loop", handler=self._handle_message)
        self._started = False
        self._chat_history: list[BaseMessage] = []
        self._agent_histories: dict[int, list[BaseMessage]] = {}

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
        response_channel: ResponseChannel[tuple[AssistantMessage, Usage | None]] = ResponseChannel()
        await self._actor.send(
            _DoSingleStep(
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                context_name=context_name,
                response_channel=response_channel,
            )
        )
        return await response_channel.wait()

    async def run_agent_loop(
        self,
        ctx: AgentContext,
        *,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
        compact_conversation_at_tokens: int,
        tool_call_actor: ToolCallActor,
    ) -> None:
        self.start()
        state_id = id(ctx.state)
        self._agent_histories[state_id] = list(ctx.state.history)
        response_channel: ResponseChannel[None] = ResponseChannel()
        await self._actor.send(
            _RunAgentLoop(
                ctx=ctx,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
                tool_call_actor=tool_call_actor,
                response_channel=response_channel,
            )
        )
        await response_channel.wait()

    async def run_chat_loop(
        self,
        *,
        model: str,
        tools: list[Tool],
        instructions: str | None,
        callbacks: ProgressCallbacks,
        completer: Completer,
        context_name: str,
        user_actor: UI,
        tool_call_actor: ToolCallActor,
    ) -> None:
        self.start()
        response_channel: ResponseChannel[None] = ResponseChannel()
        await self._actor.send(
            _RunChatLoop(
                model=model,
                tools=tools,
                instructions=instructions,
                callbacks=callbacks,
                completer=completer,
                context_name=context_name,
                user_actor=user_actor,
                tool_call_actor=tool_call_actor,
                response_channel=response_channel,
            )
        )
        await response_channel.wait()

    async def set_history(self, history: list[BaseMessage]) -> None:
        self._chat_history = list(history)

    async def get_history(self) -> list[BaseMessage]:
        return list(self._chat_history)

    async def get_agent_history(self, state_id: int) -> list[BaseMessage]:
        return list(self._agent_histories.get(state_id, []))

    async def _handle_message(self, message: _AgentMessage) -> None:
        if isinstance(message, _DoSingleStep):
            result = await _do_single_step_impl(
                history=message.history,
                model=message.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
                context_name=message.context_name,
            )
            message.response_channel.send(result)
            return None
        if isinstance(message, _RunAgentLoop):
            await self._run_agent_loop_impl(message)
            message.response_channel.send(None)
            return None
        if isinstance(message, _RunChatLoop):
            await self._run_chat_loop_impl(message)
            message.response_channel.send(None)
            return None
        raise RuntimeError(f"Unknown agent message: {message!r}")

    @staticmethod
    def handle_finish_task_result(result: FinishTaskResult, *, state: AgentState) -> str:
        state.output = AgentOutput(result=result.result, summary=result.summary)
        return "Agent output set."

    @staticmethod
    def handle_compact_conversation_result(
        result: CompactConversationResult,
        *,
        desc: AgentDescription,
        state: AgentState,
        progress_callbacks: ProgressCallbacks,
        history: list[BaseMessage] | None = None,
    ) -> str:
        target_history = history if history is not None else state.history
        clear_history(target_history)

        user_msg = UserMessage(
            content=(
                "A summary of your conversation with the client until now:\n\n"
                f"{result.summary}\n\nPlease continue your work."
            )
        )
        append_user_message(
            target_history,
            callbacks=progress_callbacks,
            context_name=desc.name,
            message=user_msg,
            force=True,
        )

        return "Conversation compacted and history reset."

    @classmethod
    def handle_tool_result_agent(
        cls,
        result: ToolResult,
        *,
        desc: AgentDescription,
        state: AgentState,
        progress_callbacks: ProgressCallbacks,
        history: list[BaseMessage] | None = None,
    ) -> str:
        if isinstance(result, FinishTaskResult):
            return cls.handle_finish_task_result(result, state=state)
        if isinstance(result, CompactConversationResult):
            return cls.handle_compact_conversation_result(
                result,
                desc=desc,
                state=state,
                progress_callbacks=progress_callbacks,
                history=history,
            )
        if isinstance(result, TextResult):
            return result.content
        return f"Tool produced result of type {type(result).__name__}"

    async def _run_agent_loop_impl(self, message: _RunAgentLoop) -> None:
        ctx = message.ctx
        desc = ctx.desc
        state = ctx.state
        state_id = id(state)
        if state_id not in self._agent_histories:
            self._agent_histories[state_id] = list(state.history)
        history = self._agent_histories[state_id]

        if state.output is not None:
            raise RuntimeError("Agent already has a result or summary.")

        start_message = _create_start_message(desc=desc)
        user_msg = UserMessage(content=start_message)
        append_user_message(
            history,
            callbacks=message.progress_callbacks,
            context_name=desc.name,
            message=user_msg,
        )

        while state.output is None:
            assistant_message, usage = await _do_single_step_impl(
                history=history,
                model=desc.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
                context_name=desc.name,
            )

            append_assistant_message(
                history,
                callbacks=message.progress_callbacks,
                context_name=desc.name,
                message=assistant_message,
            )

            if getattr(assistant_message, "tool_calls", []):
                await message.tool_call_actor.handle_tool_calls(
                    assistant_message,
                    history=history,
                    handle_tool_result=lambda result: self.handle_tool_result_agent(
                        result,
                        desc=desc,
                        state=state,
                        progress_callbacks=message.progress_callbacks,
                        history=history,
                    ),
                )
            else:
                user_msg2 = UserMessage(
                    content=(
                        "I detected a step from you without any tool calls. This is not allowed. "
                        "If you are done with your task, please call the `finish_task` tool to signal "
                        "that you are done. Otherwise, continue your work."
                    )
                )
                append_user_message(
                    history,
                    callbacks=message.progress_callbacks,
                    context_name=desc.name,
                    message=user_msg2,
                )

            if usage is not None and usage.tokens > message.compact_conversation_at_tokens:
                user_msg3 = UserMessage(
                    content=(
                        "Your conversation history has grown too large. "
                        "Compact it immediately by using the `compact_conversation` tool."
                    )
                )
                append_user_message(
                    history,
                    callbacks=message.progress_callbacks,
                    context_name=desc.name,
                    message=user_msg3,
                )

        assert state.output is not None

    async def _run_chat_loop_impl(self, message: _RunChatLoop) -> None:
        tools = list(message.tools)
        if not any(tool.name() == "compact_conversation" for tool in tools):
            tools.append(CompactConversationTool())

        if self._chat_history:
            for entry in self._chat_history:
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
                self._chat_history,
                callbacks=message.callbacks,
                context_name=message.context_name,
                message=compact_msg,
                force=True,
            )

            nonlocal need_user_input
            need_user_input = True

            return ChatCommandResult.PROCEED_WITH_MODEL

        async def _clear_cmd(arg: str | None = None) -> ChatCommandResult:
            clear_history(self._chat_history)
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
                    self._chat_history,
                    callbacks=message.callbacks,
                    context_name=message.context_name,
                    message=user_msg,
                )
                message.callbacks.on_status_message(f"Image added from {arg}.", level=StatusLevel.SUCCESS)
                return ChatCommandResult.PROCEED_WITH_PROMPT
            except Exception as exc:
                message.callbacks.on_status_message(f"Error loading image: {exc}", level=StatusLevel.ERROR)
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
            self._chat_history,
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
                answer = await message.user_actor.prompt(words=command_names)
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
                        self._chat_history,
                        callbacks=message.callbacks,
                        context_name=message.context_name,
                        message=user_msg,
                    )

            loop = asyncio.get_running_loop()
            with InterruptController(loop) as interrupt_controller:
                try:
                    do_single_step_task = loop.create_task(
                        _do_single_step_impl(
                            history=self._chat_history,
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
                        self._chat_history,
                        callbacks=message.callbacks,
                        context_name=message.context_name,
                        message=assistant_message,
                    )

                    if step_usage:
                        usage = Usage(tokens=step_usage.tokens, cost=usage.cost + step_usage.cost)

                    if getattr(assistant_message, "tool_calls", []):
                        await message.tool_call_actor.handle_tool_calls(
                            assistant_message,
                            history=self._chat_history,
                            task_created_callback=interrupt_controller.register_task,
                            handle_tool_result=lambda result: handle_tool_result_chat(
                                result,
                                history=self._chat_history,
                                callbacks=message.callbacks,
                                context_name=message.context_name,
                            ),
                        )
                    else:
                        need_user_input = True
                except asyncio.CancelledError:
                    need_user_input = True


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
