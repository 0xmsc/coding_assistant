import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, cast

from coding_assistant.framework.actor_runtime import Actor
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
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, ToolMessage, ToolResult, Usage, UserMessage
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult, TextResult
from coding_assistant.ui import UI


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
        await self._handle_tool_calls(
            message.message,
            history=message.history,
            task_created_callback=message.task_created_callback,
            handle_tool_result=message.handle_tool_result,
        )

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


@dataclass(slots=True)
class _RunAgentLoop:
    ctx: AgentContext
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    compact_conversation_at_tokens: int
    tool_call_actor: ToolCallActor


_AgentMessage = _DoSingleStep | _RunAgentLoop


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
        self._actor: Actor[_AgentMessage, Any] = Actor(name=f"{context_name}.agent-loop", handler=self._handle_message)
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
        result = await self._actor.ask(
            _DoSingleStep(
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                context_name=context_name,
            )
        )
        return cast(tuple[AssistantMessage, Usage | None], result)

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
        await self._actor.ask(
            _RunAgentLoop(
                ctx=ctx,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
                tool_call_actor=tool_call_actor,
            )
        )

    async def _handle_message(self, message: _AgentMessage) -> Any:
        if isinstance(message, _DoSingleStep):
            return await _do_single_step_impl(
                history=message.history,
                model=message.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
                context_name=message.context_name,
            )
        if isinstance(message, _RunAgentLoop):
            await self._run_agent_loop_impl(message)
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
    ) -> str:
        clear_history(state.history)

        user_msg = UserMessage(
            content=(
                "A summary of your conversation with the client until now:\n\n"
                f"{result.summary}\n\nPlease continue your work."
            )
        )
        append_user_message(
            state.history,
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
    ) -> str:
        if isinstance(result, FinishTaskResult):
            return cls.handle_finish_task_result(result, state=state)
        if isinstance(result, CompactConversationResult):
            return cls.handle_compact_conversation_result(
                result, desc=desc, state=state, progress_callbacks=progress_callbacks
            )
        if isinstance(result, TextResult):
            return result.content
        return f"Tool produced result of type {type(result).__name__}"

    async def _run_agent_loop_impl(self, message: _RunAgentLoop) -> None:
        ctx = message.ctx
        desc = ctx.desc
        state = ctx.state

        if state.output is not None:
            raise RuntimeError("Agent already has a result or summary.")

        start_message = _create_start_message(desc=desc)
        user_msg = UserMessage(content=start_message)
        append_user_message(
            state.history, callbacks=message.progress_callbacks, context_name=desc.name, message=user_msg
        )

        while state.output is None:
            assistant_message, usage = await _do_single_step_impl(
                history=state.history,
                model=desc.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
                context_name=desc.name,
            )

            append_assistant_message(
                state.history,
                callbacks=message.progress_callbacks,
                context_name=desc.name,
                message=assistant_message,
            )

            if getattr(assistant_message, "tool_calls", []):
                await message.tool_call_actor.handle_tool_calls(
                    assistant_message,
                    history=state.history,
                    handle_tool_result=lambda result: self.handle_tool_result_agent(
                        result, desc=desc, state=state, progress_callbacks=message.progress_callbacks
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
                    state.history,
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
                    state.history,
                    callbacks=message.progress_callbacks,
                    context_name=desc.name,
                    message=user_msg3,
                )

        assert state.output is not None


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
