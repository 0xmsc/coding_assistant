from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.agent.chat_policy import create_chat_start_message, handle_tool_result_chat
from coding_assistant.framework.actors.agent.formatting import format_parameters
from coding_assistant.framework.actors.agent.image_io import get_image
from coding_assistant.framework.actors.common.contracts import MessageSink
from coding_assistant.framework.actors.common.messages import (
    AgentYieldedToUser,
    ChatPromptInput,
    ClearHistoryRequested,
    CompactionRequested,
    HandleToolCallsRequest,
    HandleToolCallsResponse,
    HelpRequested,
    ImageAttachRequested,
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
    SessionExitRequested,
    UserInputFailed,
    UserTextSubmitted,
)
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.history import append_assistant_message, append_user_message, clear_history
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult, TextResult
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentOutput, AgentState, Completer
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    ProgressCallbacks,
    StatusLevel,
    Tool,
    ToolResult,
    Usage,
    UserMessage,
)
from coding_assistant.ui import UI


@dataclass(slots=True)
class _DoSingleStep:
    request_id: str
    history: list[BaseMessage]
    model: str
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    context_name: str


@dataclass(slots=True)
class _RunAgentLoop:
    request_id: str
    ctx: AgentContext
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    compact_conversation_at_tokens: int


@dataclass(slots=True)
class _RunChatLoop:
    request_id: str
    model: str
    tools: list[Tool]
    instructions: str | None
    callbacks: ProgressCallbacks
    completer: Completer
    context_name: str


_AgentMessage = (
    _DoSingleStep | _RunAgentLoop | _RunChatLoop | LLMCompleteStepResponse | HandleToolCallsResponse | ChatPromptInput
)

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
    def __init__(self, *, llm_gateway: MessageSink[LLMCompleteStepRequest], context_name: str = "agent") -> None:
        self._actor: Actor[_AgentMessage] = Actor(name=f"{context_name}.agent-loop", handler=self._handle_message)
        self._llm_gateway = llm_gateway
        self._started = False
        self._chat_history: list[BaseMessage] = []
        self._agent_histories: dict[int, list[BaseMessage]] = {}
        self._inflight: set[asyncio.Task[None]] = set()

        self._tool_call_sink: MessageSink[HandleToolCallsRequest] | None = None
        self._user_sink: MessageSink[AgentYieldedToUser] | None = None

        self._pending_api: dict[str, asyncio.Future[object]] = {}
        self._pending_llm: dict[str, asyncio.Future[tuple[AssistantMessage, Usage | None]]] = {}
        self._pending_tool: dict[str, asyncio.Future[None]] = {}
        self._pending_chat_input: asyncio.Future[ChatPromptInput] | None = None
        self._queued_chat_inputs: list[ChatPromptInput] = []

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        if self._inflight:
            for task in list(self._inflight):
                task.cancel()
            await asyncio.gather(*list(self._inflight), return_exceptions=True)
            self._inflight.clear()
        self._cancel_pending("AgentActor stopped")
        self._started = False

    async def send_message(self, message: _AgentMessage) -> None:
        self.start()
        await self._actor.send(message)

    def _next_id(self) -> str:
        return uuid4().hex

    def _cancel_pending(self, reason: str) -> None:
        for mapping in (self._pending_api, self._pending_llm, self._pending_tool):
            for fut in mapping.values():
                if not fut.done():
                    fut.set_exception(RuntimeError(reason))
            mapping.clear()
        if self._pending_chat_input is not None and not self._pending_chat_input.done():
            self._pending_chat_input.set_exception(RuntimeError(reason))
        self._pending_chat_input = None
        self._queued_chat_inputs.clear()

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
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        self._pending_api[request_id] = fut
        await self.send_message(
            _DoSingleStep(
                request_id=request_id,
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                context_name=context_name,
            )
        )
        result = await fut
        return result  # type: ignore[return-value]

    async def run_agent_loop(
        self,
        ctx: AgentContext,
        *,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
        compact_conversation_at_tokens: int,
        tool_call_actor: MessageSink[HandleToolCallsRequest],
    ) -> None:
        self.start()
        self._tool_call_sink = tool_call_actor
        state_id = id(ctx.state)
        self._agent_histories[state_id] = list(ctx.state.history)

        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        self._pending_api[request_id] = fut
        await self.send_message(
            _RunAgentLoop(
                request_id=request_id,
                ctx=ctx,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
            )
        )
        await fut

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
        tool_call_actor: MessageSink[HandleToolCallsRequest],
    ) -> None:
        self.start()
        if not hasattr(user_actor, "send_message"):
            raise RuntimeError("AgentActor requires a message-capable user actor.")
        self._user_sink = cast(MessageSink[AgentYieldedToUser], user_actor)
        self._tool_call_sink = tool_call_actor

        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        self._pending_api[request_id] = fut
        await self.send_message(
            _RunChatLoop(
                request_id=request_id,
                model=model,
                tools=tools,
                instructions=instructions,
                callbacks=callbacks,
                completer=completer,
                context_name=context_name,
            )
        )
        await fut

    async def set_history(self, history: list[BaseMessage]) -> None:
        self._chat_history = list(history)

    async def get_history(self) -> list[BaseMessage]:
        return list(self._chat_history)

    async def get_agent_history(self, state_id: int) -> list[BaseMessage]:
        return list(self._agent_histories.get(state_id, []))

    async def _handle_message(self, message: _AgentMessage) -> None:
        if isinstance(message, _DoSingleStep):
            self._track_task(
                asyncio.create_task(self._run_single_step_with_response(message), name="agent-single-step")
            )
            return None
        if isinstance(message, _RunAgentLoop):
            self._track_task(
                asyncio.create_task(self._run_agent_loop_with_response(message), name="agent-run-agent-loop")
            )
            return None
        if isinstance(message, _RunChatLoop):
            self._track_task(
                asyncio.create_task(self._run_chat_loop_with_response(message), name="agent-run-chat-loop")
            )
            return None
        if isinstance(message, LLMCompleteStepResponse):
            llm_future = self._pending_llm.pop(message.request_id, None)
            if llm_future is None:
                raise RuntimeError(f"Unknown LLM response id: {message.request_id}")
            if message.error is not None:
                llm_future.set_exception(message.error)
            else:
                if message.message is None:
                    llm_future.set_exception(RuntimeError("LLM response missing message."))
                else:
                    llm_future.set_result((message.message, message.usage))
            return None
        if isinstance(message, HandleToolCallsResponse):
            tool_future = self._pending_tool.pop(message.request_id, None)
            if tool_future is None:
                raise RuntimeError(f"Unknown tool-call response id: {message.request_id}")
            if message.error is not None:
                tool_future.set_exception(message.error)
            else:
                tool_future.set_result(None)
            return None
        if isinstance(message, UserInputFailed):
            if self._pending_chat_input is not None and not self._pending_chat_input.done():
                self._pending_chat_input.set_exception(message.error)
                self._pending_chat_input = None
            else:
                raise RuntimeError("Received UserInputFailed but no chat input is pending.")
            return None
        if isinstance(
            message,
            (
                UserTextSubmitted,
                SessionExitRequested,
                ClearHistoryRequested,
                CompactionRequested,
                ImageAttachRequested,
                HelpRequested,
            ),
        ):
            if self._pending_chat_input is not None and not self._pending_chat_input.done():
                self._pending_chat_input.set_result(message)
                self._pending_chat_input = None
            else:
                self._queued_chat_inputs.append(message)
            return None
        raise RuntimeError(f"Unknown agent message: {message!r}")

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _run_single_step_with_response(self, message: _DoSingleStep) -> None:
        fut = self._pending_api.pop(message.request_id, None)
        if fut is None:
            raise RuntimeError(f"Unknown single-step request id: {message.request_id}")
        try:
            result = await self._request_llm(
                history=message.history,
                model=message.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
            )
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    async def _run_agent_loop_with_response(self, message: _RunAgentLoop) -> None:
        fut = self._pending_api.pop(message.request_id, None)
        if fut is None:
            raise RuntimeError(f"Unknown agent-loop request id: {message.request_id}")
        try:
            await self._run_agent_loop_impl(message)
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(None)

    async def _run_chat_loop_with_response(self, message: _RunChatLoop) -> None:
        fut = self._pending_api.pop(message.request_id, None)
        if fut is None:
            raise RuntimeError(f"Unknown chat-loop request id: {message.request_id}")
        try:
            await self._run_chat_loop_impl(message)
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(None)

    async def _request_llm(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
    ) -> tuple[AssistantMessage, Usage | None]:
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[AssistantMessage, Usage | None]] = loop.create_future()
        self._pending_llm[request_id] = fut
        await self._llm_gateway.send_message(
            LLMCompleteStepRequest(
                request_id=request_id,
                history=history,
                model=model,
                tools=tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
                reply_to=self,
            )
        )
        return await fut

    async def _request_tool_calls(
        self,
        *,
        assistant_message: AssistantMessage,
        history: list[BaseMessage],
        task_created_callback: Callable[[str, asyncio.Task[object]], None] | None,
        handle_tool_result: Callable[[ToolResult], str] | None,
    ) -> None:
        if self._tool_call_sink is None:
            raise RuntimeError("AgentActor is missing ToolCallActor runtime dependency.")
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._pending_tool[request_id] = fut
        await self._tool_call_sink.send_message(
            HandleToolCallsRequest(
                request_id=request_id,
                message=assistant_message,
                history=history,
                task_created_callback=task_created_callback,
                handle_tool_result=handle_tool_result,
                reply_to=self,
            )
        )
        await fut

    async def _wait_for_chat_input(self, words: list[str] | None = None) -> ChatPromptInput:
        if self._user_sink is None:
            raise RuntimeError("AgentActor is missing UserActor runtime dependency.")
        if self._queued_chat_inputs:
            return self._queued_chat_inputs.pop(0)
        if self._pending_chat_input is not None and not self._pending_chat_input.done():
            raise RuntimeError("AgentActor is already waiting for chat input.")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ChatPromptInput] = loop.create_future()
        self._pending_chat_input = fut
        await self._user_sink.send_message(AgentYieldedToUser(request_id=self._next_id(), words=words, reply_to=self))
        return await fut

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
        if self._tool_call_sink is None:
            raise RuntimeError("AgentActor is missing ToolCallActor runtime dependency.")
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
            assistant_message, usage = await self._request_llm(
                history=history,
                model=desc.model,
                tools=message.tools,
                progress_callbacks=message.progress_callbacks,
                completer=message.completer,
            )

            append_assistant_message(
                history,
                callbacks=message.progress_callbacks,
                context_name=desc.name,
                message=assistant_message,
            )

            if getattr(assistant_message, "tool_calls", []):
                await self._request_tool_calls(
                    assistant_message=assistant_message,
                    history=history,
                    task_created_callback=None,
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
        if self._user_sink is None:
            raise RuntimeError("AgentActor is missing UserActor runtime dependency.")
        if self._tool_call_sink is None:
            raise RuntimeError("AgentActor is missing ToolCallActor runtime dependency.")
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
        command_names = ["/exit", "/compact", "/clear", "/image", "/help"]
        help_text = (
            "Available commands:\n"
            "  /exit - Exit the chat\n"
            "  /compact - Compact the conversation history\n"
            "  /clear - Clear the conversation history\n"
            "  /image - Add an image (path or URL) to history\n"
            "  /help - Show this help"
        )

        start_message = create_chat_start_message(message.instructions)
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
                prompt_input = await self._wait_for_chat_input(command_names)
                if isinstance(prompt_input, SessionExitRequested):
                    break
                if isinstance(prompt_input, CompactionRequested):
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
                    need_user_input = True
                elif isinstance(prompt_input, ClearHistoryRequested):
                    clear_history(self._chat_history)
                    message.callbacks.on_status_message("History cleared.", level=StatusLevel.SUCCESS)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, ImageAttachRequested):
                    if prompt_input.source is None:
                        message.callbacks.on_status_message(
                            "/image requires a path or URL argument.", level=StatusLevel.ERROR
                        )
                    else:
                        try:
                            data_url = await get_image(prompt_input.source)
                            image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                            user_msg = UserMessage(content=image_content)
                            append_user_message(
                                self._chat_history,
                                callbacks=message.callbacks,
                                context_name=message.context_name,
                                message=user_msg,
                            )
                            message.callbacks.on_status_message(
                                f"Image added from {prompt_input.source}.", level=StatusLevel.SUCCESS
                            )
                        except Exception as exc:
                            message.callbacks.on_status_message(f"Error loading image: {exc}", level=StatusLevel.ERROR)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, HelpRequested):
                    message.callbacks.on_status_message(help_text, level=StatusLevel.INFO)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, UserTextSubmitted):
                    user_msg = UserMessage(content=prompt_input.text)
                    append_user_message(
                        self._chat_history,
                        callbacks=message.callbacks,
                        context_name=message.context_name,
                        message=user_msg,
                    )
                else:
                    raise RuntimeError(f"Unsupported chat prompt input: {prompt_input!r}")

            loop = asyncio.get_running_loop()
            with InterruptController(loop) as interrupt_controller:
                try:
                    do_single_step_task = loop.create_task(
                        self._request_llm(
                            history=self._chat_history,
                            model=message.model,
                            tools=tools,
                            progress_callbacks=message.callbacks,
                            completer=message.completer,
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
                        await self._request_tool_calls(
                            assistant_message=assistant_message,
                            history=self._chat_history,
                            task_created_callback=lambda tool_id, task: interrupt_controller.register_task(
                                tool_id, task
                            ),
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
