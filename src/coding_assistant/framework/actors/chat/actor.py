from __future__ import annotations

import asyncio
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.agent.chat_policy import create_chat_start_message, handle_tool_result_chat
from coding_assistant.framework.actors.agent.image_io import get_image
from coding_assistant.framework.actors.common.messages import (
    AgentYieldedToUser,
    CancelToolCallsRequest,
    ChatPromptInput,
    ClearHistoryRequested,
    CompactionRequested,
    HandleToolCallsRequest,
    HandleToolCallsResponse,
    HelpRequested,
    ImageAttachRequested,
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
    RunChatRequest,
    RunCompleted,
    RunFailed,
    SessionExitRequested,
    ToolCapability,
    UserInputFailed,
    UserTextSubmitted,
)
from coding_assistant.framework.history import (
    append_assistant_message,
    append_tool_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.interrupts import InterruptController
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    StatusLevel,
    ToolMessage,
    Usage,
    UserMessage,
)


_ChatMessage = (
    RunChatRequest | RunCompleted | RunFailed | LLMCompleteStepResponse | HandleToolCallsResponse | ChatPromptInput
)


class ChatActor:
    def __init__(
        self,
        *,
        actor_directory: ActorDirectory,
        self_uri: str,
        llm_actor_uri: str,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        context_name: str = "chat",
    ) -> None:
        self._actor: Actor[_ChatMessage] = Actor(name=f"{context_name}.chat-loop", handler=self._handle_message)
        self._actor_directory = actor_directory
        self._self_uri = self_uri
        self._llm_actor_uri = llm_actor_uri
        self._progress_callbacks = progress_callbacks
        self._started = False
        self._inflight: set[asyncio.Task[None]] = set()

        self._tool_call_actor_uri: str | None = None
        self._user_actor_uri: str | None = None
        self._chat_history: list[BaseMessage] = []

        self._pending_api: dict[str, asyncio.Future[object]] = {}
        self._pending_llm: dict[str, asyncio.Future[tuple[AssistantMessage, Usage | None]]] = {}
        self._pending_tool: dict[str, asyncio.Future[HandleToolCallsResponse]] = {}
        self._pending_chat_input: asyncio.Future[ChatPromptInput] | None = None
        self._queued_chat_inputs: list[ChatPromptInput] = []

    def set_progress_callbacks(self, callbacks: ProgressCallbacks) -> None:
        self._progress_callbacks = callbacks

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
        self._cancel_pending("ChatActor stopped")
        self._started = False

    async def send_message(self, message: _ChatMessage) -> None:
        self.start()
        await self._actor.send(message)

    def _next_id(self) -> str:
        return uuid4().hex

    def _track_task(self, task: asyncio.Task[None]) -> None:
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

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
        self._user_actor_uri = None
        self._tool_call_actor_uri = None

    async def run_chat_loop(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tool_capabilities: tuple[ToolCapability, ...],
        instructions: str | None,
        context_name: str,
        user_actor_uri: str,
        tool_call_actor_uri: str,
    ) -> None:
        self.start()
        self._user_actor_uri = user_actor_uri
        self._tool_call_actor_uri = tool_call_actor_uri
        self._chat_history = list(history)

        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        self._pending_api[request_id] = fut
        await self.send_message(
            RunChatRequest(
                request_id=request_id,
                history=history,
                model=model,
                tool_capabilities=tool_capabilities,
                instructions=instructions,
                context_name=context_name,
                user_actor_uri=user_actor_uri,
                tool_call_actor_uri=tool_call_actor_uri,
                reply_to_uri=self._self_uri,
            )
        )
        await fut

    async def _handle_message(self, message: _ChatMessage) -> None:
        if isinstance(message, RunChatRequest):
            self._track_task(asyncio.create_task(self._run_chat_loop_request(message), name="chat-run-chat-loop"))
            return None
        if isinstance(message, RunCompleted):
            fut = self._pending_api.pop(message.request_id, None)
            if fut is not None and not fut.done():
                fut.set_result(None)
            return None
        if isinstance(message, RunFailed):
            fut = self._pending_api.pop(message.request_id, None)
            if fut is not None and not fut.done():
                fut.set_exception(message.error)
            return None
        if isinstance(message, LLMCompleteStepResponse):
            llm_future = self._pending_llm.pop(message.request_id, None)
            if llm_future is None:
                return None
            if message.error is not None:
                if not llm_future.done():
                    llm_future.set_exception(message.error)
            else:
                if message.message is None:
                    if not llm_future.done():
                        llm_future.set_exception(RuntimeError("LLM response missing message."))
                else:
                    if not llm_future.done():
                        llm_future.set_result((message.message, message.usage))
            return None
        if isinstance(message, HandleToolCallsResponse):
            tool_future = self._pending_tool.pop(message.request_id, None)
            if tool_future is None:
                return None
            if message.error is not None:
                if not tool_future.done():
                    tool_future.set_exception(message.error)
            else:
                if not tool_future.done():
                    tool_future.set_result(message)
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
        raise RuntimeError(f"Unknown chat message: {message!r}")

    async def _run_chat_loop_request(self, message: RunChatRequest) -> None:
        self._user_actor_uri = message.user_actor_uri
        self._tool_call_actor_uri = message.tool_call_actor_uri
        self._chat_history = list(message.history)
        try:
            await self._run_chat_loop_impl(message)
        except BaseException as exc:
            message.history.clear()
            message.history.extend(self._chat_history)
            await self._send_run_response(
                request=message,
                message=RunFailed(request_id=message.request_id, error=exc),
            )
            return
        message.history.clear()
        message.history.extend(self._chat_history)
        await self._send_run_response(
            request=message,
            message=RunCompleted(request_id=message.request_id, history=tuple(self._chat_history)),
        )

    async def _send_run_response(self, *, request: RunChatRequest, message: RunCompleted | RunFailed) -> None:
        if request.reply_to_uri is None:
            await self.send_message(message)
            return
        await self._actor_directory.send_message(uri=request.reply_to_uri, message=message)

    async def _request_llm(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tool_capabilities: tuple[ToolCapability, ...],
    ) -> tuple[AssistantMessage, Usage | None]:
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[tuple[AssistantMessage, Usage | None]] = loop.create_future()
        self._pending_llm[request_id] = fut
        await self._actor_directory.send_message(
            uri=self._llm_actor_uri,
            message=LLMCompleteStepRequest(
                request_id=request_id,
                history=tuple(history),
                model=model,
                tool_capabilities=tool_capabilities,
                reply_to_uri=self._self_uri,
            ),
        )
        return await fut

    async def _request_tool_calls(
        self,
        *,
        assistant_message: AssistantMessage,
        tool_capabilities: tuple[ToolCapability, ...],
    ) -> HandleToolCallsResponse:
        if self._tool_call_actor_uri is None:
            raise RuntimeError("ChatActor is missing ToolCallActor runtime dependency.")
        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[HandleToolCallsResponse] = loop.create_future()
        self._pending_tool[request_id] = fut
        await self._actor_directory.send_message(
            uri=self._tool_call_actor_uri,
            message=HandleToolCallsRequest(
                request_id=request_id,
                message=assistant_message,
                reply_to_uri=self._self_uri,
                tool_capabilities=tool_capabilities,
            ),
        )
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            await self._actor_directory.send_message(
                uri=self._tool_call_actor_uri,
                message=CancelToolCallsRequest(request_id=request_id),
            )
            try:
                await asyncio.shield(fut)
            except BaseException:
                pass
            raise

    async def _wait_for_chat_input(self, words: list[str] | None = None) -> ChatPromptInput:
        if self._user_actor_uri is None:
            raise RuntimeError("ChatActor is missing UserActor runtime dependency.")
        if self._queued_chat_inputs:
            return self._queued_chat_inputs.pop(0)
        if self._pending_chat_input is not None and not self._pending_chat_input.done():
            raise RuntimeError("ChatActor is already waiting for chat input.")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ChatPromptInput] = loop.create_future()
        self._pending_chat_input = fut
        await self._actor_directory.send_message(
            uri=self._user_actor_uri,
            message=AgentYieldedToUser(
                request_id=self._next_id(),
                words=words,
                reply_to_uri=self._self_uri,
            ),
        )
        return await fut

    async def _run_chat_loop_impl(self, message: RunChatRequest) -> None:
        if self._user_actor_uri is None:
            raise RuntimeError("ChatActor is missing UserActor runtime dependency.")
        if self._tool_call_actor_uri is None:
            raise RuntimeError("ChatActor is missing ToolCallActor runtime dependency.")
        callbacks = self._progress_callbacks
        tool_capabilities = message.tool_capabilities
        if not any(item.name == "compact_conversation" for item in tool_capabilities):
            raise RuntimeError("ChatActor requires compact_conversation tool capability.")

        if self._chat_history:
            for entry in self._chat_history:
                if isinstance(entry, AssistantMessage):
                    callbacks.on_assistant_message(message.context_name, entry, force=True)
                elif isinstance(entry, UserMessage):
                    callbacks.on_user_message(message.context_name, entry, force=True)

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
            callbacks=callbacks,
            context_name=message.context_name,
            message=start_user_msg,
            force=True,
        )

        usage = Usage(0, 0.0)

        while True:
            if need_user_input:
                need_user_input = False

                callbacks.on_status_message(f"💰 {usage.tokens} tokens • ${usage.cost:.2f}", level=StatusLevel.INFO)
                prompt_input = await self._wait_for_chat_input(command_names)
                if isinstance(prompt_input, SessionExitRequested):
                    break
                if isinstance(prompt_input, CompactionRequested):
                    compact_msg = UserMessage(
                        content="Immediately compact our conversation so far by using the `compact_conversation` tool."
                    )
                    append_user_message(
                        self._chat_history,
                        callbacks=callbacks,
                        context_name=message.context_name,
                        message=compact_msg,
                        force=True,
                    )
                    need_user_input = True
                elif isinstance(prompt_input, ClearHistoryRequested):
                    clear_history(self._chat_history)
                    callbacks.on_status_message("History cleared.", level=StatusLevel.SUCCESS)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, ImageAttachRequested):
                    if prompt_input.source is None:
                        callbacks.on_status_message("/image requires a path or URL argument.", level=StatusLevel.ERROR)
                    else:
                        try:
                            data_url = await get_image(prompt_input.source)
                            image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                            user_msg = UserMessage(content=image_content)
                            append_user_message(
                                self._chat_history,
                                callbacks=callbacks,
                                context_name=message.context_name,
                                message=user_msg,
                            )
                            callbacks.on_status_message(
                                f"Image added from {prompt_input.source}.", level=StatusLevel.SUCCESS
                            )
                        except Exception as exc:
                            callbacks.on_status_message(f"Error loading image: {exc}", level=StatusLevel.ERROR)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, HelpRequested):
                    callbacks.on_status_message(help_text, level=StatusLevel.INFO)
                    need_user_input = True
                    continue
                elif isinstance(prompt_input, UserTextSubmitted):
                    user_msg = UserMessage(content=prompt_input.text)
                    append_user_message(
                        self._chat_history,
                        callbacks=callbacks,
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
                            tool_capabilities=tool_capabilities,
                        ),
                        name="do_single_step",
                    )
                    interrupt_controller.register_task("do_single_step", do_single_step_task)

                    assistant_message, step_usage = await do_single_step_task
                    append_assistant_message(
                        self._chat_history,
                        callbacks=callbacks,
                        context_name=message.context_name,
                        message=assistant_message,
                    )

                    if step_usage:
                        usage = Usage(tokens=step_usage.tokens, cost=usage.cost + step_usage.cost)

                    if getattr(assistant_message, "tool_calls", []):
                        handle_tool_calls_task = loop.create_task(
                            self._request_tool_calls(
                                assistant_message=assistant_message, tool_capabilities=tool_capabilities
                            ),
                            name="tool_calls",
                        )
                        interrupt_controller.register_task("tool_calls", handle_tool_calls_task)
                        tool_call_response = await handle_tool_calls_task
                        for item in tool_call_response.results:
                            result_summary = handle_tool_result_chat(
                                item.result,
                                history=self._chat_history,
                                callbacks=callbacks,
                                context_name=message.context_name,
                            )
                            append_tool_message(
                                self._chat_history,
                                callbacks=callbacks,
                                context_name=message.context_name,
                                message=ToolMessage(
                                    tool_call_id=item.tool_call_id,
                                    name=item.name,
                                    content=result_summary,
                                ),
                                arguments=item.arguments,
                            )
                        if tool_call_response.cancelled:
                            raise asyncio.CancelledError()
                    else:
                        need_user_input = True
                except asyncio.CancelledError:
                    need_user_input = True
