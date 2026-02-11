from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.agent.formatting import format_parameters
from coding_assistant.framework.actors.common.messages import (
    CancelToolCallsRequest,
    HandleToolCallsRequest,
    HandleToolCallsResponse,
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
    RunAgentRequest,
    RunCompleted,
    RunFailed,
)
from coding_assistant.framework.history import (
    append_assistant_message,
    append_tool_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult, TextResult
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentOutput, AgentState, Completer
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    ProgressCallbacks,
    Tool,
    ToolMessage,
    ToolResult,
    Usage,
    UserMessage,
)


@dataclass(slots=True)
class _DoSingleStep:
    request_id: str
    history: list[BaseMessage]
    model: str
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    context_name: str


_AgentMessage = (
    _DoSingleStep | RunAgentRequest | RunCompleted | RunFailed | LLMCompleteStepResponse | HandleToolCallsResponse
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
    def __init__(
        self,
        *,
        actor_directory: ActorDirectory,
        self_uri: str,
        llm_actor_uri: str,
        context_name: str = "agent",
    ) -> None:
        self._actor: Actor[_AgentMessage] = Actor(name=f"{context_name}.agent-loop", handler=self._handle_message)
        self._actor_directory = actor_directory
        self._self_uri = self_uri
        self._llm_actor_uri = llm_actor_uri
        self._started = False
        self._agent_histories: dict[int, list[BaseMessage]] = {}
        self._inflight: set[asyncio.Task[None]] = set()

        self._tool_call_actor_uri: str | None = None

        self._pending_api: dict[str, asyncio.Future[object]] = {}
        self._pending_llm: dict[str, asyncio.Future[tuple[AssistantMessage, Usage | None]]] = {}
        self._pending_tool: dict[str, asyncio.Future[HandleToolCallsResponse]] = {}
        self._run_agent_runtime: dict[str, tuple[ProgressCallbacks, Completer]] = {}

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
        self._run_agent_runtime.clear()
        self._tool_call_actor_uri = None

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
        tool_call_actor_uri: str,
    ) -> None:
        self.start()
        self._tool_call_actor_uri = tool_call_actor_uri
        state_id = id(ctx.state)
        self._agent_histories[state_id] = list(ctx.state.history)

        request_id = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[object] = loop.create_future()
        self._pending_api[request_id] = fut
        self._run_agent_runtime[request_id] = (progress_callbacks, completer)
        await self.send_message(
            RunAgentRequest(
                request_id=request_id,
                ctx=ctx,
                tools=tools,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
            )
        )
        await fut
        ctx.state.history = list(self._agent_histories.get(state_id, []))

    async def _handle_message(self, message: _AgentMessage) -> None:
        if isinstance(message, _DoSingleStep):
            self._track_task(
                asyncio.create_task(self._run_single_step_with_response(message), name="agent-single-step")
            )
            return None
        if isinstance(message, RunAgentRequest):
            self._track_task(asyncio.create_task(self._run_agent_loop_request(message), name="agent-run-agent-loop"))
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

    async def _run_agent_loop_request(self, message: RunAgentRequest) -> None:
        runtime = self._run_agent_runtime.pop(message.request_id, None)
        if runtime is None:
            await self.send_message(
                RunFailed(request_id=message.request_id, error=RuntimeError("Missing agent runtime config."))
            )
            return
        progress_callbacks, completer = runtime
        try:
            await self._run_agent_loop_impl(message, progress_callbacks=progress_callbacks, completer=completer)
        except BaseException as exc:
            await self.send_message(RunFailed(request_id=message.request_id, error=exc))
            return
        await self.send_message(RunCompleted(request_id=message.request_id))

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
        await self._actor_directory.send_message(
            uri=self._llm_actor_uri,
            message=LLMCompleteStepRequest(
                request_id=request_id,
                history=tuple(history),
                model=model,
                tools=tools,
                completer=completer,
                progress_callbacks=progress_callbacks,
                reply_to_uri=self._self_uri,
            ),
        )
        return await fut

    async def _request_tool_calls(
        self,
        *,
        assistant_message: AssistantMessage,
        tools: Sequence[Tool],
    ) -> HandleToolCallsResponse:
        if self._tool_call_actor_uri is None:
            raise RuntimeError("AgentActor is missing ToolCallActor runtime dependency.")
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
                tools=tuple(tools),
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

    async def _run_agent_loop_impl(
        self, message: RunAgentRequest, *, progress_callbacks: ProgressCallbacks, completer: Completer
    ) -> None:
        if self._tool_call_actor_uri is None:
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
            callbacks=progress_callbacks,
            context_name=desc.name,
            message=user_msg,
        )

        while state.output is None:
            assistant_message, usage = await self._request_llm(
                history=history,
                model=desc.model,
                tools=message.tools,
                progress_callbacks=progress_callbacks,
                completer=completer,
            )

            append_assistant_message(
                history,
                callbacks=progress_callbacks,
                context_name=desc.name,
                message=assistant_message,
            )

            if getattr(assistant_message, "tool_calls", []):
                tool_call_response = await self._request_tool_calls(
                    assistant_message=assistant_message, tools=message.tools
                )
                for item in tool_call_response.results:
                    result_summary = self.handle_tool_result_agent(
                        item.result,
                        desc=desc,
                        state=state,
                        progress_callbacks=progress_callbacks,
                        history=history,
                    )
                    append_tool_message(
                        history,
                        callbacks=progress_callbacks,
                        context_name=desc.name,
                        message=ToolMessage(
                            tool_call_id=item.tool_call_id,
                            name=item.name,
                            content=result_summary,
                        ),
                        arguments=item.arguments,
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
                    callbacks=progress_callbacks,
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
                    callbacks=progress_callbacks,
                    context_name=desc.name,
                    message=user_msg3,
                )

        assert state.output is not None
