import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, AsyncIterator, Iterable, Sequence
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.history import append_tool_message
from coding_assistant.framework.actors.common.messages import (
    HandleToolCallsRequest,
    HandleToolCallsResponse,
    RunAgentRequest,
    RunChatRequest,
    ToolCapability,
    ToolCallExecutionResult,
)
from coding_assistant.framework.actors.common.reply_waiters import register_run_reply_waiter
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.chat.actor import ChatActor
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.framework.actors.tool_call.actor import ToolCallActor
from coding_assistant.framework.actors.tool_call.capabilities import ToolCapabilityActor, register_tool_capabilities
from coding_assistant.framework.parameters import Parameter
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import AgentDescription, AgentState, AgentContext, Completer
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    FunctionCall as FunctionCall,
    Tool,
    ToolCall as ToolCall,
    ToolMessage,
    ToolResult,
    Usage,
)
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.ui import ActorUI, UI, UserActor


def FakeMessage(
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    reasoning_content: str | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning_content,
    )


class FakeCompleter:
    def __init__(self, script: Iterable[AssistantMessage | Exception]) -> None:
        self.script: list[AssistantMessage | Exception] = list(script)
        self._total_tokens = 0

    async def __call__(self, messages: Any, *, model: Any, tools: Any, callbacks: Any) -> Completion:
        if hasattr(self, "before_completion") and callable(self.before_completion):
            await self.before_completion()

        if not self.script:
            raise AssertionError("FakeCompleter script exhausted")

        action = self.script.pop(0)

        if isinstance(action, Exception):
            raise action

        # Simple mockup for token calculation
        text = str(action)
        toks = len(text)
        self._total_tokens += toks

        usage = Usage(tokens=self._total_tokens, cost=0.0)
        return Completion(message=action, usage=usage)


def make_ui_mock(
    *,
    ask_sequence: list[tuple[str, str]] | None = None,
    confirm_sequence: list[tuple[str, bool]] | None = None,
) -> UI:
    ui = Mock()

    # Use local copies so tests can inspect remaining expectations after calls if needed
    ask_seq = list(ask_sequence) if ask_sequence is not None else None
    confirm_seq = list(confirm_sequence) if confirm_sequence is not None else None

    async def _ask(prompt_text: str, default: str | None = None) -> str:
        assert ask_seq is not None, "UI.ask was called but no ask_sequence was provided"
        assert len(ask_seq) > 0, "UI.ask was called more times than expected"
        expected_prompt, value = ask_seq.pop(0)
        assert prompt_text == expected_prompt, f"Unexpected ask prompt. Expected: {expected_prompt}, got: {prompt_text}"
        return value

    async def _confirm(prompt_text: str) -> bool:
        assert confirm_seq is not None, "UI.confirm was called but no confirm_sequence was provided"
        assert len(confirm_seq) > 0, "UI.confirm was called more times than expected"
        expected_prompt, value = confirm_seq.pop(0)
        assert prompt_text == expected_prompt, (
            f"Unexpected confirm prompt. Expected: {expected_prompt}, got: {prompt_text}"
        )
        return bool(value)

    ui.ask = AsyncMock(side_effect=_ask)
    ui.confirm = AsyncMock(side_effect=_confirm)

    async def _prompt(words: list[str] | None = None) -> str:
        # In chat mode, prompt uses a generic '> ' prompt
        return await _ask("> ", None)

    ui.prompt = AsyncMock(side_effect=_prompt)

    # Expose remaining expectations for introspection in tests (optional)
    ui._remaining_ask_expectations = ask_seq
    ui._remaining_confirm_expectations = confirm_seq

    return ui


def make_test_agent(
    *,
    name: str = "TestAgent",
    model: str = "TestMode",
    parameters: Sequence[Parameter] | None = None,
    tools: Iterable[Tool] | None = None,
    history: list[BaseMessage] | None = None,
) -> tuple[AgentDescription, AgentState]:
    desc = AgentDescription(
        name=name,
        model=model,
        parameters=list(parameters) if parameters is not None else [],
        tools=list(tools) if tools is not None else [],
    )
    state = AgentState(history=list(history) if history is not None else [])
    return desc, state


def make_test_context(
    *,
    name: str = "TestAgent",
    model: str = "TestMode",
    parameters: Sequence[Parameter] | None = None,
    tools: Iterable[Tool] | None = None,
    history: list[BaseMessage] | None = None,
) -> AgentContext:
    desc, state = make_test_agent(
        name=name,
        model=model,
        parameters=parameters,
        tools=tools,
        history=history,
    )
    return AgentContext(desc=desc, state=state)


async def execute_tool_calls_via_messages(
    actor: ToolCallActor,
    *,
    message: AssistantMessage,
) -> HandleToolCallsResponse:
    request_id = uuid4().hex
    future: asyncio.Future[HandleToolCallsResponse] = asyncio.get_running_loop().create_future()
    if actor._actor_directory is None:  # pyright: ignore[reportPrivateUsage]
        raise RuntimeError("ToolCallActor test helper requires actor_directory.")
    reply_uri = f"actor://test/reply/tool-calls/{request_id}"

    @dataclass(slots=True)
    class _ReplyActor:
        async def send_message(self, response: HandleToolCallsResponse) -> None:
            if response.request_id != request_id:
                future.set_exception(RuntimeError(f"Mismatched tool response id: {response.request_id}"))
                return
            future.set_result(response)

    actor._actor_directory.register(uri=reply_uri, actor=_ReplyActor())  # pyright: ignore[reportPrivateUsage]
    try:
        await actor.send_message(
            HandleToolCallsRequest(
                request_id=request_id,
                message=message,
                reply_to_uri=reply_uri,
            )
        )
        return await future
    finally:
        actor._actor_directory.unregister(uri=reply_uri)  # pyright: ignore[reportPrivateUsage]


def append_tool_call_results_to_history(
    *,
    history: list[BaseMessage],
    execution_results: list[ToolCallExecutionResult],
    context_name: str,
    progress_callbacks: ProgressCallbacks,
    handle_tool_result: Callable[[ToolResult], str] | None = None,
) -> None:
    for item in execution_results:
        if handle_tool_result is not None:
            result_summary = handle_tool_result(item.result)
        elif isinstance(item.result, TextResult):
            result_summary = item.result.content
        else:
            result_summary = f"Tool produced result of type {type(item.result).__name__}"
        append_tool_message(
            history,
            callbacks=progress_callbacks,
            context_name=context_name,
            message=ToolMessage(
                tool_call_id=item.tool_call_id,
                name=item.name,
                content=result_summary,
            ),
            arguments=item.arguments,
        )


@asynccontextmanager
async def agent_actor_scope(
    *,
    context_name: str = "test",
    completer: Completer | None = None,
    progress_callbacks: ProgressCallbacks | None = None,
) -> AsyncIterator[AgentActor]:
    actor_directory = ActorDirectory()
    agent_uri = f"actor://{context_name}/agent"
    llm_uri = f"actor://{context_name}/llm"

    llm_actor = LLMActor(
        completer=completer or FakeCompleter([]),
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        context_name=context_name,
        actor_directory=actor_directory,
    )
    actor = AgentActor(
        context_name=context_name,
        actor_directory=actor_directory,
        self_uri=agent_uri,
        llm_actor_uri=llm_uri,
    )
    actor_directory.register(uri=agent_uri, actor=actor)
    actor_directory.register(uri=llm_uri, actor=llm_actor)
    llm_actor.start()
    actor.start()
    try:
        yield actor
    finally:
        await actor.stop()
        await llm_actor.stop()
        actor_directory.unregister(uri=agent_uri)
        actor_directory.unregister(uri=llm_uri)


@asynccontextmanager
async def tool_call_actor_scope(
    *,
    tools: Sequence[Tool],
    ui: UI,
    context_name: str = "test",
    progress_callbacks: ProgressCallbacks | None = None,
    tool_callbacks: ToolCallbacks | None = None,
) -> AsyncIterator[ToolCallActor]:
    actor_directory = ActorDirectory()
    capabilities, capability_actors = register_tool_capabilities(
        actor_directory=actor_directory,
        tools=tuple(tools),
        context_name=context_name,
        uri_prefix=f"actor://{context_name}/tool-capability",
    )
    actor = ToolCallActor(
        default_tool_capabilities=capabilities,
        ui=ui,
        context_name=context_name,
        actor_directory=actor_directory,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        tool_callbacks=tool_callbacks or NullToolCallbacks(),
    )
    actor.start()
    try:
        yield actor
    finally:
        await actor.stop()
        for capability_actor in capability_actors:
            await capability_actor.stop()
        for capability in capabilities:
            actor_directory.unregister(uri=capability.uri)


@dataclass(slots=True)
class ActorBundle:
    agent_actor: AgentActor
    chat_actor: ChatActor
    llm_actor: LLMActor
    tool_call_actor: ToolCallActor
    user_actor: UI
    actor_directory: ActorDirectory
    tool_capabilities: tuple[ToolCapability, ...]
    tool_capability_actors: tuple[ToolCapabilityActor, ...]
    agent_actor_uri: str
    chat_actor_uri: str
    llm_actor_uri: str
    tool_call_actor_uri: str
    user_actor_uri: str


async def run_chat_via_messages(
    actors: ActorBundle,
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks,
    completer: Any,
    context_name: str,
) -> None:
    actors.llm_actor.configure_runtime(completer=completer, progress_callbacks=callbacks)
    actors.chat_actor.set_progress_callbacks(callbacks)
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())
    request_id = uuid4().hex
    reply_uri = f"actor://test/reply/chat/{request_id}"
    future = register_run_reply_waiter(actors.actor_directory, request_id=request_id, reply_uri=reply_uri)
    runtime_capabilities, runtime_actors = register_tool_capabilities(
        actor_directory=actors.actor_directory,
        tools=tuple(tools_with_meta),
        context_name=context_name,
        uri_prefix=f"actor://{context_name}/runtime-chat-capability",
    )
    try:
        await actors.actor_directory.send_message(
            uri=actors.chat_actor_uri,
            message=RunChatRequest(
                request_id=request_id,
                history=history,
                model=model,
                tool_capabilities=runtime_capabilities,
                instructions=instructions,
                context_name=context_name,
                user_actor_uri=actors.user_actor_uri,
                tool_call_actor_uri=actors.tool_call_actor_uri,
                reply_to_uri=reply_uri,
            ),
        )
        await future
    finally:
        for capability_actor in runtime_actors:
            await capability_actor.stop()
        for capability in runtime_capabilities:
            actors.actor_directory.unregister(uri=capability.uri)
        actors.actor_directory.unregister(uri=reply_uri)


async def run_agent_via_messages(
    actors: ActorBundle,
    *,
    ctx: AgentContext,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks,
    completer: Completer,
    compact_conversation_at_tokens: int,
) -> None:
    actors.llm_actor.configure_runtime(completer=completer, progress_callbacks=progress_callbacks)
    actors.agent_actor.set_progress_callbacks(progress_callbacks)
    request_id = uuid4().hex
    reply_uri = f"actor://test/reply/agent/{request_id}"
    future = register_run_reply_waiter(actors.actor_directory, request_id=request_id, reply_uri=reply_uri)
    runtime_capabilities, runtime_actors = register_tool_capabilities(
        actor_directory=actors.actor_directory,
        tools=tuple(tools),
        context_name=ctx.desc.name,
        uri_prefix=f"actor://{ctx.desc.name}/runtime-agent-capability",
    )
    try:
        await actors.actor_directory.send_message(
            uri=actors.agent_actor_uri,
            message=RunAgentRequest(
                request_id=request_id,
                ctx=ctx,
                tool_capabilities=runtime_capabilities,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
                tool_call_actor_uri=actors.tool_call_actor_uri,
                reply_to_uri=reply_uri,
            ),
        )
        await future
    finally:
        for capability_actor in runtime_actors:
            await capability_actor.stop()
        for capability in runtime_capabilities:
            actors.actor_directory.unregister(uri=capability.uri)
        actors.actor_directory.unregister(uri=reply_uri)


@asynccontextmanager
async def system_actor_scope_for_tests(
    *,
    tools: Sequence[Tool],
    ui: UI,
    context_name: str = "test",
    progress_callbacks: ProgressCallbacks | None = None,
    tool_callbacks: ToolCallbacks | None = None,
) -> AsyncIterator[ActorBundle]:
    actor_directory = ActorDirectory()
    agent_actor_uri = f"actor://{context_name}/agent"
    chat_actor_uri = f"actor://{context_name}/chat"
    llm_actor_uri = f"actor://{context_name}/llm"
    tool_call_actor_uri = f"actor://{context_name}/tool-call"
    user_actor_uri = f"actor://{context_name}/user"

    owns_user_actor = not isinstance(ui, ActorUI)
    user_actor = (
        ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=context_name, actor_directory=actor_directory)
    )
    registered_capabilities, registered_capability_actors = register_tool_capabilities(
        actor_directory=actor_directory,
        tools=tuple(tools),
        context_name=context_name,
        uri_prefix=f"actor://{context_name}/tool-capability",
    )
    tool_call_actor = ToolCallActor(
        default_tool_capabilities=registered_capabilities,
        ui=ui,
        context_name=context_name,
        actor_directory=actor_directory,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        tool_callbacks=tool_callbacks or NullToolCallbacks(),
    )
    llm_actor = LLMActor(
        completer=FakeCompleter([]),  # overwritten via configure_runtime in run helpers
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        context_name=context_name,
        actor_directory=actor_directory,
    )
    chat_actor = ChatActor(
        context_name=context_name,
        actor_directory=actor_directory,
        self_uri=chat_actor_uri,
        llm_actor_uri=llm_actor_uri,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
    )
    agent_actor = AgentActor(
        context_name=context_name,
        actor_directory=actor_directory,
        self_uri=agent_actor_uri,
        llm_actor_uri=llm_actor_uri,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
    )
    actor_directory.register(uri=agent_actor_uri, actor=agent_actor)
    actor_directory.register(uri=chat_actor_uri, actor=chat_actor)
    actor_directory.register(uri=llm_actor_uri, actor=llm_actor)
    actor_directory.register(uri=tool_call_actor_uri, actor=tool_call_actor)
    actor_directory.register(uri=user_actor_uri, actor=user_actor)

    if owns_user_actor and isinstance(user_actor, ActorUI):
        user_actor.start()
    llm_actor.start()
    tool_call_actor.start()
    agent_actor.start()
    chat_actor.start()
    try:
        yield ActorBundle(
            agent_actor=agent_actor,
            chat_actor=chat_actor,
            llm_actor=llm_actor,
            tool_call_actor=tool_call_actor,
            user_actor=user_actor,
            actor_directory=actor_directory,
            tool_capabilities=registered_capabilities,
            tool_capability_actors=registered_capability_actors,
            agent_actor_uri=agent_actor_uri,
            chat_actor_uri=chat_actor_uri,
            llm_actor_uri=llm_actor_uri,
            tool_call_actor_uri=tool_call_actor_uri,
            user_actor_uri=user_actor_uri,
        )
    finally:
        await chat_actor.stop()
        await tool_call_actor.stop()
        await agent_actor.stop()
        await llm_actor.stop()
        for capability_actor in registered_capability_actors:
            await capability_actor.stop()
        for capability in registered_capabilities:
            actor_directory.unregister(uri=capability.uri)
        if owns_user_actor and isinstance(user_actor, ActorUI):
            await user_actor.stop()
        actor_directory.unregister(uri=agent_actor_uri)
        actor_directory.unregister(uri=chat_actor_uri)
        actor_directory.unregister(uri=llm_actor_uri)
        actor_directory.unregister(uri=tool_call_actor_uri)
        actor_directory.unregister(uri=user_actor_uri)
