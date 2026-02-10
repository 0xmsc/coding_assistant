from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Iterable, Sequence
from unittest.mock import AsyncMock, Mock

from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.execution import AgentActor, LLMActor, ToolCallActor
from coding_assistant.framework.parameters import Parameter
from coding_assistant.framework.types import AgentDescription, AgentState, AgentContext
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    FunctionCall as FunctionCall,
    Tool,
    ToolCall as ToolCall,
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


@asynccontextmanager
async def agent_actor_scope(*, context_name: str = "test") -> AsyncIterator[AgentActor]:
    llm_actor = LLMActor(context_name=context_name)
    actor = AgentActor(context_name=context_name, llm_gateway=llm_actor)
    llm_actor.start()
    actor.start()
    try:
        yield actor
    finally:
        await actor.stop()
        await llm_actor.stop()


@asynccontextmanager
async def tool_call_actor_scope(
    *,
    tools: Sequence[Tool],
    ui: UI,
    context_name: str = "test",
    progress_callbacks: ProgressCallbacks | None = None,
    tool_callbacks: ToolCallbacks | None = None,
) -> AsyncIterator[ToolCallActor]:
    actor = ToolCallActor(
        tools=tools,
        ui=ui,
        context_name=context_name,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        tool_callbacks=tool_callbacks or NullToolCallbacks(),
    )
    actor.start()
    try:
        yield actor
    finally:
        await actor.stop()


@dataclass(slots=True)
class ActorBundle:
    agent_actor: AgentActor
    tool_call_actor: ToolCallActor
    user_actor: UI


@asynccontextmanager
async def system_actor_scope_for_tests(
    *,
    tools: Sequence[Tool],
    ui: UI,
    context_name: str = "test",
    progress_callbacks: ProgressCallbacks | None = None,
    tool_callbacks: ToolCallbacks | None = None,
) -> AsyncIterator[ActorBundle]:
    owns_user_actor = not isinstance(ui, ActorUI)
    user_actor = ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=context_name)
    tool_call_actor = ToolCallActor(
        tools=tools,
        ui=user_actor,
        context_name=context_name,
        progress_callbacks=progress_callbacks or NullProgressCallbacks(),
        tool_callbacks=tool_callbacks or NullToolCallbacks(),
    )
    llm_actor = LLMActor(context_name=context_name)
    agent_actor = AgentActor(context_name=context_name, llm_gateway=llm_actor)

    if owns_user_actor and isinstance(user_actor, ActorUI):
        user_actor.start()
    llm_actor.start()
    tool_call_actor.start()
    agent_actor.start()
    try:
        yield ActorBundle(
            agent_actor=agent_actor,
            tool_call_actor=tool_call_actor,
            user_actor=user_actor,
        )
    finally:
        await tool_call_actor.stop()
        await agent_actor.stop()
        await llm_actor.stop()
        if owns_user_actor and isinstance(user_actor, ActorUI):
            await user_actor.stop()
