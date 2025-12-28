from typing import Iterable, Sequence
from unittest.mock import AsyncMock, Mock

from coding_assistant.framework.parameters import Parameter
from coding_assistant.framework.types import AgentDescription, AgentState, AgentContext, Tool
from coding_assistant.llm.types import (
    AssistantMessage,
    FunctionCall as FunctionCall,
    BaseMessage,
    ToolCall as ToolCall,
    Completion,
    Usage,
)
from coding_assistant.ui import UI


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

    async def __call__(self, messages, *, model, tools, callbacks) -> Completion:
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
