from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from coding_assistant.core.agent_session import (
    AgentSession,
    ScheduledRun,
    SessionState,
)
from coding_assistant.core.compacting_session import COMPACTION_PROMPT, AutoCompactingSession
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    FunctionCall,
    SystemMessage,
    ToolCall,
    Usage,
    UserMessage,
)


class ScriptedStreamer:
    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        if not self._completions:
            raise RuntimeError("No scripted completion available.")
        completion = self._completions.pop(0)
        yield CompletionEvent(completion=completion)


@pytest.mark.asyncio
async def test_compacting_session_properties_delegation() -> None:
    inner = Mock(spec=AgentSession)
    inner.model = "test-model"
    inner.history = [SystemMessage(content="instructions")]
    inner.state = SessionState(running=False, usage=Usage(tokens=1000, cost=0.01))

    session = AutoCompactingSession(inner, token_budget=50_000)

    assert session.model == "test-model"
    assert session.token_budget == 50_000
    assert session.history == [SystemMessage(content="instructions")]
    assert session.state == SessionState(running=False, usage=Usage(tokens=1000, cost=0.01))


@pytest.mark.asyncio
async def test_enqueue_prompt_under_budget_does_not_compact() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=5_000, cost=0.05))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt("Hello")

    assert result is not None
    assert inner.enqueue_prompt.call_count == 1
    inner.enqueue_prompt.assert_awaited_once_with("Hello")


@pytest.mark.asyncio
async def test_enqueue_prompt_over_budget_triggers_compaction_first() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt("Continue the task")

    assert result is not None
    assert inner.enqueue_prompt.call_count == 2
    assert inner.enqueue_prompt.await_args_list[0].args == (COMPACTION_PROMPT,)
    assert inner.enqueue_prompt.await_args_list[1].args == ("Continue the task",)


@pytest.mark.asyncio
async def test_enqueue_prompt_over_budget_with_compaction_prompt_does_not_duplicate() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt(COMPACTION_PROMPT)

    assert result is not None
    assert inner.enqueue_prompt.call_count == 1
    inner.enqueue_prompt.assert_awaited_once_with(COMPACTION_PROMPT)


@pytest.mark.asyncio
async def test_enqueue_prompt_with_none_budget_never_compacts() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=1_000_000, cost=10.0))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=None)
    result = await session.enqueue_prompt("Big context")

    assert result is not None
    assert inner.enqueue_prompt.call_count == 1
    inner.enqueue_prompt.assert_awaited_once_with("Big context")


@pytest.mark.asyncio
async def test_enqueue_prompt_with_missing_token_count_never_compacts() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=None, cost=None))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt("No tokens tracked")

    assert result is not None
    assert inner.enqueue_prompt.call_count == 1
    inner.enqueue_prompt.assert_awaited_once_with("No tokens tracked")


@pytest.mark.asyncio
async def test_enqueue_prompt_if_idle_under_budget() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=5_000, cost=0.05))
    inner.enqueue_prompt_if_idle = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt_if_idle("Idle prompt")

    assert result is not None
    inner.enqueue_prompt_if_idle.assert_awaited_once_with("Idle prompt")


@pytest.mark.asyncio
async def test_enqueue_prompt_if_idle_over_budget_when_idle() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))
    inner.enqueue_prompt = AsyncMock(return_value=Mock(spec=ScheduledRun))

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt_if_idle("Idle prompt")

    assert result is not None
    assert inner.enqueue_prompt.call_count == 2
    assert inner.enqueue_prompt.await_args_list[0].args == (COMPACTION_PROMPT,)
    assert inner.enqueue_prompt.await_args_list[1].args == ("Idle prompt",)


@pytest.mark.asyncio
async def test_enqueue_prompt_if_idle_over_budget_when_busy() -> None:
    inner = Mock(spec=AgentSession)
    inner.state = SessionState(running=True, usage=Usage(tokens=15_000, cost=0.15))
    inner.enqueue_prompt = AsyncMock()

    session = AutoCompactingSession(inner, token_budget=10_000)
    result = await session.enqueue_prompt_if_idle("Busy prompt")

    assert result is None
    inner.enqueue_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_control_delegation_methods() -> None:
    inner = Mock(spec=AgentSession)
    inner.enqueue_steering_prompt = AsyncMock(return_value=True)
    inner.pop_last_queued_prompt = AsyncMock(return_value="popped")
    inner.cancel_current_run = AsyncMock(return_value=True)
    inner.resume = AsyncMock(return_value=True)
    inner.close = AsyncMock()
    inner.subscribe = Mock()

    session = AutoCompactingSession(inner, token_budget=10_000)

    assert await session.enqueue_steering_prompt("steer") is True
    inner.enqueue_steering_prompt.assert_awaited_once_with("steer")

    assert await session.pop_last_queued_prompt() == "popped"
    inner.pop_last_queued_prompt.assert_awaited_once()

    assert await session.cancel_current_run(pause_queue=True) is True
    inner.cancel_current_run.assert_awaited_once_with(pause_queue=True)

    assert await session.resume() is True
    inner.resume.assert_awaited_once()

    await session.close()
    inner.close.assert_awaited_once()

    session.subscribe()
    inner.subscribe.assert_called_once()


@pytest.mark.asyncio
async def test_auto_compaction_integration_with_real_session() -> None:
    compact_call = ToolCall(
        id="call-compact",
        function=FunctionCall(
            name="compact_conversation",
            arguments='{"summary": "Compacted summary of step 1"}',
        ),
    )
    streamer = ScriptedStreamer(
        [
            # First turn: returns a response with high usage (over budget)
            Completion(
                message=AssistantMessage(content="Done with step 1"),
                usage=Usage(tokens=1000, cost=0.01),
            ),
            # Auto-compaction turn: model calls compact_conversation
            Completion(
                message=AssistantMessage(tool_calls=[compact_call]),
                usage=Usage(tokens=200, cost=0.002),
            ),
            # Continuation after compaction tool
            Completion(
                message=AssistantMessage(content="Compacted. Ready for next prompt."),
                usage=Usage(tokens=200, cost=0.002),
            ),
            # Second user prompt execution
            Completion(
                message=AssistantMessage(content="Done with step 2"),
                usage=Usage(tokens=300, cost=0.003),
            ),
        ],
    )
    raw_session = AgentSession(
        history=[SystemMessage(content="System instructions")],
        model="test-model",
        tools=[],
        completion_streamer=streamer,
    )
    session = AutoCompactingSession(raw_session, token_budget=800)

    async with session.subscribe():
        # Step 1: initial prompt (under budget initially)
        scheduled1 = await session.enqueue_prompt("Run step 1")
        assert scheduled1 is not None
        outcome1 = await scheduled1.completion
        assert outcome1.stop_reason == "end_turn"
        assert session.state.usage is not None
        assert session.state.usage.tokens == 1000  # now over budget (800)

        # Step 2: next prompt enqueued when over budget -> triggers auto-compaction before prompt
        scheduled2 = await session.enqueue_prompt("Run step 2")
        assert scheduled2 is not None
        outcome2 = await scheduled2.completion
        assert outcome2.stop_reason == "end_turn"

    # Verify history after compaction contains compacted summary
    history = session.history
    assert len(history) == 5
    assert history[0] == SystemMessage(content="System instructions")
    assert isinstance(history[1], UserMessage)
    assert "Compacted summary of step 1" in str(history[1].content)
    assert history[2] == AssistantMessage(content="Compacted. Ready for next prompt.")
    assert history[3] == UserMessage(content="Run step 2")
    assert history[4] == AssistantMessage(content="Done with step 2")

    await session.close()
