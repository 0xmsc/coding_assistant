from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    AgentSessionProtocol,
    PromptContent,
    RunFailedEvent,
    RunFinishedEvent,
    ScheduledRun,
    SessionState,
)
from coding_assistant.core.compacting_session import COMPACTION_PROMPT, AutoCompactingSession
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    FunctionCall,
    SystemMessage,
    ToolCall,
    Usage,
    UserMessage,
)


class RecordingSession:
    def __init__(self) -> None:
        self.model = "test-model"
        self.state = SessionState(running=False, usage=Usage(tokens=0, cost=0.0))
        self.history: list[BaseMessage] = []
        self.enqueued: list[PromptContent] = []
        self.enqueued_if_idle: list[PromptContent] = []
        self.steering: list[PromptContent] = []
        self.popped: PromptContent | None = None
        self.cancelled = False
        self.resumed = False
        self.closed = False
        self._subscribers: list[asyncio.Queue[AgentSessionEvent]] = []

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[AgentSessionEvent]]:
        queue: asyncio.Queue[AgentSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            yield queue
        finally:
            self._subscribers.remove(queue)

    async def enqueue_prompt(self, content: PromptContent) -> ScheduledRun:
        self.enqueued.append(content)
        return _make_scheduled_run()

    async def enqueue_prompt_if_idle(self, content: PromptContent) -> ScheduledRun:
        self.enqueued_if_idle.append(content)
        return _make_scheduled_run()

    async def enqueue_steering_prompt(self, content: PromptContent) -> bool:
        self.steering.append(content)
        return True

    async def pop_last_queued_prompt(self) -> PromptContent | None:
        return self.popped

    async def cancel_current_run(self, *, pause_queue: bool = False) -> bool:
        self.cancelled = pause_queue
        return True

    async def resume(self) -> bool:
        self.resumed = True
        return True

    async def close(self) -> None:
        self.closed = True

    def publish(self, event: AgentSessionEvent) -> None:
        for queue in self._subscribers:
            queue.put_nowait(event)


class ScriptedStreamer:
    def __init__(self, completions: list[Completion]) -> None:
        self._completions = list(completions)

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        if not self._completions:
            raise RuntimeError("No scripted completion available.")
        yield CompletionEvent(completion=self._completions.pop(0))


def _make_scheduled_run() -> ScheduledRun:
    return ScheduledRun(completion=asyncio.get_running_loop().create_future())


async def _wait_for_steering(session: RecordingSession, count: int) -> None:
    async with asyncio.timeout(1):
        while len(session.steering) < count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_agent_session_protocol_conformance() -> None:
    inner = RecordingSession()
    session = AutoCompactingSession(inner, token_budget=10_000)
    assert isinstance(session, AgentSessionProtocol)
    await session.close()


@pytest.mark.asyncio
async def test_delegates_session_interface() -> None:
    inner = RecordingSession()
    inner.history = [SystemMessage(content="instructions")]
    inner.state = SessionState(running=False, usage=Usage(tokens=1000, cost=0.01))
    inner.popped = "popped"
    session = AutoCompactingSession(inner, token_budget=50_000)

    assert session.model == "test-model"
    assert session.token_budget == 50_000
    assert session.history == [SystemMessage(content="instructions")]
    assert session.state == inner.state
    assert await session.enqueue_prompt("queued") is not None
    assert await session.enqueue_prompt_if_idle("idle") is not None
    assert await session.enqueue_steering_prompt("steer") is True
    assert await session.pop_last_queued_prompt() == "popped"
    assert await session.cancel_current_run(pause_queue=True) is True
    assert await session.resume() is True

    assert inner.enqueued == ["queued"]
    assert inner.enqueued_if_idle == ["idle"]
    assert inner.steering == ["steer"]
    assert inner.cancelled is True
    assert inner.resumed is True

    await session.close()
    assert inner.closed is True


@pytest.mark.asyncio
async def test_finished_over_budget_run_enqueues_compaction_as_steering() -> None:
    inner = RecordingSession()
    session = AutoCompactingSession(inner, token_budget=10_000)
    await session.enqueue_prompt("start monitor")
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))

    inner.publish(RunFinishedEvent(summary="done"))
    await _wait_for_steering(inner, 1)

    assert inner.steering == [COMPACTION_PROMPT]
    await session.close()


@pytest.mark.asyncio
async def test_each_finished_over_budget_run_enqueues_another_reminder() -> None:
    inner = RecordingSession()
    session = AutoCompactingSession(inner, token_budget=10_000)
    await session.enqueue_prompt("start monitor")
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))

    inner.publish(RunFinishedEvent(summary="first"))
    await _wait_for_steering(inner, 1)
    inner.publish(RunFinishedEvent(summary="second"))
    await _wait_for_steering(inner, 2)

    assert inner.steering == [COMPACTION_PROMPT, COMPACTION_PROMPT]
    await session.close()


@pytest.mark.asyncio
async def test_failed_run_does_not_enqueue_compaction() -> None:
    inner = RecordingSession()
    session = AutoCompactingSession(inner, token_budget=10_000)
    await session.enqueue_prompt("start monitor")
    inner.state = SessionState(running=False, usage=Usage(tokens=15_000, cost=0.15))

    inner.publish(RunFailedEvent(error="failed"))
    await asyncio.sleep(0)

    assert inner.steering == []
    await session.close()


@pytest.mark.asyncio
async def test_pending_compaction_prompt_prevents_duplicate_reminder() -> None:
    inner = RecordingSession()
    session = AutoCompactingSession(inner, token_budget=10_000)
    await session.enqueue_prompt("start monitor")
    inner.state = SessionState(
        running=False,
        pending_prompts=(COMPACTION_PROMPT,),
        usage=Usage(tokens=15_000, cost=0.15),
    )

    inner.publish(RunFinishedEvent(summary="done"))
    await asyncio.sleep(0)

    assert inner.steering == []
    await session.close()


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
            Completion(
                message=AssistantMessage(content="Done with step 1"),
                usage=Usage(tokens=1000, cost=0.01),
            ),
            Completion(
                message=AssistantMessage(tool_calls=[compact_call]),
                usage=Usage(tokens=1100, cost=0.002),
            ),
            Completion(
                message=AssistantMessage(content="Compacted. Ready for next prompt."),
                usage=Usage(tokens=200, cost=0.002),
            ),
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

    try:
        scheduled1 = await session.enqueue_prompt("Run step 1")
        assert scheduled1 is not None
        assert (await scheduled1.completion).stop_reason == "end_turn"

        async with asyncio.timeout(1):
            while len(session.history) < 2 or "Compacted summary of step 1" not in str(session.history[1].content):
                await asyncio.sleep(0)

        scheduled2 = await session.enqueue_prompt("Run step 2")
        assert scheduled2 is not None
        assert (await scheduled2.completion).stop_reason == "end_turn"
        assert session.history[-2:] == [
            UserMessage(content="Run step 2"),
            AssistantMessage(content="Done with step 2"),
        ]
    finally:
        await session.close()
