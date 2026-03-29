from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    Tool,
    ToolCall,
    Usage,
    UserMessage,
)


@dataclass
class StreamStep:
    message: AssistantMessage
    started_event: asyncio.Event | None = None
    release_event: asyncio.Event | None = None


class ControlledStreamer:
    def __init__(self, steps: list[StreamStep]) -> None:
        self.steps = list(steps)
        self.prompts: list[str | list[dict[str, Any]]] = []

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del tools, model
        if not self.steps:
            raise AssertionError("Streamer script exhausted.")

        self.prompts.append(_get_latest_user_content(messages))
        step = self.steps.pop(0)
        if step.started_event is not None:
            step.started_event.set()
        if isinstance(step.message.content, str) and step.message.content:
            yield ContentDeltaEvent(content=step.message.content)
        if step.release_event is not None:
            await step.release_event.wait()
        yield CompletionEvent(completion=Completion(message=step.message, usage=Usage(tokens=10, cost=0.0)))


class FailingStreamer:
    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        raise RuntimeError("boom")
        yield


class EchoTool(Tool):
    def name(self) -> str:
        return "echo_tool"

    def description(self) -> str:
        return "Echo the provided text."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return f"echo:{parameters['text']}"


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_session(*, completion_streamer: Any, tools: list[Tool] | None = None) -> AgentSession:
    return AgentSession(
        history=make_system_history(),
        model="test-model",
        tools=tools or [],
        completion_streamer=completion_streamer,
    )


def _get_latest_user_content(messages: list[BaseMessage]) -> str | list[dict[str, Any]]:
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            return message.content
    raise AssertionError("No user message present in streamer call.")


async def wait_for_event(queue: asyncio.Queue[AgentSessionEvent], event_type: type[object]) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if isinstance(event, event_type):
            return event


async def wait_for_matching_event(
    queue: asyncio.Queue[AgentSessionEvent],
    predicate: Callable[[object], bool],
) -> object:
    while True:
        event = await asyncio.wait_for(queue.get(), timeout=1)
        if predicate(event):
            return event


@pytest.mark.asyncio
async def test_agent_session_runs_prompt_and_updates_history() -> None:
    session = make_session(
        completion_streamer=ControlledStreamer([StreamStep(message=AssistantMessage(content="Hello from the worker"))]),
    )

    async with session.subscribe() as queue:
        initial_state = await wait_for_event(queue, StateChangedEvent)
        assert isinstance(initial_state, StateChangedEvent)

        assert await session.enqueue_prompt("Hi") is True

        finished_event = await wait_for_event(queue, RunFinishedEvent)

    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Hello from the worker"
    assert session.history[-1] == AssistantMessage(content="Hello from the worker")
    assert session.state.promptable is True
    assert session.state.running is False
    await session.close()


@pytest.mark.asyncio
async def test_agent_session_queues_prompts_fifo_while_run_is_in_flight() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(message=AssistantMessage(content="Second result")),
        ]
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await session.enqueue_prompt("second") is True

        first_release.set()

        first_finished = await wait_for_event(queue, RunFinishedEvent)
        second_finished = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(first_finished, RunFinishedEvent)
    assert isinstance(second_finished, RunFinishedEvent)
    assert [first_finished.summary, second_finished.summary] == ["First result", "Second result"]
    assert streamer.prompts == ["first", "second"]
    assert session.history == [
        *make_system_history(),
        UserMessage(content="first"),
        AssistantMessage(content="First result"),
        UserMessage(content="second"),
        AssistantMessage(content="Second result"),
    ]


@pytest.mark.asyncio
async def test_agent_session_priority_prompt_runs_before_existing_queued_prompts() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(message=AssistantMessage(content="Priority result")),
            StreamStep(message=AssistantMessage(content="Second result")),
        ]
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await session.enqueue_prompt("second") is True
        assert await session.enqueue_prompt("priority", priority=True) is True

        first_release.set()

        finished_events = [await wait_for_event(queue, RunFinishedEvent) for _ in range(3)]

    await session.close()
    assert [event.summary for event in finished_events if isinstance(event, RunFinishedEvent)] == [
        "First result",
        "Priority result",
        "Second result",
    ]
    assert streamer.prompts == ["first", "priority", "second"]


@pytest.mark.asyncio
async def test_agent_session_interrupt_cancels_current_run_and_drops_partial_output_from_history() -> None:
    first_started = asyncio.Event()
    never_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="Working..."),
                started_event=first_started,
                release_event=never_release,
            ),
            StreamStep(message=AssistantMessage(content="Done second")),
        ]
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await session.interrupt_and_enqueue("second") is True

        cancelled_event = await wait_for_event(queue, RunCancelledEvent)
        finished_event = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(cancelled_event, RunCancelledEvent)
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Done second"
    assert streamer.prompts == ["first", "second"]
    assert session.history == [
        *make_system_history(),
        UserMessage(content="first"),
        UserMessage(content="second"),
        AssistantMessage(content="Done second"),
    ]


@pytest.mark.asyncio
async def test_agent_session_cancel_current_run_publishes_cancellation_and_restores_state() -> None:
    started = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="Working..."),
                started_event=started,
                release_event=asyncio.Event(),
            )
        ]
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Do the task") is True
        await asyncio.wait_for(started.wait(), timeout=1)

        await session.cancel_current_run()

        cancelled_event = await wait_for_event(queue, RunCancelledEvent)
        state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and not event.state.running,
        )

    await session.close()
    assert isinstance(cancelled_event, RunCancelledEvent)
    assert isinstance(state_event, StateChangedEvent)
    assert state_event.state.promptable is True
    assert state_event.state.running is False
    assert state_event.state.queued_prompt_count == 0


@pytest.mark.asyncio
async def test_agent_session_emits_tool_call_and_finish_events() -> None:
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="echo_tool",
                                arguments='{"text": "hello"}',
                            ),
                        )
                    ]
                )
            ),
            StreamStep(message=AssistantMessage(content="Done")),
        ]
    )
    session = make_session(completion_streamer=streamer, tools=[EchoTool()])

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Use the tool") is True

        tool_event = await wait_for_event(queue, ToolCallsEvent)
        finished_event = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(tool_event, ToolCallsEvent)
    assert tool_event.message.tool_calls[0].function.name == "echo_tool"
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Done"


@pytest.mark.asyncio
async def test_agent_session_publishes_run_failed_event() -> None:
    session = make_session(completion_streamer=FailingStreamer())

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Hi") is True
        failed_event = await wait_for_event(queue, RunFailedEvent)

    await session.close()
    assert isinstance(failed_event, RunFailedEvent)
    assert failed_event.error == "boom"
