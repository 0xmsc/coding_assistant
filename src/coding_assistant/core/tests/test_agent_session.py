from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
    ToolCallUpdateEvent,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)


@dataclass
class StreamStep:
    message: AssistantMessage
    started_event: asyncio.Event | None = None
    release_event: asyncio.Event | None = None
    usage: Usage | None = Usage(tokens=10, cost=0.0)


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
        yield CompletionEvent(
            completion=Completion(
                message=step.message,
                usage=step.usage,
            ),
        )


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

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        return TextToolResult(content=f"echo:{parameters['text']}")


class BlockingEchoTool(EchoTool):
    def __init__(self, *, started_event: asyncio.Event, release_event: asyncio.Event) -> None:
        self._started_event = started_event
        self._release_event = release_event

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        self._started_event.set()
        await self._release_event.wait()
        return await super().execute(parameters)


class SlowTool(Tool):
    def __init__(self, *, started_event: asyncio.Event, release_event: asyncio.Event) -> None:
        self._started_event = started_event
        self._release_event = release_event

    def name(self) -> str:
        return "slow_tool"

    def description(self) -> str:
        return "A tool that blocks until released."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        self._started_event.set()
        await self._release_event.wait()
        return TextToolResult(content=f"slow:{parameters['text']}")


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
    assert session.state.running is False
    await session.close()


@pytest.mark.asyncio
async def test_agent_session_enqueue_prompt_if_idle_rejects_busy_session() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    session = make_session(
        completion_streamer=ControlledStreamer(
            [
                StreamStep(
                    message=AssistantMessage(content="First result"),
                    started_event=first_started,
                    release_event=first_release,
                ),
            ],
        ),
    )

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert await session.enqueue_prompt_if_idle("second") is False
        assert session.state.pending_prompts == ()

        first_release.set()
        await wait_for_event(queue, RunFinishedEvent)

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
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await asyncio.wait_for(first_started.wait(), timeout=1)
        assert await session.enqueue_prompt("second") is True
        assert session.state.pending_prompts == ("second",)

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
async def test_agent_session_starts_queued_prompt_only_after_current_run_finishes() -> None:
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First result"),
                started_event=first_started,
                release_event=first_release,
            ),
            StreamStep(
                message=AssistantMessage(content="Second result"),
                started_event=second_started,
            ),
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, PromptStartedEvent) and event.content == "first",
        )
        await asyncio.wait_for(first_started.wait(), timeout=1)

        assert await session.enqueue_prompt("second") is True

        await asyncio.sleep(0)
        pending_events: list[AgentSessionEvent] = []
        while not queue.empty():
            pending_events.append(queue.get_nowait())
        assert not any(isinstance(event, PromptStartedEvent) and event.content == "second" for event in pending_events)
        assert second_started.is_set() is False

        first_release.set()

        await asyncio.wait_for(second_started.wait(), timeout=1)
        finished_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, RunFinishedEvent) and event.summary == "Second result",
        )

    await session.close()
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Second result"


@pytest.mark.asyncio
async def test_agent_session_inserts_steering_prompt_into_active_run_after_tool_boundary() -> None:
    tool_started = asyncio.Event()
    tool_release = asyncio.Event()
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
                        ),
                    ],
                ),
            ),
            StreamStep(message=AssistantMessage(content="Steered result")),
        ],
    )
    session = make_session(
        completion_streamer=streamer,
        tools=[BlockingEchoTool(started_event=tool_started, release_event=tool_release)],
    )

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await wait_for_event(queue, ToolCallsEvent)
        await asyncio.wait_for(tool_started.wait(), timeout=1)

        assert await session.enqueue_steering_prompt("steer now") is True
        assert session.state.pending_prompts == ("steer now",)

        tool_release.set()

        steering_started_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, PromptStartedEvent) and event.content == "steer now",
        )
        finished_event = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(steering_started_event, PromptStartedEvent)
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Steered result"
    assert streamer.prompts == ["first", "steer now"]
    assert session.history == [
        *make_system_history(),
        UserMessage(content="first"),
        AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    function=FunctionCall(
                        name="echo_tool",
                        arguments='{"text": "hello"}',
                    ),
                ),
            ],
        ),
        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hello"),
        UserMessage(content="steer now"),
        AssistantMessage(content="Steered result"),
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
            ),
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Do the task") is True
        await asyncio.wait_for(started.wait(), timeout=1)

        await session.cancel_current_run()

        cancelled_event = await wait_for_event(queue, RunCancelledEvent)
        assert session.state.running is False
        assert session.state.paused is False
        assert len(session.state.pending_prompts) == 0
        state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and not event.state.running,
        )

    await session.close()
    assert isinstance(cancelled_event, RunCancelledEvent)
    assert isinstance(state_event, StateChangedEvent)
    assert state_event.state.running is False
    assert state_event.state.paused is False
    assert len(state_event.state.pending_prompts) == 0
    assert session.history == [
        *make_system_history(),
        UserMessage(content="Do the task"),
    ]


@pytest.mark.asyncio
async def test_agent_session_cancel_with_pause_queue_keeps_pending_prompt_stopped_until_resume() -> None:
    started = asyncio.Event()
    second_started = asyncio.Event()
    never_release = asyncio.Event()
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="Working..."),
                started_event=started,
                release_event=never_release,
            ),
            StreamStep(
                message=AssistantMessage(content="Resumed result"),
                started_event=second_started,
            ),
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("abandoned prompt") is True
        await asyncio.wait_for(started.wait(), timeout=1)
        assert await session.enqueue_prompt("queued prompt") is True

        await session.cancel_current_run(pause_queue=True)
        await wait_for_event(queue, RunCancelledEvent)
        paused_state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and event.state.paused,
        )
        assert session.state.running is False
        assert session.state.paused is True
        assert session.state.pending_prompts == ("queued prompt",)

        await asyncio.sleep(0)
        assert second_started.is_set() is False

        assert await session.resume() is True
        resumed_state_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, StateChangedEvent) and not event.state.paused and event.state.running,
        )
        await asyncio.wait_for(second_started.wait(), timeout=1)
        finished_event = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(paused_state_event, StateChangedEvent)
    assert paused_state_event.state.paused is True
    assert isinstance(resumed_state_event, StateChangedEvent)
    assert resumed_state_event.state.paused is False
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Resumed result"
    assert streamer.prompts == ["abandoned prompt", "queued prompt"]
    assert session.history == [
        *make_system_history(),
        UserMessage(content="abandoned prompt"),
        UserMessage(content="queued prompt"),
        AssistantMessage(content="Resumed result"),
    ]


@pytest.mark.asyncio
async def test_agent_session_cancel_during_tool_execution_preserves_completed_and_cancelled_tool_messages() -> None:
    slow_started = asyncio.Event()
    never_release = asyncio.Event()
    tool_calls = [
        ToolCall(
            id="call-1",
            function=FunctionCall(
                name="echo_tool",
                arguments='{"text": "done"}',
            ),
        ),
        ToolCall(
            id="call-2",
            function=FunctionCall(
                name="slow_tool",
                arguments='{"text": "later"}',
            ),
        ),
    ]
    streamer = ControlledStreamer([StreamStep(message=AssistantMessage(tool_calls=tool_calls))])
    session = make_session(
        completion_streamer=streamer,
        tools=[EchoTool(), SlowTool(started_event=slow_started, release_event=never_release)],
    )

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Use tools") is True
        await wait_for_event(queue, ToolCallsEvent)
        await asyncio.wait_for(slow_started.wait(), timeout=1)

        await session.cancel_current_run()
        cancelled_event = await wait_for_event(queue, RunCancelledEvent)

    await session.close()
    assert isinstance(cancelled_event, RunCancelledEvent)
    assert session.history == [
        *make_system_history(),
        UserMessage(content="Use tools"),
        AssistantMessage(tool_calls=tool_calls),
        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:done"),
        ToolMessage(
            tool_call_id="call-2",
            name="slow_tool",
            content="Tool execution cancelled by user before completion.",
        ),
    ]


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
                        ),
                    ],
                ),
            ),
            StreamStep(message=AssistantMessage(content="Done")),
        ],
    )
    session = make_session(completion_streamer=streamer, tools=[EchoTool()])

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Use the tool") is True

        tool_event = await wait_for_event(queue, ToolCallsEvent)
        tool_started_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, ToolCallUpdateEvent) and event.event.status == "in_progress",
        )
        tool_completed_event = await wait_for_matching_event(
            queue,
            lambda event: isinstance(event, ToolCallUpdateEvent) and event.event.status == "completed",
        )
        finished_event = await wait_for_event(queue, RunFinishedEvent)

    await session.close()
    assert isinstance(tool_event, ToolCallsEvent)
    assert tool_event.message.tool_calls[0].function.name == "echo_tool"
    assert isinstance(tool_started_event, ToolCallUpdateEvent)
    assert tool_started_event.event.tool_call_id == "call-1"
    assert tool_started_event.event.raw_input == {"text": "hello"}
    assert isinstance(tool_completed_event, ToolCallUpdateEvent)
    assert tool_completed_event.event.raw_output == "echo:hello"
    assert isinstance(finished_event, RunFinishedEvent)
    assert finished_event.summary == "Done"


@pytest.mark.asyncio
async def test_agent_session_publishes_run_failed_event() -> None:
    session = make_session(completion_streamer=FailingStreamer())

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("Hi") is True
        failed_event = await wait_for_event(queue, RunFailedEvent)
        assert session.state.running is False
        assert len(session.state.pending_prompts) == 0

    await session.close()
    assert isinstance(failed_event, RunFailedEvent)
    assert failed_event.error == "boom"


@pytest.mark.asyncio
async def test_agent_session_accumulates_usage_from_completion_event() -> None:
    """State should reflect cumulative tokens and cost after each run."""
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First run"),
                usage=Usage(tokens=1000, cost=0.01),
            ),
            StreamStep(
                message=AssistantMessage(content="Second run"),
                usage=Usage(tokens=2000, cost=0.02),
            ),
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await wait_for_event(queue, RunFinishedEvent)

        assert session.state.usage is not None
        assert session.state.usage.tokens == 1000
        assert session.state.usage.cost == 0.01

        assert await session.enqueue_prompt("second") is True
        await wait_for_event(queue, RunFinishedEvent)

        # tokens is set to latest Usage.tokens; cost accumulates
        assert session.state.usage.tokens == 2000
        assert session.state.usage.cost == 0.03

    await session.close()


@pytest.mark.asyncio
async def test_agent_session_usage_not_accumulated_when_completion_has_no_usage() -> None:
    """A CompletionEvent with no usage should not affect totals."""
    streamer = ControlledStreamer(
        [
            StreamStep(
                message=AssistantMessage(content="First run"),
                usage=Usage(tokens=500, cost=0.005),
            ),
            StreamStep(
                message=AssistantMessage(content="Second run"),
                usage=None,
            ),
        ],
    )
    session = make_session(completion_streamer=streamer)

    async with session.subscribe() as queue:
        await wait_for_event(queue, StateChangedEvent)
        assert await session.enqueue_prompt("first") is True
        await wait_for_event(queue, RunFinishedEvent)
        assert session.state.usage is not None
        assert session.state.usage.tokens == 500
        assert session.state.usage.cost == 0.005

        assert await session.enqueue_prompt("second") is True
        await wait_for_event(queue, RunFinishedEvent)
        # Totals unchanged since usage was None
        assert session.state.usage is not None
        assert session.state.usage.tokens == 500
        assert session.state.usage.cost == 0.005

    await session.close()
