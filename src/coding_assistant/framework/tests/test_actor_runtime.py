import json5
from dataclasses import dataclass
from typing import Any
import pytest

import coding_assistant.trace as trace
from coding_assistant.framework.actor_runtime import Actor, ActorStoppedError, ResponseChannel
from coding_assistant.trace import enable_tracing


@pytest.mark.asyncio
async def test_actor_send_processes_messages() -> None:
    seen: list[str] = []

    async def handler(message: str) -> None:
        seen.append(message)

    actor = Actor[str](name="test", handler=handler)
    actor.start()
    await actor.send("hello")
    await actor.stop()

    assert seen == ["hello"]


@pytest.mark.asyncio
async def test_actor_send_with_response_channel() -> None:
    @dataclass(slots=True)
    class Request:
        message: str
        response_channel: ResponseChannel[str]

    async def handler(message: Request) -> None:
        message.response_channel.send(f"ok:{message.message}")

    actor = Actor[Request](name="test", handler=handler)
    actor.start()
    response_channel: ResponseChannel[str] = ResponseChannel()
    await actor.send(Request(message="ping", response_channel=response_channel))
    result = await response_channel.wait()
    await actor.stop()

    assert result == "ok:ping"


@pytest.mark.asyncio
async def test_actor_send_after_stop_raises() -> None:
    async def handler(message: str) -> None:
        return None

    actor = Actor[str](name="test", handler=handler)
    actor.start()
    await actor.stop()

    with pytest.raises(ActorStoppedError):
        await actor.send("late")


@pytest.mark.asyncio
async def test_actor_traces_lifecycle_and_latency(tmp_path: Any) -> None:
    trace._trace_dir = None
    trace_dir = tmp_path / "traces"
    enable_tracing(trace_dir)

    async def handler(message: str) -> None:
        return None

    actor = Actor[str](name="trace", handler=handler)
    actor.start()
    await actor.send("ping")
    await actor.stop()

    files = list(trace_dir.glob("*_actor_event.json5"))
    assert files, "Expected actor trace events to be written"

    events = [json5.loads(path.read_text()) for path in files]
    event_types = {event["event"] for event in events}
    assert {"actor.start", "actor.stop", "actor.message"} <= event_types

    message_event = next(event for event in events if event["event"] == "actor.message")
    assert message_event["actor"] == "trace"
    assert message_event["message_type"] == "str"
    assert message_event["queue_wait_ms"] >= 0
    assert message_event["handler_ms"] >= 0

    trace._trace_dir = None
