import pytest

from coding_assistant.framework.actor_runtime import Actor, ActorStoppedError


@pytest.mark.asyncio
async def test_actor_send_processes_messages() -> None:
    seen: list[str] = []

    async def handler(message: str) -> None:
        seen.append(message)

    actor = Actor[str, None](name="test", handler=handler)
    actor.start()
    await actor.send("hello")
    await actor.stop()

    assert seen == ["hello"]


@pytest.mark.asyncio
async def test_actor_ask_returns_response() -> None:
    async def handler(message: str) -> str:
        return f"ok:{message}"

    actor = Actor[str, str](name="test", handler=handler)
    actor.start()
    result = await actor.ask("ping")
    await actor.stop()

    assert result == "ok:ping"


@pytest.mark.asyncio
async def test_actor_send_after_stop_raises() -> None:
    async def handler(message: str) -> None:
        return None

    actor = Actor[str, None](name="test", handler=handler)
    actor.start()
    await actor.stop()

    with pytest.raises(ActorStoppedError):
        await actor.send("late")
