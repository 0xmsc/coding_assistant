import asyncio
from dataclasses import dataclass
from collections.abc import Sequence

import pytest

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.common.messages import LLMCompleteStepRequest, LLMCompleteStepResponse
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    UserMessage,
)


@pytest.mark.asyncio
async def test_llm_actor_uses_configured_runtime_dependencies() -> None:
    seen_callbacks: dict[str, object] = {}

    callback_a = NullProgressCallbacks()
    callback_b = NullProgressCallbacks()

    async def completer_a(
        messages: list[BaseMessage], *, model: str, tools: Sequence[Tool], callbacks: ProgressCallbacks
    ) -> Completion:
        await asyncio.sleep(0.01)
        seen_callbacks["a"] = callbacks
        return Completion(message=AssistantMessage(content="from-a"))

    async def completer_b(
        messages: list[BaseMessage], *, model: str, tools: Sequence[Tool], callbacks: ProgressCallbacks
    ) -> Completion:
        seen_callbacks["b"] = callbacks
        return Completion(message=AssistantMessage(content="from-b"))

    @dataclass(slots=True)
    class ReplyActor:
        futures: dict[str, asyncio.Future[LLMCompleteStepResponse]]

        async def send_message(self, message: LLMCompleteStepResponse) -> None:
            self.futures[message.request_id].set_result(message)

    actor_directory = ActorDirectory()
    llm_actor = LLMActor(
        context_name="test-llm",
        actor_directory=actor_directory,
        completer=completer_a,
        progress_callbacks=callback_a,
    )
    llm_actor.start()
    try:
        loop = asyncio.get_running_loop()
        futures: dict[str, asyncio.Future[LLMCompleteStepResponse]] = {
            "a": loop.create_future(),
            "b": loop.create_future(),
        }
        reply_actor = ReplyActor(futures=futures)
        actor_directory.register(uri="actor://test-llm/reply", actor=reply_actor)

        await llm_actor.send_message(
            LLMCompleteStepRequest(
                request_id="a",
                history=(UserMessage(content="one"),),
                model="test-model",
                tool_capabilities=tuple(),
                reply_to_uri="actor://test-llm/reply",
            )
        )
        response_a = await futures["a"]
        llm_actor.configure_runtime(completer=completer_b, progress_callbacks=callback_b)
        await llm_actor.send_message(
            LLMCompleteStepRequest(
                request_id="b",
                history=(UserMessage(content="two"),),
                model="test-model",
                tool_capabilities=tuple(),
                reply_to_uri="actor://test-llm/reply",
            )
        )
        response_b = await futures["b"]
    finally:
        actor_directory.unregister(uri="actor://test-llm/reply")
        await llm_actor.stop()

    assert response_a.message is not None
    assert response_b.message is not None
    assert response_a.message.content == "from-a"
    assert response_b.message.content == "from-b"
    assert seen_callbacks["a"] is callback_a
    assert seen_callbacks["b"] is callback_b
