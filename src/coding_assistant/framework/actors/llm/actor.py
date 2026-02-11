from __future__ import annotations

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
)


class LLMActor:
    def __init__(self, *, context_name: str = "llm") -> None:
        self._actor: Actor[LLMCompleteStepRequest] = Actor(name=f"{context_name}.llm", handler=self._handle_message)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._actor.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._actor.stop()
        self._started = False

    async def send_message(self, message: LLMCompleteStepRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: LLMCompleteStepRequest) -> None:
        try:
            if not message.history:
                raise RuntimeError("History is required in order to run a step.")
            completion = await message.completer(
                list(message.history),
                model=message.model,
                tools=message.tools,
                callbacks=message.progress_callbacks,
            )
            await message.reply_to.send_message(
                LLMCompleteStepResponse(
                    request_id=message.request_id,
                    message=completion.message,
                    usage=completion.usage,
                )
            )
        except BaseException as exc:
            await message.reply_to.send_message(
                LLMCompleteStepResponse(
                    request_id=message.request_id,
                    message=None,
                    usage=None,
                    error=exc,
                )
            )
        return None
