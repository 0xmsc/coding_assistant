from __future__ import annotations

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    ConfigureLLMRuntimeRequest,
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
)
from coding_assistant.framework.types import Completer
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks


class LLMActor:
    def __init__(self, *, context_name: str = "llm") -> None:
        self._actor: Actor[LLMCompleteStepRequest | ConfigureLLMRuntimeRequest] = Actor(
            name=f"{context_name}.llm", handler=self._handle_message
        )
        self._completer: Completer = openai_complete
        self._progress_callbacks: ProgressCallbacks = NullProgressCallbacks()
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

    async def send_message(self, message: LLMCompleteStepRequest | ConfigureLLMRuntimeRequest) -> None:
        self.start()
        await self._actor.send(message)

    async def _handle_message(self, message: LLMCompleteStepRequest | ConfigureLLMRuntimeRequest) -> None:
        if isinstance(message, ConfigureLLMRuntimeRequest):
            self._completer = message.completer
            self._progress_callbacks = message.progress_callbacks
            return None
        try:
            if not message.history:
                raise RuntimeError("History is required in order to run a step.")
            completion = await self._completer(
                list(message.history),
                model=message.model,
                tools=message.tools,
                callbacks=self._progress_callbacks,
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
