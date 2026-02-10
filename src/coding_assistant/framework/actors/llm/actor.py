from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    ConfigureLLMRuntimeRequest,
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
)
from coding_assistant.framework.types import Completer
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    Usage,
)


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

    async def complete_step(
        self,
        *,
        history: Sequence[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
    ) -> tuple[AssistantMessage, Usage | None]:
        await self.send_message(ConfigureLLMRuntimeRequest(completer=completer, progress_callbacks=progress_callbacks))
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[AssistantMessage, Usage | None]] = loop.create_future()

        @dataclass(slots=True)
        class _ReplySink:
            async def send_message(self, message: LLMCompleteStepResponse) -> None:
                if message.request_id != request_id:
                    future.set_exception(RuntimeError(f"Mismatched LLM response id: {message.request_id}"))
                    return
                if message.error is not None:
                    future.set_exception(message.error)
                    return
                if message.message is None:
                    future.set_exception(RuntimeError("LLM response missing message."))
                    return
                future.set_result((message.message, message.usage))

        await self.send_message(
            LLMCompleteStepRequest(
                request_id=request_id,
                history=tuple(history),
                model=model,
                tools=tools,
                reply_to=_ReplySink(),
            )
        )
        return await future

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
