from __future__ import annotations

from dataclasses import dataclass

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    LLMCompleteStepRequest,
    LLMCompleteStepResponse,
    ToolCapability,
)
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, Tool, ToolResult


@dataclass(slots=True)
class _SchemaOnlyTool(Tool):
    _name: str
    _description: str
    _parameters: dict[str, object]

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def parameters(self) -> dict[str, object]:
        return self._parameters

    async def execute(self, parameters: dict[str, object]) -> ToolResult:
        raise RuntimeError("Schema-only tool does not support execution.")


def _capability_to_schema_tool(capability: ToolCapability) -> Tool:
    return _SchemaOnlyTool(
        _name=capability.name,
        _description=capability.description,
        _parameters=capability.parameters,
    )


class LLMActor:
    def __init__(
        self,
        *,
        completer: Completer,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        context_name: str = "llm",
        actor_directory: ActorDirectory | None = None,
    ) -> None:
        self._actor: Actor[LLMCompleteStepRequest] = Actor(name=f"{context_name}.llm", handler=self._handle_message)
        self._actor_directory = actor_directory
        self._completer = completer
        self._progress_callbacks = progress_callbacks
        self._started = False

    def configure_runtime(self, *, completer: Completer, progress_callbacks: ProgressCallbacks) -> None:
        self._completer = completer
        self._progress_callbacks = progress_callbacks

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
            completion = await self._completer(
                list(message.history),
                model=message.model,
                tools=[_capability_to_schema_tool(item) for item in message.tool_capabilities],
                callbacks=self._progress_callbacks,
            )
            await self._send_response(
                message,
                LLMCompleteStepResponse(
                    request_id=message.request_id,
                    message=completion.message,
                    usage=completion.usage,
                ),
            )
        except BaseException as exc:
            await self._send_response(
                message,
                LLMCompleteStepResponse(request_id=message.request_id, message=None, usage=None, error=exc),
            )
        return None

    async def _send_response(self, request: LLMCompleteStepRequest, response: LLMCompleteStepResponse) -> None:
        if self._actor_directory is None:
            raise RuntimeError("LLMActor cannot send by URI without actor directory.")
        await self._actor_directory.send_message(uri=request.reply_to_uri, message=response)
