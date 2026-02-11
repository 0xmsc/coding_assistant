import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Generic, TypeVar
from uuid import uuid4

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    AgentYieldedToUser,
    AskRequest,
    AskResponse,
    ChatPromptInput,
    ClearHistoryRequested,
    CompactionRequested,
    ConfirmRequest,
    ConfirmResponse,
    HelpRequested,
    ImageAttachRequested,
    PromptRequest,
    PromptResponse,
    SessionExitRequested,
    UserInputFailed,
    UserTextSubmitted,
)
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


def _parse_chat_prompt_input(raw_input: str) -> ChatPromptInput:
    stripped = raw_input.strip()
    parts = stripped.split(maxsplit=1)
    command = parts[0] if parts else ""
    arg = parts[1] if len(parts) > 1 else None

    if command == "/exit":
        return SessionExitRequested()
    if command == "/compact":
        return CompactionRequested()
    if command == "/clear":
        return ClearHistoryRequested()
    if command == "/image":
        return ImageAttachRequested(source=arg.strip() if arg else None)
    if command == "/help":
        return HelpRequested()
    return UserTextSubmitted(text=raw_input)


_UIMessage = AgentYieldedToUser | AskRequest | ConfirmRequest | PromptRequest
ResponseT = TypeVar("ResponseT")


@dataclass(slots=True)
class _LocalReplySink(Generic[ResponseT]):
    future: asyncio.Future[ResponseT]
    extractor: Callable[[object], ResponseT | BaseException]

    async def send_message(self, message: object) -> None:
        try:
            value = self.extractor(message)
            if isinstance(value, BaseException):
                self.future.set_exception(value)
            else:
                self.future.set_result(value)
        except BaseException as exc:
            self.future.set_exception(exc)


class ActorUI(UI):
    def __init__(self, ui: UI, *, context_name: str = "ui", actor_directory: ActorDirectory | None = None) -> None:
        self._ui = ui
        self._actor: Actor[_UIMessage] = Actor(name=f"{context_name}.ui", handler=self._handle_message)
        self._actor_directory = actor_directory
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

    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        sink = _LocalReplySink[str](
            future=future,
            extractor=lambda msg: self._extract_ask_response(msg, request_id),
        )
        await self.send_message(
            AskRequest(request_id=request_id, prompt_text=prompt_text, default=default, reply_to=sink)
        )
        return await future

    async def confirm(self, prompt_text: str) -> bool:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        sink = _LocalReplySink[bool](
            future=future,
            extractor=lambda msg: self._extract_confirm_response(msg, request_id),
        )
        await self.send_message(ConfirmRequest(request_id=request_id, prompt_text=prompt_text, reply_to=sink))
        return await future

    async def prompt(self, words: list[str] | None = None) -> str:
        request_id = uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        sink = _LocalReplySink[str](
            future=future,
            extractor=lambda msg: self._extract_prompt_response(msg, request_id),
        )
        await self.send_message(PromptRequest(request_id=request_id, words=words, reply_to=sink))
        return await future

    async def send_message(self, message: _UIMessage) -> None:
        self.start()
        await self._actor.send(message)

    @staticmethod
    def _extract_ask_response(message: object, request_id: str) -> str | BaseException:
        if not isinstance(message, AskResponse):
            raise RuntimeError(f"Unexpected ask response type: {type(message).__name__}")
        if message.request_id != request_id:
            raise RuntimeError(f"Mismatched ask response id: {message.request_id}")
        if message.error is not None:
            return message.error
        if message.value is None:
            raise RuntimeError("Ask response missing value.")
        return message.value

    @staticmethod
    def _extract_confirm_response(message: object, request_id: str) -> bool | BaseException:
        if not isinstance(message, ConfirmResponse):
            raise RuntimeError(f"Unexpected confirm response type: {type(message).__name__}")
        if message.request_id != request_id:
            raise RuntimeError(f"Mismatched confirm response id: {message.request_id}")
        if message.error is not None:
            return message.error
        if message.value is None:
            raise RuntimeError("Confirm response missing value.")
        return message.value

    @staticmethod
    def _extract_prompt_response(message: object, request_id: str) -> str | BaseException:
        if not isinstance(message, PromptResponse):
            raise RuntimeError(f"Unexpected prompt response type: {type(message).__name__}")
        if message.request_id != request_id:
            raise RuntimeError(f"Mismatched prompt response id: {message.request_id}")
        if message.error is not None:
            return message.error
        if message.value is None:
            raise RuntimeError("Prompt response missing value.")
        if not isinstance(message.value, str):
            raise RuntimeError(f"Expected string prompt value, got {type(message.value).__name__}.")
        return message.value

    async def _handle_message(self, message: _UIMessage) -> None:
        if isinstance(message, AgentYieldedToUser):
            try:
                prompt_response = await self._ui.prompt(message.words)
                prompt_input = _parse_chat_prompt_input(prompt_response)
                if message.reply_to_uri is not None:
                    if self._actor_directory is None:
                        raise RuntimeError("UserActor cannot send by URI without actor directory.")
                    await self._actor_directory.send_message(uri=message.reply_to_uri, message=prompt_input)
                elif message.reply_to is not None:
                    await message.reply_to.send_message(prompt_input)
                else:
                    raise RuntimeError("AgentYieldedToUser is missing reply target.")
            except BaseException as exc:
                logger.exception("Failed to handle AgentYieldedToUser message: %s", exc)
                if message.reply_to_uri is not None:
                    if self._actor_directory is None:
                        raise RuntimeError("UserActor cannot send by URI without actor directory.")
                    await self._actor_directory.send_message(
                        uri=message.reply_to_uri, message=UserInputFailed(error=exc)
                    )
                elif message.reply_to is not None:
                    await message.reply_to.send_message(UserInputFailed(error=exc))
                else:
                    raise RuntimeError("AgentYieldedToUser is missing reply target.")
            return None
        if isinstance(message, AskRequest):
            try:
                ask_response = await self._ui.ask(message.prompt_text, default=message.default)
                await message.reply_to.send_message(AskResponse(request_id=message.request_id, value=ask_response))
            except BaseException as exc:
                await message.reply_to.send_message(AskResponse(request_id=message.request_id, value=None, error=exc))
            return None
        if isinstance(message, ConfirmRequest):
            try:
                confirm_response = await self._ui.confirm(message.prompt_text)
                await message.reply_to.send_message(
                    ConfirmResponse(request_id=message.request_id, value=confirm_response)
                )
            except BaseException as exc:
                await message.reply_to.send_message(
                    ConfirmResponse(request_id=message.request_id, value=None, error=exc)
                )
            return None
        if isinstance(message, PromptRequest):
            try:
                prompt_response = await self._ui.prompt(message.words)
                prompt_value: str | ChatPromptInput = prompt_response
                if message.parse_chat_intent:
                    prompt_value = _parse_chat_prompt_input(prompt_response)
                await message.reply_to.send_message(PromptResponse(request_id=message.request_id, value=prompt_value))
            except BaseException as exc:
                await message.reply_to.send_message(
                    PromptResponse(request_id=message.request_id, value=None, error=exc)
                )
            return None
        raise RuntimeError(f"Unknown UI message: {message!r}")


class UserActor(ActorUI):
    def __init__(self, ui: UI, *, context_name: str = "user", actor_directory: ActorDirectory | None = None) -> None:
        super().__init__(ui, context_name=context_name, actor_directory=actor_directory)


@asynccontextmanager
async def ui_actor_scope(ui: UI, *, context_name: str) -> AsyncIterator[UI]:
    if isinstance(ui, ActorUI):
        yield ui
        return
    actor_ui = ActorUI(ui, context_name=context_name)
    actor_ui.start()
    try:
        yield actor_ui
    finally:
        await actor_ui.stop()
