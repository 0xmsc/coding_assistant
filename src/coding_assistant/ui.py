import asyncio
import logging
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Generic, TypeVar
from uuid import uuid4

from typing import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import create_confirm_session
from rich import print
from rich.console import Console
from rich.rule import Rule

from coding_assistant.framework.actor_runtime import Actor
from coding_assistant.framework.actors.common.messages import (
    AskRequest,
    AskResponse,
    ConfirmRequest,
    ConfirmResponse,
    PromptRequest,
    PromptResponse,
)
from coding_assistant.paths import get_app_cache_dir

logger = logging.getLogger(__name__)


class UI(ABC):
    @abstractmethod
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        pass

    @abstractmethod
    async def confirm(self, prompt_text: str) -> bool:
        pass

    @abstractmethod
    async def prompt(self, words: list[str] | None = None) -> str:
        pass


class SlashCompleter(Completer):
    def __init__(self, words: list[str]):
        # We use a custom completer that only triggers when the user types '/'
        # and matches the full word including the slash.
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


class PromptToolkitUI(UI):
    def __init__(self) -> None:
        history_dir = get_app_cache_dir()
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "history"
        self._session: PromptSession[str] = PromptSession(
            history=FileHistory(str(history_file)),
            enable_open_in_editor=True,
        )

    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        Console().bell()
        print(Rule(style="dim"))
        print(prompt_text)
        return await self._session.prompt_async("> ", default=default or "")

    async def confirm(self, prompt_text: str) -> bool:
        Console().bell()
        print(Rule(style="dim"))
        return await create_confirm_session(prompt_text).prompt_async()

    async def prompt(self, words: list[str] | None = None) -> str:
        Console().bell()
        print(Rule(style="dim"))
        completer = SlashCompleter(words) if words else None
        return await self._session.prompt_async("> ", completer=completer, complete_while_typing=True)


class DefaultAnswerUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        logger.info(f"Skipping user input for prompt: {prompt_text}")
        return default or "UI is not available. Assume the user gave the most sensible answer."

    async def confirm(self, prompt_text: str) -> bool:
        logger.info(f"Skipping user confirmation for prompt: {prompt_text}")
        return True

    async def prompt(self, words: list[str] | None = None) -> str:
        logger.info("Skipping user input for generic prompt")
        return "UI is not available. Assume the user provided the most sensible input."


class NullUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        raise RuntimeError("No UI available")

    async def confirm(self, prompt_text: str) -> bool:
        raise RuntimeError("No UI available")

    async def prompt(self, words: list[str] | None = None) -> str:
        raise RuntimeError("No UI available")


_UIMessage = AskRequest | ConfirmRequest | PromptRequest
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
    def __init__(self, ui: UI, *, context_name: str = "ui") -> None:
        self._ui = ui
        self._actor: Actor[_UIMessage] = Actor(name=f"{context_name}.ui", handler=self._handle_message)
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
        return message.value

    async def _handle_message(self, message: _UIMessage) -> None:
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
                await message.reply_to.send_message(
                    PromptResponse(request_id=message.request_id, value=prompt_response)
                )
            except BaseException as exc:
                await message.reply_to.send_message(
                    PromptResponse(request_id=message.request_id, value=None, error=exc)
                )
            return None
        raise RuntimeError(f"Unknown UI message: {message!r}")


class UserActor(ActorUI):
    def __init__(self, ui: UI, *, context_name: str = "user") -> None:
        super().__init__(ui, context_name=context_name)


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
