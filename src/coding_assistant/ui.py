import logging
import re
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Any, cast

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


@dataclass(slots=True)
class _Ask:
    prompt_text: str
    default: str | None


@dataclass(slots=True)
class _Confirm:
    prompt_text: str


@dataclass(slots=True)
class _Prompt:
    words: list[str] | None


_UIMessage = _Ask | _Confirm | _Prompt


class ActorUI(UI):
    def __init__(self, ui: UI, *, context_name: str = "ui") -> None:
        self._ui = ui
        self._actor: Actor[_UIMessage, Any] = Actor(name=f"{context_name}.ui", handler=self._handle_message)
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
        self.start()
        return cast(str, await self._actor.ask(_Ask(prompt_text=prompt_text, default=default)))

    async def confirm(self, prompt_text: str) -> bool:
        self.start()
        return cast(bool, await self._actor.ask(_Confirm(prompt_text=prompt_text)))

    async def prompt(self, words: list[str] | None = None) -> str:
        self.start()
        return cast(str, await self._actor.ask(_Prompt(words=words)))

    async def _handle_message(self, message: _UIMessage) -> Any:
        if isinstance(message, _Ask):
            return await self._ui.ask(message.prompt_text, default=message.default)
        if isinstance(message, _Confirm):
            return await self._ui.confirm(message.prompt_text)
        if isinstance(message, _Prompt):
            return await self._ui.prompt(message.words)
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
