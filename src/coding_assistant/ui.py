import logging
import re
from abc import ABC, abstractmethod

from typing import Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import create_confirm_session
from rich.console import Console
from rich.rule import Rule

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
