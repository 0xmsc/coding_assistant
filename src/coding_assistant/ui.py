import logging
from abc import ABC, abstractmethod

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import create_confirm_session
from rich.console import Console

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
    async def prompt(self) -> str:
        pass


class PromptToolkitUI(UI):
    def __init__(self):
        history_dir = get_app_cache_dir()
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "history"
        self._session = PromptSession(
            history=FileHistory(str(history_file)),
            enable_open_in_editor=True,
        )

    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        Console().bell()
        print()
        print(prompt_text)
        return await self._session.prompt_async("> ", default=default or "")

    async def confirm(self, prompt_text: str) -> bool:
        Console().bell()
        print()
        return await create_confirm_session(prompt_text).prompt_async()

    async def prompt(self) -> str:
        Console().bell()
        print()
        return await self._session.prompt_async("> ")


class DefaultAnswerUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        logger.info(f"Skipping user input for prompt: {prompt_text}")
        return default or "UI is not available."

    async def confirm(self, prompt_text: str) -> bool:
        logger.info(f"Skipping user confirmation for prompt: {prompt_text}")
        return False

    async def prompt(self) -> str:
        logger.info("Skipping user input for generic prompt")
        return "UI is not available."


class NullUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        raise RuntimeError("No UI available")

    async def confirm(self, prompt_text: str) -> bool:
        raise RuntimeError("No UI available")

    async def prompt(self) -> str:
        raise RuntimeError("No UI available")
