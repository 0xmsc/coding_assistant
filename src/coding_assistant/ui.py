import logging
from abc import ABC, abstractmethod

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import create_confirm_session
from rich.console import Console

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
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        Console().bell()
        print(prompt_text)
        return await PromptSession().prompt_async("> ", default=default or "")

    async def confirm(self, prompt_text: str) -> bool:
        Console().bell()
        return await create_confirm_session(prompt_text).prompt_async()

    async def prompt(self) -> str:
        Console().bell()
        return await PromptSession().prompt_async("> ")


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
