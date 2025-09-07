from abc import ABC, abstractmethod
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import create_confirm_session

logger = logging.getLogger(__name__)


class UI(ABC):
    @abstractmethod
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        pass

    @abstractmethod
    async def confirm(self, prompt_text: str) -> bool:
        pass


class PromptToolkitUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        print(prompt_text)
        return await PromptSession().prompt_async("> ", default=default or "")

    async def confirm(self, prompt_text: str) -> bool:
        return await create_confirm_session(prompt_text).prompt_async()


class DefaultAnswerUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        logger.info(f"Skipping user input for prompt: {prompt_text}")
        return default or "UI is not available."

    async def confirm(self, prompt_text: str) -> bool:
        logger.info(f"Skipping user confirmation for prompt: {prompt_text}")
        return False


class NullUI(UI):
    async def ask(self, prompt_text: str, default: str | None = None) -> str:
        raise RuntimeError("No UI available")

    async def confirm(self, prompt_text: str) -> bool:
        raise RuntimeError("No UI available")
