from typing import TYPE_CHECKING
from coding_assistant.llm.litellm import complete as litellm_complete

if TYPE_CHECKING:
    from coding_assistant.framework.types import Completer


def get_completer(name: str) -> "Completer":
    if name == "openai":
        return openai_complete
    return litellm_complete
