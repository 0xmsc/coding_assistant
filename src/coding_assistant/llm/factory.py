from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.llm.litellm import complete as litellm_complete
from coding_assistant.framework.types import Completer


def get_completer(name: str) -> Completer:
    if name == "openai":
        return openai_complete
    return litellm_complete
