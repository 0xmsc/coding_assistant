from coding_assistant.llm.custom import complete as custom_complete
from coding_assistant.llm.litellm import complete as litellm_complete
from coding_assistant.framework.types import Completer


def get_completer(name: str) -> Completer:
    if name == "custom":
        return custom_complete
    return litellm_complete
