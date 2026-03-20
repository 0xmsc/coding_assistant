from coding_assistant.llm.types import BaseMessage


def clear_history(history: list[BaseMessage]) -> None:
    if history:
        history[:] = [history[0]]
