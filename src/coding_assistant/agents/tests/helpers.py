from dataclasses import dataclass
from coding_assistant.agents.types import Agent


@dataclass
class FakeFunction:
    name: str
    arguments: str = "{}"


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


async def no_feedback(_: Agent):
    """A feedback function that returns no feedback (used in tests)."""
    return None
