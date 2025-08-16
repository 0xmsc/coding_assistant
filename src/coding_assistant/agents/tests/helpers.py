from dataclasses import dataclass
from typing import Iterable, Sequence

from coding_assistant.agents.parameters import Parameter
from coding_assistant.agents.types import Agent, Tool
from coding_assistant.tools.mcp import MCPServer


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


def make_test_agent(
    *,
    name: str = "TestAgent",
    model: str = "TestMode",
    description: str = "TestDescription",
    parameters: Sequence[Parameter] | None = None,
    feedback_function=no_feedback,
    tools: Iterable[Tool] | None = None,
    mcp_servers: list[MCPServer] | None = None,
    tool_confirmation_patterns: list[str] | None = None,
    history: list[dict] | None = None,
) -> Agent:
    """Create a minimal Agent instance for tests with sensible defaults.

    You can override any field via keyword arguments.
    """
    return Agent(
        name=name,
        model=model,
        description=description,
        parameters=list(parameters) if parameters is not None else [],
        feedback_function=feedback_function,
        tools=list(tools) if tools is not None else [],
        mcp_servers=list(mcp_servers) if mcp_servers is not None else [],
        tool_confirmation_patterns=list(tool_confirmation_patterns) if tool_confirmation_patterns is not None else [],
        history=list(history) if history is not None else [],
    )
