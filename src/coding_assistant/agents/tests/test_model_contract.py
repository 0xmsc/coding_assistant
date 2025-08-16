import json
import pytest

from coding_assistant.agents.callbacks import NullCallbacks
from coding_assistant.agents.execution import do_single_step
from coding_assistant.agents.types import Agent, TextResult
from coding_assistant.llm.model import Completion
from coding_assistant.tools.tools import FinishTaskTool, ShortenConversation


class FakeFunction:
    def __init__(self, name: str, arguments: str = "{}"):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, id: str, function: FakeFunction):
        self.id = id
        self.function = function


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list[FakeToolCall] | None = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        data = {"role": self.role}
        if self.content is not None:
            data["content"] = self.content
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return data

    def model_dump_json(self):
        return json.dumps(self.model_dump())


class FakeCompleter:
    def __init__(self, message: FakeMessage, tokens: int = 10):
        self.message = message
        self.tokens = tokens

    async def __call__(self, messages, model, tools, callbacks):
        return Completion(message=self.message, tokens=self.tokens)


class DummyTool:
    def name(self):
        return "dummy"

    def description(self):
        return ""

    def parameters(self):
        return {}

    async def execute(self, parameters):
        return TextResult(content="ok")


@pytest.mark.asyncio
async def test_do_single_step_adds_shorten_prompt_on_token_threshold(monkeypatch):
    # Build a message that has no tool calls (so we also test that branch), with high token usage
    fake_message = FakeMessage(content="hi")
    completer = FakeCompleter(fake_message, tokens=999999)
    monkeypatch.setattr("coding_assistant.agents.execution.complete", completer, raising=True)

    async def _no_feedback(_: Agent):
        return None

    agent = Agent(
        name="A",
        model="fake",
        description="",
        parameters=[],
        feedback_function=_no_feedback,
        tools=[DummyTool(), FinishTaskTool(), ShortenConversation()],
        mcp_servers=[],
        tool_confirmation_patterns=[],
        history=[{"role": "user", "content": "start"}],
    )

    # Running a single step should append the assistant message and then a user message prompting to shorten
    msg = await do_single_step(
        agent,
        NullCallbacks(),
        shorten_conversation_at_tokens=1000,  # much below completer.tokens
        no_truncate_tools=set(),
    )

    assert msg.content == "hi"
    # The last user message should be the shorten instruction
    assert any(
        h.get("role") == "user" and "summarize" in h.get("content", "") for h in agent.history
    )