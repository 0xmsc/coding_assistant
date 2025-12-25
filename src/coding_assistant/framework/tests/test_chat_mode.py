import json
import pytest

from coding_assistant.llm.types import UserMessage
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FunctionCall,
    FakeMessage,
    ToolCall,
    make_ui_mock,
)
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.builtin_tools import CompactConversationTool as CompactConversation
from coding_assistant.framework.types import Tool, TextResult
from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks


class FakeEchoTool(Tool):
    def __init__(self):
        self.called_with = None

    def name(self) -> str:
        return "fake.echo"

    def description(self) -> str:
        return "Echo a provided text"

    def parameters(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    async def execute(self, parameters: dict) -> TextResult:
        self.called_with = parameters
        return TextResult(content=f"echo: {parameters['text']}")


@pytest.mark.asyncio
async def test_chat_step_prompts_user_on_no_tool_calls_once():
    completer = FakeCompleter([FakeMessage(content="Hello")])
    history = [UserMessage(content="start")]
    tools = []
    model = "test-model"
    parameters = []

    ui = make_ui_mock(ask_sequence=[("> ", "User reply"), ("> ", "User reply 2")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            parameters=parameters,
            context_name="test",
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    roles = [m.role for m in history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_step_executes_tools_without_prompt():
    echo_call = ToolCall("1", FunctionCall("fake.echo", json.dumps({"text": "hi"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[echo_call])])

    echo_tool = FakeEchoTool()
    history = [UserMessage(content="start")]
    tools = [echo_tool]
    model = "test-model"
    parameters = []

    ui = make_ui_mock(ask_sequence=[("> ", "Hi")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            parameters=parameters,
            context_name="test",
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    assert echo_tool.called_with == {"text": "hi"}


@pytest.mark.asyncio
async def test_chat_mode_does_not_require_finish_task_tool():
    completer = FakeCompleter([FakeMessage(content="Hi there")])
    history = [UserMessage(content="start")]
    tools = []
    model = "test-model"
    parameters = []

    ui = make_ui_mock(ask_sequence=[("> ", "Ack"), ("> ", "Ack 2")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            parameters=parameters,
            context_name="test",
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    roles = [m.role for m in history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_exit_command_stops_loop_without_appending_command():
    completer = FakeCompleter([FakeMessage(content="Hello chat")])
    history = [UserMessage(content="start")]
    tools = []
    model = "test-model"
    parameters = []

    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    await run_chat_loop(
        history=history,
        model=model,
        tools=tools,
        parameters=parameters,
        context_name="test",
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    assert not any(m.role == "user" and (m.content or "").strip() == "/exit" for m in history)
    assert history[-1].role == "user"


@pytest.mark.asyncio
async def test_chat_loop_prompts_after_compact_command():
    # Sequence:
    # 1. User enters /compact -> calls _compact_cmd -> appends message, returns PROCEED_WITH_MODEL
    # 2. Model responds with tool_call compact_conversation
    # 3. Tool executes
    # 4. LOOP SHOULD PROMPT USER

    compact_call = ToolCall("1", FunctionCall("compact_conversation", json.dumps({"summary": "Compacted"})))
    completer = FakeCompleter(
        [FakeMessage(tool_calls=[compact_call]), FakeMessage(content="Should not be reached autonomously")]
    )

    compact_tool = CompactConversation()
    history = [UserMessage(content="start")]
    tools = [compact_tool]
    model = "test-model"
    parameters = []

    ui = make_ui_mock(ask_sequence=[("> ", "/compact"), ("> ", "/exit")])

    await run_chat_loop(
        history=history,
        model=model,
        tools=tools,
        parameters=parameters,
        context_name="test",
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    assert ui.prompt.call_count == 2
    assert history[-1].role == "tool"
    assert "compacted" in history[-1].content.lower()


@pytest.mark.asyncio
async def test_chat_compact_conversation_not_forced_in_callbacks():
    compact_call = ToolCall("1", FunctionCall("compact_conversation", json.dumps({"summary": "Compacted summary"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[compact_call])])

    compact_tool = CompactConversation()
    history = [UserMessage(content="start")]
    tools = [compact_tool]
    model = "test-model"
    parameters = []

    class SpyCallbacks(NullProgressCallbacks):
        def __init__(self):
            self.user_messages = []

        def on_user_message(self, context_name: str, content: str, force: bool = False):
            self.user_messages.append((content, force))

    callbacks = SpyCallbacks()
    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    # or just use handle_tool_call logic via run_chat_loop.

    # or just let it start and provide a real message from UI.

    ui = make_ui_mock(ask_sequence=[("> ", "Please compact"), ("> ", "/exit")])

    with pytest.raises(AssertionError, match="Completer script exhausted"):
        await run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            parameters=parameters,
            context_name="test",
            callbacks=callbacks,
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    summary_user_msg = next((c, f) for c, f in callbacks.user_messages if "Compacted summary" in c)
    assert summary_user_msg[1] is False, "Summary message should not be forced in chat mode"
