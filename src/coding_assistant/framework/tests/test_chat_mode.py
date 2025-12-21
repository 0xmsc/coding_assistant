import json
import pytest

from coding_assistant.llm.types import UserMessage
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FunctionCall,
    FakeMessage,
    ToolCall,
    make_test_agent,
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
    # Assistant emits no tool calls -> in chat mode we should prompt the user once and append reply
    completer = FakeCompleter([FakeMessage(content="Hello")])
    desc, state = make_test_agent(tools=[], history=[UserMessage(content="start")])

    ui = make_ui_mock(ask_sequence=[("> ", "User reply"), ("> ", "User reply 2")])

    # Run a single chat-loop iteration by exhausting the completer after one step
    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            parameters=desc.parameters,
            context_name=desc.name,
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Should prompt first, then assistant responds, then prompt again
    roles = [m.role for m in state.history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_step_executes_tools_without_prompt():
    echo_call = ToolCall("1", FunctionCall("fake.echo", json.dumps({"text": "hi"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[echo_call])])

    echo_tool = FakeEchoTool()
    desc, state = make_test_agent(tools=[echo_tool], history=[UserMessage(content="start")])

    ui = make_ui_mock(ask_sequence=[("> ", "Hi")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            parameters=desc.parameters,
            context_name=desc.name,
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Tool must have executed
    assert echo_tool.called_with == {"text": "hi"}


@pytest.mark.asyncio
async def test_chat_mode_does_not_require_finish_task_tool():
    # No finish_task tool; chat mode should still allow a step
    completer = FakeCompleter([FakeMessage(content="Hi there")])
    desc, state = make_test_agent(tools=[], history=[UserMessage(content="start")])

    ui = make_ui_mock(ask_sequence=[("> ", "Ack"), ("> ", "Ack 2")])

    with pytest.raises(AssertionError, match="FakeCompleter script exhausted"):
        await run_chat_loop(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            parameters=desc.parameters,
            context_name=desc.name,
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Should be assistant followed by next user prompt
    roles = [m.role for m in state.history[-2:]]
    assert roles == ["assistant", "user"]


@pytest.mark.asyncio
async def test_chat_exit_command_stops_loop_without_appending_command():
    # Assistant sends a normal message, user replies with /exit which should stop the loop
    completer = FakeCompleter([FakeMessage(content="Hello chat")])
    desc, state = make_test_agent(tools=[], history=[UserMessage(content="start")])

    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    # Should return cleanly without exhausting the completer further
    await run_chat_loop(
        history=state.history,
        model=desc.model,
        tools=desc.tools,
        parameters=desc.parameters,
        context_name=desc.name,
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    # Verify that '/exit' was not appended to history
    assert not any(m.role == "user" and (m.content or "").strip() == "/exit" for m in state.history)
    # No assistant step should have happened; last message remains the start message
    assert state.history[-1].role == "user"


@pytest.mark.asyncio
async def test_chat_loop_prompts_after_compact_command():
    # Test that /compact command forces a user prompt after the next tool step
    # Even if that logic is autonomous by default

    # Sequence:
    # 1. User enters /compact -> calls _compact_cmd -> appends message, returns PROCEED_WITH_MODEL
    # 2. Model responds with tool_call compact_conversation
    # 3. Tool executes
    # 4. LOOP SHOULD PROMPT USER

    compact_call = ToolCall("1", FunctionCall("compact_conversation", json.dumps({"summary": "Compacted"})))
    # The first message comes from the model in response to the injected compact message
    # The second message is to check if it tries to loop again automatically (it shouldn't)
    completer = FakeCompleter(
        [FakeMessage(tool_calls=[compact_call]), FakeMessage(content="Should not be reached autonomously")]
    )

    compact_tool = CompactConversation()
    desc, state = make_test_agent(tools=[compact_tool], history=[UserMessage(content="start")])

    # Mock UI: first is /compact, second is /exit to stop the loop after verifying it prompted
    ui = make_ui_mock(ask_sequence=[("> ", "/compact"), ("> ", "/exit")])

    await run_chat_loop(
        history=state.history,
        model=desc.model,
        tools=desc.tools,
        parameters=desc.parameters,
        context_name=desc.name,
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
    )

    # If the logic works, ui.prompt was called twice
    assert ui.prompt.call_count == 2
    # Most recent history should be the tool result summary
    assert state.history[-1].role == "tool"
    assert "compacted" in state.history[-1].content.lower()


@pytest.mark.asyncio
async def test_chat_compact_conversation_not_forced_in_callbacks():
    # Test that when compact_conversation is called via model, its summary message is NOT forced in callbacks
    compact_call = ToolCall("1", FunctionCall("compact_conversation", json.dumps({"summary": "Compacted summary"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[compact_call])])

    compact_tool = CompactConversation()
    desc, state = make_test_agent(tools=[compact_tool], history=[UserMessage(content="start")])

    class SpyCallbacks(NullProgressCallbacks):
        def __init__(self):
            self.user_messages = []

        def on_user_message(self, context_name: str, content: str, force: bool = False):
            self.user_messages.append((content, force))

    callbacks = SpyCallbacks()
    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    # We expect one automated step from the model if we start with /compact,
    # or just use handle_tool_call logic via run_chat_loop.
    # To trigger it, let's just make the completer provide the compact call immediately.

    # Actually, we can just call run_chat_loop and it will call the model.
    # But wait, run_chat_loop starts with ui.prompt if history is empty or after UserMessage.

    # Let's trigger the loop by providing an initial history that ends with an assistant message,
    # or just let it start and provide a real message from UI.
    # If history is empty, run_chat_loop sets need_user_input = True.
    # If history has messages, it replays them and then need_user_input = True.

    ui = make_ui_mock(ask_sequence=[("> ", "Please compact"), ("> ", "/exit")])

    with pytest.raises(AssertionError, match="Completer script exhausted"):
        await run_chat_loop(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            parameters=desc.parameters,
            context_name=desc.name,
            callbacks=callbacks,
            tool_callbacks=NullToolCallbacks(),
            completer=completer,
            ui=ui,
        )

    # Find the summary message in callbacks
    summary_user_msg = next((c, f) for c, f in callbacks.user_messages if "Compacted summary" in c)
    assert summary_user_msg[1] is False, "Summary message should not be forced in chat mode"


@pytest.mark.asyncio
async def test_chat_loop_with_initial_prompt():
    # Setup:
    # 1. The agent is started with an initial prompt "Initial Prompt".
    # 2. The assistant responds with "Response".
    # 3. Then the loop should wait for user input. We provide "/exit".

    completer = FakeCompleter([FakeMessage(content="Response")])
    desc, state = make_test_agent(tools=[], history=[])

    # We expect ui.prompt to be called once for "/exit" because the initial prompt
    # should bypass the first user input requirement.
    ui = make_ui_mock(ask_sequence=[("> ", "/exit")])

    await run_chat_loop(
        history=state.history,
        model=desc.model,
        tools=desc.tools,
        parameters=desc.parameters,
        context_name=desc.name,
        callbacks=NullProgressCallbacks(),
        tool_callbacks=NullToolCallbacks(),
        completer=completer,
        ui=ui,
        initial_prompt="Initial Prompt",
    )

    user_messages = [m for m in state.history if m.role == "user"]
    assistant_messages = [m for m in state.history if m.role == "assistant"]

    # First user message is setup instructions, second is our initial prompt.
    assert len(user_messages) == 2
    assert "You are in chat mode" in user_messages[0].content
    assert user_messages[1].content == "Initial Prompt"

    assert len(assistant_messages) == 1
    assert assistant_messages[0].content == "Response"

    # Verify ui.prompt was called only once for /exit
    assert ui.prompt.call_count == 1

