import json

import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.history import (
    append_assistant_message,
)
from coding_assistant.framework.execution import (
    do_single_step,
    handle_tool_calls,
)
from coding_assistant.framework.agent import (
    _handle_compact_conversation_result,
    _handle_finish_task_result,
)
from coding_assistant.framework.models import UserMessage, AssistantMessage
from coding_assistant.framework.tests.helpers import (
    FakeFunction,
    FakeMessage,
    FakeToolCall,
    FakeCompleter,
    make_test_agent,
    make_ui_mock,
    message_to_dict,
)
from coding_assistant.framework.types import ToolResult, FinishTaskResult, CompactConversationResult, TextResult
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


@pytest.mark.asyncio
async def test_compact_conversation_resets_history():
    # Prepare agent with some existing history that should be cleared
    desc, state = make_test_agent(
        tools=[FinishTaskTool(), CompactConversation()],
        history=[
            UserMessage(content="old start"),
            AssistantMessage(content="old reply"),
        ],
    )

    class SpyCallbacks(NullProgressCallbacks):
        def __init__(self):
            self.user_messages = []

        def on_user_message(self, context_name: str, content: str, force: bool = False):
            self.user_messages.append((content, force))

    callbacks = SpyCallbacks()

    # Invoke compact_conversation tool directly
    summary_text = "This is the summary of prior conversation."
    tool_call = FakeToolCall(
        id="shorten-1",
        function=FakeFunction(
            name="compact_conversation",
            arguments=json.dumps({"summary": summary_text}),
        ),
    )

    msg = FakeMessage(tool_calls=[tool_call])

    def handle_tool_result(result: ToolResult) -> str:
        if isinstance(result, CompactConversationResult):
            return _handle_compact_conversation_result(result, desc, state, callbacks)
        return str(result)

    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        callbacks,
        tool_callbacks=NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
        handle_tool_result=handle_tool_result,
    )

    # Verify that the summary user message was forced
    assert any(force for content, force in callbacks.user_messages if summary_text in content)

    # History should be reset to keeping the first message + summary message, followed by the tool result message
    assert len(state.history) >= 3
    assert state.history[0] == UserMessage(content="old start")

    assert message_to_dict(state.history[1]) == {
        "role": "user",
        "content": (
            f"A summary of your conversation with the client until now:\n\n{summary_text}\n\nPlease continue your work."
        ),
    }

    assert message_to_dict(state.history[2]) == {
        "tool_call_id": "shorten-1",
        "role": "tool",
        "name": "compact_conversation",
        "content": "Conversation compacted and history reset.",
    }

    # Subsequent steps should continue from the new history
    finish_call = FakeToolCall(
        "finish-1",
        FakeFunction(
            "finish_task",
            json.dumps({"result": "done", "summary": "sum"}),
        ),
    )

    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    msg, _ = await do_single_step(
        state.history,
        desc.model,
        desc.tools,
        callbacks,
        completer=completer,
        context_name=desc.name,
    )

    # Append assistant message to history
    append_assistant_message(state.history, callbacks, desc.name, msg)

    def handle_tool_result_2(result: ToolResult) -> str:
        if isinstance(result, FinishTaskResult):
            return _handle_finish_task_result(result, state)
        if isinstance(result, TextResult):
            return result.content
        return str(result)

    await handle_tool_calls(
        msg,
        desc.tools,
        state.history,
        callbacks,
        NullToolCallbacks(),
        ui=make_ui_mock(),
        context_name=desc.name,
        handle_tool_result=handle_tool_result_2,
    )

    # Verify the assistant tool call and finish result were appended after the reset messages
    assert message_to_dict(state.history[-2]) == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "finish-1",
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "arguments": '{"result": "done", "summary": "sum"}',
                },
            }
        ],
    }
    assert message_to_dict(state.history[-1]) == {
        "tool_call_id": "finish-1",
        "role": "tool",
        "name": "finish_task",
        "content": "Agent output set.",
    }
