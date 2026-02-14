from typing import Any
import json

import pytest

from coding_assistant.llm.types import NullProgressCallbacks, AssistantMessage, UserMessage, message_to_dict
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FakeMessage,
    FunctionCall,
    ToolCall,
    make_test_agent,
    make_ui_mock,
    run_agent_via_messages,
    system_actor_scope_for_tests,
)
from coding_assistant.framework.types import AgentContext
from coding_assistant.framework.builtin_tools import FinishTaskTool, CompactConversationTool as CompactConversation


@pytest.mark.asyncio
async def test_compact_conversation_resets_history() -> None:
    desc, state = make_test_agent(
        tools=[FinishTaskTool(), CompactConversation()],
        history=[
            UserMessage(content="old start"),
            AssistantMessage(content="old reply"),
        ],
    )

    class SpyCallbacks(NullProgressCallbacks):
        def __init__(self) -> None:
            self.user_messages: Any = []

        def on_user_message(self, context_name: str, message: UserMessage, *, force: bool = False) -> Any:
            content = message.content if isinstance(message.content, str) else str(message.content)
            self.user_messages.append((content, force))

    callbacks = SpyCallbacks()
    summary_text = "This is the summary of prior conversation."
    compact_call = ToolCall(
        id="shorten-1",
        function=FunctionCall(name="compact_conversation", arguments=json.dumps({"summary": summary_text})),
    )
    finish_call = ToolCall("finish-1", FunctionCall("finish_task", json.dumps({"result": "done", "summary": "sum"})))
    completer = FakeCompleter([FakeMessage(tool_calls=[compact_call]), FakeMessage(tool_calls=[finish_call])])

    async with system_actor_scope_for_tests(
        tools=desc.tools,
        ui=make_ui_mock(),
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actors:
        await run_agent_via_messages(
            actors,
            ctx=AgentContext(desc=desc, state=state),
            tools=desc.tools,
            progress_callbacks=callbacks,
            completer=completer,
            compact_conversation_at_tokens=200_000,
        )

    assert state.output is not None
    assert state.output.result == "done"
    assert any(force for content, force in callbacks.user_messages if summary_text in content)

    assert message_to_dict(state.history[0]) == {
        "role": "user",
        "content": "old start",
    }
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
    assert message_to_dict(state.history[3]) == {
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
    assert message_to_dict(state.history[4]) == {
        "tool_call_id": "finish-1",
        "role": "tool",
        "name": "finish_task",
        "content": "Agent output set.",
    }
