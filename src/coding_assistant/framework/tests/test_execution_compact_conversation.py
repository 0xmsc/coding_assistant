from typing import Any
import json

import pytest

from coding_assistant.llm.types import NullProgressCallbacks
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.history import append_assistant_message
from coding_assistant.llm.types import AssistantMessage, ToolResult, UserMessage, message_to_dict
from coding_assistant.framework.tests.helpers import (
    FakeCompleter,
    FakeMessage,
    FunctionCall,
    ToolCall,
    agent_actor_scope,
    append_tool_call_results_to_history,
    execute_tool_calls_via_messages,
    make_test_agent,
    make_ui_mock,
    tool_call_actor_scope,
)
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult, TextResult
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
    tool_call = ToolCall(
        id="shorten-1",
        function=FunctionCall(name="compact_conversation", arguments=json.dumps({"summary": summary_text})),
    )

    msg = FakeMessage(tool_calls=[tool_call])

    def handle_tool_result(result: ToolResult) -> str:
        if isinstance(result, CompactConversationResult):
            return AgentActor.handle_compact_conversation_result(
                result, desc=desc, state=state, progress_callbacks=callbacks
            )
        return str(result)

    async with tool_call_actor_scope(
        tools=desc.tools,
        ui=make_ui_mock(),
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=state.history,
            execution_results=response.results,
            context_name=desc.name,
            progress_callbacks=callbacks,
            handle_tool_result=handle_tool_result,
        )

    assert any(force for content, force in callbacks.user_messages if summary_text in content)

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

    finish_call = ToolCall("finish-1", FunctionCall("finish_task", json.dumps({"result": "done", "summary": "sum"})))

    completer = FakeCompleter([FakeMessage(tool_calls=[finish_call])])

    async with agent_actor_scope(context_name=desc.name) as agent_actor:
        msg, _ = await agent_actor.do_single_step(
            history=state.history,
            model=desc.model,
            tools=desc.tools,
            progress_callbacks=callbacks,
            completer=completer,
            context_name=desc.name,
        )

    append_assistant_message(state.history, callbacks=callbacks, context_name=desc.name, message=msg)

    def handle_tool_result_2(result: ToolResult) -> str:
        if isinstance(result, FinishTaskResult):
            return AgentActor.handle_finish_task_result(result, state=state)
        if isinstance(result, TextResult):
            return result.content
        return str(result)

    async with tool_call_actor_scope(
        tools=desc.tools,
        ui=make_ui_mock(),
        context_name=desc.name,
        progress_callbacks=callbacks,
    ) as actor:
        response = await execute_tool_calls_via_messages(actor, message=msg)
        append_tool_call_results_to_history(
            history=state.history,
            execution_results=response.results,
            context_name=desc.name,
            progress_callbacks=callbacks,
            handle_tool_result=handle_tool_result_2,
        )

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
