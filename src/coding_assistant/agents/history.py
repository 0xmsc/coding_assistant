from coding_assistant.agents.callbacks import AgentProgressCallbacks


def append_tool_message(
    history: list,
    callbacks: AgentProgressCallbacks,
    agent_name: str,
    tool_call_id: str,
    function_name: str,
    function_args: dict,
    function_call_result: str,
):
    callbacks.on_tool_message(agent_name, tool_call_id, function_name, function_args, function_call_result)

    history.append(
        {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": function_name,
            "content": function_call_result,
        }
    )


def append_user_message(history: list, callbacks: AgentProgressCallbacks, agent_name: str, content: str):
    callbacks.on_user_message(agent_name, content)

    history.append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(history: list, callbacks: AgentProgressCallbacks, agent_name: str, message):
    if message.content:
        callbacks.on_assistant_message(agent_name, message.content)

    message_dump = message.model_dump()
    history.append(message_dump)
