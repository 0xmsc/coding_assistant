from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel

from coding_assistant.logging import print_agent_progress

console = Console()


def run_agent(agent, task, name=None):
    console.print(Panel(f"{name}: {task}", title="Task", border_style="green"))
    config = {"configurable": {"thread_id": "thread"}}
    input = {"messages": HumanMessage(content=task)}

    latest = None
    for chunk in agent.stream(input=input, config=config):
        print_agent_progress(chunk, name=name)

        if "agent" in chunk and "messages" in chunk["agent"]:
            for message in chunk["agent"]["messages"]:
                if isinstance(message, AIMessage):
                    latest = message

    return latest.content
