from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from pprint import pformat

from coding_assistant.config import get_global_config
from coding_assistant.logging import print_agent_progress

console = Console()


def create_context_prunning_prompt_function(system_prompt: str):
    def context_prunning_prompt(state):
        current_messages = [SystemMessage(content=system_prompt)] + state["messages"]

        current_messages = trim_messages(
            current_messages,
            strategy="last",
            token_counter=get_global_config().model_factory(),
            max_tokens=50_000,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=False,
        )

        return current_messages

    return context_prunning_prompt


def run_agent(agent, task, name, ask_user_for_feedback=False):
    console.print(Panel(Markdown(task), title=f"Agent task: {name}", border_style="green"))
    config = {"configurable": {"thread_id": "thread"}}
    input = {"messages": HumanMessage(content=task)}

    latest = None

    while True:
        # Do one round of the agent, until it stops
        for chunk in agent.stream(input=input, config=config):
            print_agent_progress(chunk, name=name)

            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    if isinstance(message, AIMessage):
                        latest = message

        if not ask_user_for_feedback:
            break

        # Ask user for feedback
        feedback = Prompt.ask("Feedback")
        input = {"messages": HumanMessage(content=feedback)}

    return latest.content
