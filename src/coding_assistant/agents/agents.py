import signal
import threading
import sys
import functools

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from pprint import pformat
from langchain_core.runnables import RunnableConfig

from coding_assistant.config import get_global_config
from coding_assistant.logging import print_agent_progress

console = Console()


class MyAgentState(AgentState):
    notebook: dict = {}
    task: str


def _format_system_prompt_with_current_state(system_prompt, state):
    notebook_facts = ""

    if not state["notebook"]:
        notebook_facts = "Empty"
    else:
        for key, fact in enumerate(state["notebook"]):
            notebook_facts += f"Key: {key}\nFact: {fact}\n\n"

    notebook_facts = notebook_facts.strip()
    return system_prompt.format(notebook_facts=notebook_facts, task=state["task"])


def create_context_prunning_prompt_function(system_prompt: str, system_message_type="system"):
    def context_prunning_prompt(state):
        nonlocal system_prompt
        nonlocal system_message_type

        system_prompt_with_state = _format_system_prompt_with_current_state(system_prompt=system_prompt, state=state)

        if system_message_type == "system":
            system_message = SystemMessage(content=system_prompt_with_state)
        elif system_message_type == "developer":
            # See this crap: https://github.com/langchain-ai/langchain/issues/28895
            system_message = SystemMessage(
                content=system_prompt_with_state,
                additional_kwargs={"__openai_role__": "developer"},
            )
        else:
            raise ValueError(f"Unknown system message type: {system_message_type}")

        current_messages = [system_message] + state["messages"]
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


def create_agent(prompt, tools, model):
    memory = MemorySaver()
    return create_react_agent(
        model,
        tools,
        checkpointer=memory,
        prompt=prompt,
        state_schema=MyAgentState,
    )


class InterruptibleSection(object):
    interrupt_requested: bool

    def __enter__(self):
        self.interrupt_requested = False
        if threading.current_thread() is threading.main_thread():
            self.original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, functools.partial(self._interrupt_handler))
        else:
            self.original_handler = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_handler:
            signal.signal(signal.SIGINT, self.original_handler)

    def _interrupt_handler(self, signum, frame):
        if self.interrupt_requested:
            # The user really seems to want to quit :O
            print(f" Interrupt requested two times, exiting...")
            sys.exit()

        print(f" Interrupt requested, waiting for next chance to interrupt agent...")
        self.interrupt_requested = True


def run_agent(agent, task, name, ask_user_for_feedback=False):
    console.print(Panel(Markdown(task), title=f"Agent task: {name}", border_style="green"))
    config = RunnableConfig(configurable={"thread_id": "thread"}, recursion_limit=50)
    input: MyAgentState = {"messages": HumanMessage(content=task), "notebook": dict(), "task": task}

    latest = None

    while True:
        # Do one round of the agent, until it stops (or is interrupted)
        with InterruptibleSection() as interruptible_section:
            for chunk in agent.stream(input=input, config=config):
                print_agent_progress(chunk, name=name)

                if "agent" in chunk and "messages" in chunk["agent"]:
                    messages = chunk["agent"]["messages"]

                    # Record the last AIMessage, it is the agents output
                    for message in messages:
                        if isinstance(message, AIMessage):
                            latest = message

                    # Check if we can interrupt the agent
                    # NOTE: We can only interrupt when the last message was an AIMessage **without** tool_calls
                    can_interrupt = messages[-1] and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls
                    if interruptible_section.interrupt_requested and can_interrupt:
                        print(f"Agent {name} has been interrupted, the last message was '{messages[-1].content}'")
                        interruptible_section.interrupt_requested = False
                        break

        if not ask_user_for_feedback:
            break

        # Ask user for feedback
        feedback = Prompt.ask("Feedback")
        input = {"messages": HumanMessage(content=feedback)}

    return latest.content
