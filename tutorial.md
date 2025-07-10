# Coding Assistant Detailed Tutorial

**File path: `tutorial.md`**

This tutorial explains the codebase with actual excerpts. Junior developers will learn by example.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Configuration (`config.py`)](#configuration-configpy)
3. [CLI Entry Point (`main.py`)](#cli-entry-point-mainpy)
4. [MCP Servers Setup (`tools.py`)](#mcp-servers-setup-toolspy)
5. [LLM Integration (`model.py`)](#llm-integration-modelpy)
6. [Agent Core (`logic.py`)](#agent-core-logicpy)
   - `Agent` dataclass
   - `run_agent_loop`
   - `do_single_step`
7. [Agent Tools (`agents/tools.py`)](#agent-tools-agentstoolspy)
   - OrchestratorTool
   - ResearchTool
   - DevelopTool
   - AskUserTool
   - FeedbackTool
8. [Example: File Creation with MCP](#example-file-creation-with-mcp)
9. [Example: Agent Loop in Action](#example-agent-loop-in-action)
10. [Summary](#summary)

---

## Project Structure

```
.
├── tutorial.md          # This tutorial
├── README.md            # Project overview
├── src/
│   └── coding_assistant/
│       ├── config.py
│       ├── main.py
│       ├── tools.py
│       ├── llm/
│       │   └── model.py
│       └── agents/
│           ├── logic.py
│           └── tools.py
└── pyproject.toml
```

---

## Configuration (`config.py`)

**Path:** `src/coding_assistant/config.py`

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    working_directory: Path   # Base path for filesystem/git operations
    model: str                # Default LLM model (e.g., "o4-mini")
    expert_model: str | None  # Optional expert model (e.g., "o3")
```

- **working_directory**: Where MCP filesystem and git servers point.
- **model**: Default LLM model name.
- **expert_model**: Secondary expert-level model name.

---

## CLI Entry Point (`main.py`)

**Path:** `src/coding_assistant/main.py`

```python
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--task", type=str, help="Task for the orchestrator agent."
    )
    parser.add_argument(
        "--research", type=str, help="Question for the research agent."
    )
    parser.add_argument(
        "--expert", type=str, help="Task for the expert developer agent."
    )
    parser.add_argument(
        "-w", "--working_directory", type=Path,
        default=Path(os.getcwd()),
    )
    return parser.parse_args()
```

```python
async def _main():
    args = parse_args()
    config = load_config(args)
    os.chdir(config.working_directory)

    async with get_all_mcp_servers(config) as mcp_servers:
        tools = Tools(mcp_servers=mcp_servers)
        if args.task:
            tool = OrchestratorTool(config, tools)
            result = await tool.execute({"task": args.task})
            print(f"Finished with: {result}")
        else:
            sys.exit(1)
```

- Parses CLI flags (`--task`, `--research`, `--expert`).
- Loads configuration and changes into the working directory.
- Launches all MCP servers.
- Instantiates the appropriate top-level tool and prints its result.

---

## MCP Servers Setup (`tools.py`)

**Path:** `src/coding_assistant/tools.py`

```python
@asynccontextmanager
async def get_filesystem_server(config: Config):
    async with _get_mcp_server(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            str(config.working_directory),
        ],
    ) as server:
        yield server
```

```python
@asynccontextmanager
async def get_all_mcp_servers(config: Config):
    if not config.working_directory.exists():
        raise ValueError("Working directory does not exist.")

    servers: List[MCPServer] = []
    async with AsyncExitStack() as stack:
        servers.append(await stack.enter_async_context(get_filesystem_server(config)))
        servers.append(await stack.enter_async_context(get_git_server(config)))
        servers.append(await stack.enter_async_context(get_fetch_server()))
        servers.append(await stack.enter_async_context(get_memory_server(config)))
        servers.append(await stack.enter_async_context(get_shell_server()))
        if os.environ.get("TAVILY_API_KEY"):
            servers.append(await stack.enter_async_context(get_tavily_server()))
        yield servers
```

- Each `get_*_server` function spins up an MCP server via `npx` or `uvx`.
- `get_all_mcp_servers` collects them into a list for agent use.

---

## LLM Integration (`model.py`)

**Path:** `src/coding_assistant/llm/model.py`

```python
import litellm

def complete(
    messages: list[dict],
    tools: list = [],
    model: str = "o4-mini",
) -> dict:
    completion = litellm.completion(
        messages=messages,
        tools=tools,
        model=model,
        reasoning_effort="high",
    )
    return completion["choices"][0]["message"]
```

- Calls `litellm.completion` with messages and tool schemas.
- Returns the first message object for further processing.

---

## Agent Core (`logic.py`)

**Path:** `src/coding_assistant/agents/logic.py`

### `Agent` dataclass
```python
@dataclass
class Agent:
    name: str
    model: str
    description: str
    parameters: list[Parameter]
    tools: list[Tool]
    mcp_servers: list[MCPServer]
    history: list = field(default_factory=list)
    result: str | None = None
    feedback_function: Callable = None
```

- Maintains agent context, tool definitions, chat history, and final `result`.

### `run_agent_loop`
```python
async def run_agent_loop(agent: Agent) -> str:
    # Ensure finish_task is available
    if not any(t.name() == "finish_task" for t in agent.tools):
        agent.tools.append(FinishTaskTool(agent))

    while not agent.result:
        await do_single_step(agent)

    # Optionally gather feedback and repeat
    return agent.result
```

- Loops until the agent calls the `finish_task` tool, setting `agent.result`.

### `do_single_step`
```python
async def do_single_step(agent: Agent):
    # Step 1: Add system or continuation messages
    if not agent.history:
        system_message = create_system_message(agent)
        agent.history.append({"role": "system", "content": system_message})

    # Step 2: Call LLM with available tools
    completion = complete(agent.history, model=agent.model, tools=tools)
    message = completion["choices"][0]["message"]
    agent.history.append(message.model_dump())

    # Step 3: Dispatch any tool calls
    for tool_call in message.tool_calls or []:
        await handle_tool_call(tool_call, agent)

    return message
```

- Builds on chat history and uses function-calling-enhanced prompts.
- Dispatches MCP and custom tools via `handle_tool_call`.

---

## Agent Tools (`agents/tools.py`)

**Path:** `src/coding_assistant/agents/tools.py`

### `OrchestratorTool`
```python
class OrchestratorTool(Tool):
    def name(self): return "launch_orchestrator_agent"
    def description(self): ...
    def parameters(self):
        return {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}

    async def execute(self, parameters):
        orchestrator_agent = Agent(
            name="Orchestrator",
            description=self.description(),
            parameters=fill_parameters(self.parameters(), parameters),
            mcp_servers=self._tools.mcp_servers,
            tools=[ResearchTool(self._config, self._tools), DevelopTool(self._config, self._tools), AskUserTool()],
            model=self._config.model,
            feedback_function=...,  # Interactive and agent feedback
        )
        return await run_agent_loop(orchestrator_agent)
```

### `ResearchTool`
```python
class ResearchTool(Tool):
    def name(self): return "launch_research_agent"
    def parameters(self): ...
    async def execute(self, parameters):
        researcher = Agent(..., tools=[], ...)
        return await run_agent_loop(researcher)
```

### `DevelopTool`
```python
class DevelopTool(Tool):
    def name(self): return "launch_developer_agent"
    async def execute(self, parameters):
        developer = Agent(..., tools=[], ...)
        return await run_agent_loop(developer)
```

### `AskUserTool`
```python
class AskUserTool(Tool):
    def name(self): return "ask_user"
    async def execute(self, parameters):
        from rich.prompt import Prompt
        return Prompt.ask(parameters["question"], default=parameters.get("default_answer"))
```

### `FeedbackTool`
```python
class FeedbackTool(Tool):
    def name(self): return "launch_feedback_agent"
    async def execute(self, parameters):
        feedback_agent = Agent(..., tools=[], ...)
        return await run_agent_loop(feedback_agent)
```

---

## Example: File Creation with MCP

Agents can call the filesystem tool:

```json
{
  "name": "mcp_filesystem_write_file",
  "arguments": "{\"path\": \"tutorial.md\", \"content\": \"# Hello World\\nThis is a test.\"}"
}
```

This will create or overwrite `tutorial.md` at the root.

---

## Example: Agent Loop in Action

1. **System Message** sent to LLM:
   ```txt
   You are an agent named `Orchestrator`.
   ## Task
   Your task is to ...
   ```
2. **LLM Response** with a tool call:
   ```json
   {
     "tool_calls": [
       {
         "name": "mcp_filesystem_list_directory",
         "arguments": "{\"path\": \"src\"}"
       }
     ]
   }
   ```
3. **MCP Server** returns directory listing.
4. **Agent** continues planning or calls `launch_developer_agent` next.

---

## Summary

- We provided **file paths** and **code snippets** for each module.
- Explained how **agents** and **tools** interact with the MCP.
- Showed practical **examples** of function-calling and file operations.
- You should now feel comfortable reading, using, and extending the codebase.
