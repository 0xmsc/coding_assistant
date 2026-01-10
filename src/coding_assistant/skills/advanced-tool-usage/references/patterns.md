# Advanced Orchestration Patterns

Using the `redirect_tool_call` meta-tool enables sophisticated workflows that were previously impossible due to context window limits.

## The Search-then-Process Pipeline

Commonly used with Search tools (like Tavily) or Web Browsing tools:

1. **Call**: `redirect_tool_call(tool_name="tavily_search", tool_args={"query": "..."}, output_file="search_raw.json")`
2. **Analysis**: Use `python_execute` to parse the JSON and extract specific URLs or snippets.
    ```python
    import json
    with open("search_raw.json") as f:
        data = json.load(f)
    # process data...
    ```

## The Log Analysis Pipeline

When analyzing production logs or large test outputs, prefer standard shell redirection for efficiency:

1. **Direct Redirect**: `shell_execute(command="journalctl -u service > logs.txt")`
2. **Filter**: `shell_execute(command="rg 'ERROR' logs.txt | head -n 20")`
3. **Report**: Summarize only the found errors.

## Advantages of Redirection vs redirect_tool_call

- **Standard Redirection**: Fastest for shell commands and command-line tools. Use whenever you are already inside `shell_execute`.
- **redirect_tool_call**: Use for MCP tools (like `tavily_search`), `python_execute` results, or any other non-shell tool output.

## Infinite Recursion Warning
The `redirect_tool_call` tool is protected against calling itself, but be careful not to create circular dependencies in your pipelines where two tools depend on each other's file outputs indefinitely.
