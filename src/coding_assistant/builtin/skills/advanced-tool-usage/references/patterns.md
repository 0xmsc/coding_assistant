# Advanced Orchestration Patterns

Using the `redirect_tool_call` meta-tool enables sophisticated workflows that were previously impossible due to context window limits.

## The Search-then-Process Pipeline

Commonly used with MCP-backed search or fetch tools:

1. **Call**: `redirect_tool_call(tool_name="mcp_call", tool_args={"server": "my-server", "tool": "search", "arguments": {"query": "..."}}, output_file="search_raw.json")`
2. **Discover first**: Use `mcp_list_tools(server="my-server")` to find the actual MCP tool name and argument schema.
3. **Analysis**: Use `python_execute` to parse the JSON and extract specific URLs or snippets.
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

- **Direct File Writing**: Prefer shell redirection from `shell_execute` or `open().write()` in `python_execute`. This avoids the tool-call output buffer.
- **redirect_tool_call**: Primarily intended for outputs returned through tools like `mcp_call` that produce structured or text data but lack their own file-output parameters.

## Infinite Recursion Warning
The `redirect_tool_call` tool is protected against calling itself, but be careful not to create circular dependencies in your pipelines where two tools depend on each other's file outputs indefinitely.
