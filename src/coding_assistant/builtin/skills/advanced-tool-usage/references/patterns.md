# Detailed Orchestration Patterns

## 1. Multi-Stage Pipeline Example

```markdown
1. **Call**: `redirect_tool_call(tool_name="some_search_tool", tool_args={"query": "..."}, output_file="/tmp/search_results.json")`
2. **Transform**: Use `python_execute` to parse `/tmp/search_results.json` and extract only the relevant items.
3. **Report**: Summarize the extracted items directly in the response.
```

## 2. Choosing the Right Mechanism

- **redirect_tool_call**: Primarily intended for outputs returned through tools that produce structured or text data but lack their own file-output parameters.
- **shell_execute redirection (`>`)**: Always prefer native shell redirection when already running bash commands.
- **python_execute file writing**: Always prefer native `Path.write_text()` when executing Python scripts.
