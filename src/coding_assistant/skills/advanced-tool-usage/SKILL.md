---
name: advanced-tool-usage
description: Guidelines for multi-stage tool orchestration and handling large data using 'redirect_tool_call'. Use this when you need to process large amounts of data without exhausting the context window or when building complex data pipelines.
---

# Advanced Tool Usage

## Core Principles
- **Context Economy**: Never bring raw, voluminous data into the conversation if you only need a refined subset.
- **Pipeline Thinking**: View tools as modular blocks that can pass data through files.
- **Offloading**: Use `redirect_tool_call` to "capture" output into external storage.

## Patterns

### 1. The Pipelining Pattern
When a tool's output is the input for another tool:
1.  **Redirect**: Call the first tool using `redirect_tool_call`.
2.  **Process**: Call the second tool (e.g., `python_execute` or `shell_execute`) and pass the file path created in step 1 as an argument.
3.  **Refine**: Read only the final processed result into the conversation.

### 2. The Context Buffer Pattern
When working with large files or long logs:
- Redirect the reading tool (e.g., `cat`, `tavily_search`) to a temporary file.
- Use `rg` or `grep` to extract only the relevant lines from that file.

### 4. Workspace Management for Pipelines
When building multi-stage pipelines that generate multiple files:
- Use `shell_execute` with `mktemp -d` to create a dedicated scratch directory.
- Direct all intermediate `redirect_tool_call` outputs into that directory to keep the workspace clean.
- Example: `redirect_tool_call(..., output_file="/tmp/tmp.X/step1.json")`

### 3. The Large Data Export
When the user requests a result that is too large for markdown (e.g., a 5MB JSON dump):
- Use `redirect_tool_call` with a specific `output_file` name.
- Inform the user of the file location instead of printing the content.

## When to use `redirect_tool_call`
- The expected output is > 50 lines.
- The output is raw data (JSON, CSV, long logs) that needs further processing.
- You are chaining multiple tools together.

## References
- [Detailed Orchestration Patterns](references/patterns.md)
