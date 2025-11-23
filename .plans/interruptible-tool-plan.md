# Interruptible tool execution revamp plan

## Context
- Users can send interrupts. Today this only toggles a flag (InterruptibleSection.was_interrupted) in the event loop.
- MCP tools keep running even after the user interrupts.
- Goal: when a user interrupts, all running tool calls stop immediately and the agent sees a tool result explaining the interruption.
- We need to record "User interrupted the tool call" (or similar) as the tool result so history reflects it.

## Constraints & open questions
- Need to detect interrupts from UI (PromptToolkit) and propagate to run loop.
- Must cancel asyncio tasks (tool call coroutines) and ensure underlying processes (e.g., shell, python) terminate.
- Need a consistent message format for interrupted tool results.
- Consider both orchestrator loop and chat loop.
- Need to update callback/telemetry handling so they know the tool was interrupted.
- Tests must cover interrupts for single tool call, multiple parallel tool calls, and tool queueing.
- Understand how MCP client handles cancellations—does `ClientSession.call_tool` support cancellation cleanly? Might need server changes.

## Implementation plan
1. **Signal propagation & orchestration**
   - Audit where interrupts are captured: `InterruptibleSection` wraps run loops today.
   - Introduce an explicit `InterruptController` (shared state) that exposes:
     - `request_interrupt()` when user hits Ctrl+C or /exit (depending on context).
     - `register_task(task, on_cancel)` to track running tool execution tasks and their cleanup callbacks.
     - `cancel_all(reason="user_interrupt")` which cancels tracked tasks and awaits their completion.
   - Modify `run_agent_loop` and `run_chat_loop` to enter an `InterruptibleSection`, and upon interrupt, call the controller to cancel outstanding tool calls and inject interruption messages into history.

2. **Tool-level cancellation hooks**
   - Extend `Tool` base class to optionally expose `async def cancel(self):` or `def cancel(self):` for cleanup.
   - When wrapping MCP tools, implement cancellation by aborting the underlying `ClientSession.call_tool` future and issuing a kill signal to the MCP sub-process if supported (investigate `fastmcp` API for cancellation; may require new functionality to terminate running command/process via `ClientSession.cancel()` or by closing transport).
   - For built-in tools (FinishTaskTool, ShortenConversation, LaunchAgent), cancellation likely trivial (no external process). For `mcp_coding_assistant_mcp_shell_execute` and python execution, cancellation must send `Process.kill()`.

3. **Execute-tool pipeline updates**
   - When scheduling tool calls in `handle_tool_calls`, register each `asyncio.Task` with the InterruptController.
   - Wrap each tool execution in try/finally to deregister.
   - On cancellation, ensure `execute_tool_call` raises a dedicated `ToolInterruptedError`.
   - Catch `ToolInterruptedError` in `handle_tool_call` and append a tool message with standardized content, e.g., "Tool execution interrupted by user." Also set tool callback responses accordingly.

4. **Recording tool results**
   - Define a helper `append_interrupted_tool_message(tool_call, reason)` that appends `"User interrupted the tool call."` to history and to callbacks/traces.
   - Ensure tokens/telemetry capture this state.

6. **User interface / UX**
   - Clarify how a user issues interrupts in chat mode vs orchestrator (Ctrl+C vs /exit). Document in README/instructions.
   - If interrupt happens while awaiting user input, we probably exit the loop gracefully.

7. **Testing strategy**
   - Unit tests for InterruptController: registering tasks, cancellation order, repeated interrupts.
   - Integration tests in `agents/tests/test_run_loop_slices.py` covering:
     - Single tool call interrupted mid-flight.
     - Multiple parallel tool calls: ensure all tasks get cancellation message.
     - Tool return after cancellation should not append extra message.
   - Tests for MCP shell tool verifying subprocess killed (maybe by running `sleep` and ensuring command ends quickly when cancelled).

8. **Documentation**
   - Update `.plans/`? (this doc) and README to explain new interrupt behavior.
   - Mention new API surface for tools (cancel hooks) and instructions for MCP contributions.

## Risks & mitigations
- **Protocol limitations**: MCP might not yet support cancellation signals. If true, we may need to wrap shell/python execution locally to keep process handles.
- **Task cancellation races**: need to avoid leaving `asyncio.CancelledError` bubbling to agent loop and crashing. Handle gracefully.
- **User confusion**: On interrupt, agent should clearly state tool was interrupted and prompt for next action.

## Next steps
1. Prototype InterruptController (maybe in `coding_assistant/agents/interrupts.py`).
2. Add cancellation hooks to Tool interface and built-in tools.
3. Modify MCP tool wrappers & servers to surface cancellable handles.
4. Update `handle_tool_calls` to integrate controller.
5. Implement history logging for interruptions.
6. Add tests.
7. Document behavior.

## Detailed design notes
### Interrupt controller API
- Keep `ToolCallCancellationManager` minimal and layer a higher-level `InterruptController` over it. The controller owns:
  - `register_tool_task(call_id: str, task: Task, cancel_cb: Callable[[], Awaitable[None]] | None)`
  - `cancel(reason: InterruptReason)` which simultaneously cancels tasks via `task.cancel()` and awaits any explicit cleanup callbacks.
  - `consume_interrupts()` helper so loops know whether to inject "interrupted" user guidance after cancellation.
- Controller should be shared between agent/chat loops, so expose a factory (`create_interrupt_controller(loop)`).
- Make sure repeated interrupts do not keep growing the tracked-task set; cancellation should be idempotent and tolerate tasks that already finished.

### Tool cancellation lifecycle
1. `handle_tool_calls` schedules `execute_tool_call` for each tool. Before awaiting it, register the underlying `asyncio.Task` with the controller.
2. When SIGINT fires, `InterruptibleSection` invokes `controller.cancel(reason="user_interrupt")`.
3. Cancellation propagates into `execute_tool_call`, which should translate `asyncio.CancelledError` into a `ToolInterruptedError` so the rest of the pipeline can branch on it.
4. `handle_tool_call` catches `ToolInterruptedError`, appends a "tool interrupted" history item, calls `tool_callbacks.on_tool_error(...)` with a structured payload, and returns early without invoking result handlers.
5. The UI should receive a lightweight toast/log entry explaining that tools were cancelled and that the agent can decide what to do next.

### History + callback updates
- Introduce `InterruptedToolResult(TextResult)`? Maybe overkill; instead append a tool message whose `content` is `"Tool execution interrupted by user (Ctrl+C)."`.
- Ensure `append_tool_message` is always invoked exactly once per tool call, even for interrupts, so downstream auditing does not see missing entries.
- `AgentToolCallbacks` should gain an explicit `on_tool_interrupted` hook so telemetry can distinguish failures vs user-driven interrupts.
- When interrupts occur during queued tool calls (i.e., LLM returned multiple tool calls and we process them sequentially), ensure that we still emit interruption messages for every call that never started (maybe mark them as `"skipped due to earlier interrupt"`).

### Telemetry & observability
- Add OpenTelemetry span attributes `tool.interrupted=true`, `tool.interrupt_reason=user` when a tool gets cancelled.
- Emit structured log entries at INFO level whenever a cancellation request is issued and when each tool actually stops. Include duration between start and cancellation for troubleshooting.
- Consider a metric counter `tool_interrupts_total` labelled by tool name and reason.
- Capture whether cancellation succeeded within a timeout; if not, surface a warning so we can detect hung subprocesses.

### MCP shell/python specifics
- Wrap the existing MCP bridge so that each call retains a handle to the spawned subprocess PID. On cancellation, send SIGTERM, wait ~1s, then SIGKILL if necessary.
- For Python execution, rely on `asyncio.subprocess.Process`. We must ensure stdout/stderr pipes are closed to avoid deadlocks when the process is killed mid-stream.
- If MCP servers do not yet support cancellation, document the limitation and optionally fall back to best-effort (mark tool as interrupted even if underlying process keeps running, but at least stop awaiting it).

### UI/UX copy
- Standardize user-facing copy: `"Received interrupt. Cancelling all running tools..."` followed by per-tool updates.
- After cancellation, append a system-style reminder encouraging the user/agent to decide whether to retry, choose another approach, or finish.
- For chat mode, if the user hits Ctrl+C while the agent is waiting for input, interpret it as "abort session" rather than "cancel tools" and exit gracefully.

## Test matrix (draft)
| Scenario | Setup | Expected behavior |
| --- | --- | --- |
| Single tool interrupted | Tool sleeps for 5s; send interrupt after 1s | Tool task cancelled, history contains interruption message, no crash |
| Parallel tools | Two long-running MCP calls started via multi-call message | Both tasks cancelled, both record interruption entries, controller clears state |
| Sequential queue | Three tool calls, interrupt during first | First marked interrupted, remaining two skipped with "not executed due to interrupt" note |
| Callback prevention | `before_tool_execution` returns result immediately, then interrupt occurs | No cancellation triggered because no running tasks |
| Chat loop waiting for user | Agent idle requesting input, user sends Ctrl+C | Session exits without stack trace |
| Non-interruptible mode | `run_chat_loop(..., is_interruptible=False)` | Ctrl+C ignored until section ends |
| Failed cancellation | Simulate tool that ignores cancel | Controller logs warning after timeout |

## Follow-up questions / items
- Do we need back-pressure to prevent new tool calls from starting while cancellation is in-flight?
- Should interrupts propagate to child agents (launched via `launch_agent` tool), and if so, how do we link controllers across agent boundaries?
- Are there UI affordances for resuming a cancelled tool, or should the agent explicitly re-issue the tool call if desired?
- Consider persisting interrupt events to run transcripts so that future debugging clearly shows user intent.
