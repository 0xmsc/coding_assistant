# Module Simplification Plan

## Objective

Reduce the production module surface area under YAGNI by removing unused/speculative code, deleting sub-agent support, and folding thin glue modules into their nearest real owners while preserving current non-sub-agent behavior.

## Scope

In scope:

- Remove sub-agent and expert-model support from the runtime and CLI surface.
- Delete dead production modules that are not on the live runtime path.
- Merge thin wrapper modules into nearby modules that already own the behavior.
- Flatten the CLI adapter package into a single top-level module.
- Update tests, imports, and docs to match the simplified layout.

Out of scope:

- Event-generator refactors.
- Changing MCP tool behavior beyond import/module relocation.
- Reworking the built-in MCP server package structure.
- Adding compatibility shims for removed APIs.

## Success Criteria

- `run_agent(...)` only advances one transcript through model/tool turns until the assistant yields or the run fails.
- No production code path supports launching child agents or selecting an expert model.
- Unused modules `parameters.py` and `interrupts.py` are removed rather than kept as dormant abstractions.
- Thin modules `agent_types.py`, `config.py`, `defaults.py`, and `llm/adapters.py` no longer exist as separate files.
- CLI behavior is preserved for interactive and single-shot runs, minus removed sub-agent/expert-model surface.
- `just test` and `just lint` are green.

## Target Layout

Keep:

- `src/coding_assistant/agent.py`
- `src/coding_assistant/cli.py`
- `src/coding_assistant/history.py`
- `src/coding_assistant/history_store.py`
- `src/coding_assistant/instructions.py`
- `src/coding_assistant/ui.py`
- `src/coding_assistant/image.py`
- `src/coding_assistant/paths.py`
- `src/coding_assistant/trace.py`
- `src/coding_assistant/sandbox.py`
- `src/coding_assistant/tools/builtin.py`
- `src/coding_assistant/tools/mcp.py`
- `src/coding_assistant/llm/types.py`
- `src/coding_assistant/llm/openai.py`
- `src/coding_assistant/mcp/*`

Merge:

- `src/coding_assistant/agent_types.py` -> `src/coding_assistant/agent.py`
- `src/coding_assistant/config.py` -> `src/coding_assistant/tools/mcp.py`
- `src/coding_assistant/defaults.py` -> `src/coding_assistant/cli.py`
- `src/coding_assistant/llm/adapters.py` -> `src/coding_assistant/llm/openai.py`
- `src/coding_assistant/adapters/cli.py` -> `src/coding_assistant/cli.py`

Delete:

- `src/coding_assistant/agent_types.py`
- `src/coding_assistant/config.py`
- `src/coding_assistant/defaults.py`
- `src/coding_assistant/llm/adapters.py`
- `src/coding_assistant/parameters.py`
- `src/coding_assistant/interrupts.py`
- `src/coding_assistant/adapters/cli.py`
- `src/coding_assistant/adapters/__init__.py`

Also remove unused code after relocation:

- `LaunchAgentSchema`
- `LaunchAgentTool`
- `expert_model`
- recursive child-run helpers in `agent.py`
- `NullUI` if it remains unused after import cleanup

## Steps

- [x] Step 1: Simplify the agent runtime.
  - Remove sub-agent support from `src/coding_assistant/agent.py`.
  - Remove `expert_model` from `run_agent(...)` and all callers.
  - Move `AgentRunResult` into `src/coding_assistant/agent.py`.
  - Keep built-in compaction and redirected tool output support if still used.

- [x] Step 2: Shrink built-in tool surface.
  - Remove `LaunchAgentSchema` and `LaunchAgentTool` from `src/coding_assistant/tools/builtin.py`.
  - Update tool construction in `src/coding_assistant/agent.py` to build only the remaining built-ins.
  - Remove sub-agent-specific tests and add/adjust tests for the simplified tool set if needed.

- [x] Step 3: Flatten CLI composition glue.
  - Create `src/coding_assistant/cli.py` by folding in logic from `src/coding_assistant/adapters/cli.py` and `src/coding_assistant/defaults.py`.
  - Update `src/coding_assistant/main.py` to import `run_cli` from `src/coding_assistant/cli.py`.
  - Delete the `adapters` package once imports and tests are updated.

- [x] Step 4: Collapse thin helper modules into their owners.
  - Move `MCPServerConfig` into `src/coding_assistant/tools/mcp.py`.
  - Move the input-schema adaptation helper from `src/coding_assistant/llm/adapters.py` into `src/coding_assistant/llm/openai.py`.
  - Remove now-dead constructor arguments and fields such as unused `root_directory`/`coding_assistant_root` plumbing if they become redundant after the merge.

- [x] Step 5: Remove dead code.
  - Delete `src/coding_assistant/parameters.py` and its tests.
  - Delete `src/coding_assistant/interrupts.py` and its tests.
  - Remove `NullUI` if still unused after the CLI refactor.
  - Remove small dead fields like `MCPWrappedTool._server_name` if still unused.

- [x] Step 6: Update docs and packaging references.
  - Remove `launch sub-agents` and `--expert-model` from `README.md`.
  - Replace “chat mode vs agent mode” wording with interactive vs non-interactive CLI wording.
  - Update imports/examples to the new `cli.py` module path where relevant.

- [x] Step 7: Verify.
  - Run `just test`.
  - Run `just lint`.
  - Sanity-check the CLI help text and README examples against the simplified flags and module names.

## Risks

- Import churn can break tests and entrypoints if the CLI flattening is done before all references are updated.
- Removing sub-agent support changes a documented feature, so README and help text must be updated in the same change.
- Deleting dead modules will remove tests with them; verification must stay focused on live behavior, not preserving coverage numbers.

## Recommended Execution Order

1. Simplify `agent.py` and `tools/builtin.py`.
2. Remove `expert_model` from CLI and docs.
3. Merge `defaults.py` and `adapters/cli.py` into `cli.py`.
4. Merge `config.py` and `llm/adapters.py` into their owners.
5. Delete dead modules and their tests.
6. Run verification and fix fallout.

## Outcome

Implemented end to end. Verification passed with `just test` and `just lint`.
