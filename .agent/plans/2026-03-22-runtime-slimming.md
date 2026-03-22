# Runtime Slimming

## Objective

- [ ] Reduce `src/coding_assistant/runtime/` to a minimal conversation kernel.
- [ ] Remove persistence, transcript-rewrite helpers, and compaction policy from the runtime layer.
- [ ] Prefer deleting or temporarily dropping features over keeping policy-heavy behavior inside `runtime/`.

## Approach

- [ ] Treat `AssistantSession` as the only real runtime primitive: a state machine that owns in-memory conversation state, model turns, and tool-result handoff.
- [ ] Move storage and compaction concerns upward into `ManagedSession` or a small non-runtime controller layer.
- [ ] If a feature cannot be extracted cleanly in this pass, remove it and leave follow-up reintroduction for later.

## Scope

- [ ] In scope:
  - Remove `HistoryStore` and `FileHistoryStore` from `runtime/` exports and from `AssistantSession`.
  - Remove `SessionOptions` and runtime-owned `compact_conversation_at_tokens`.
  - Remove `compact_history()` and `runtime/history.py`.
  - Move or inline trivial runtime helpers (`engine.py`, `tool_spec.py`) where appropriate.
  - Update managed/session-driving code, tests, and docs to match the slimmer boundary.
- [ ] Out of scope:
  - Redesigning CLI UX beyond what is needed to preserve current behavior.
  - Reworking MCP tool implementations.
  - Preserving backwards compatibility for every current `runtime` export if it materially complicates the simplification.
- [ ] Allowed simplifications:
  - Drop automatic token-threshold compaction in this pass.
  - Drop runtime-specific transcript sanitization if it belongs more naturally to persistence code outside `runtime/`.
  - Accept a smaller public surface for `coding_assistant.runtime` if that keeps the boundary coherent.
- [ ] Explicitly preserve:
  - Support for multiple tool calls emitted in a single model step.

## Steps

- [ ] Define the target runtime contract.
  - `runtime/` should ideally contain only `session.py`, `events.py`, and a minimal `__init__.py`.
  - `AssistantSession` should accept plain tool definitions rather than runtime-specific wrapper types if feasible.
  - Runtime methods should cover: `start`, `send_user_message`, `submit_tool_result`, `submit_tool_error`, `cancel`, `next_event`, and `events`.
- [ ] Remove persistence from the runtime layer.
  - Move `HistoryStore`, `FileHistoryStore`, and load/save helpers out of `runtime/`.
  - Move persistence checkpoints from `AssistantSession` into `ManagedSession` or a dedicated driver/controller.
  - Update CLI/default session wiring to load and save history outside the runtime.
- [ ] Remove compaction from the runtime layer.
  - Delete runtime-owned compaction threshold handling.
  - Move `compact_conversation` transcript rewriting into `ManagedSession` or another non-runtime helper.
  - Decide whether automatic compaction is dropped entirely for this pass or re-homed immediately in the managed layer.
- [ ] Simplify runtime implementation details.
  - Inline or relocate `complete_single_step`, `parse_tool_call_arguments`, and related helpers if a standalone `engine.py` is no longer justified.
  - Remove `ToolSpec` if the session can depend on the existing `ToolDefinition` protocol directly.
  - Reassess whether streaming delta emission stays in the runtime or becomes optional follow-up scope.
  - Keep queued handling for multiple tool calls in one assistant turn even if other runtime features are removed.
- [ ] Update public surfaces and tests.
  - Adjust `coding_assistant.__init__`, `runtime.__init__`, and dependent imports.
  - Rewrite tests so core runtime tests cover only the state machine, while persistence/compaction tests target the managed or controller layer.
  - Update README/API examples to reflect the slimmer boundary.

## Verification

- [ ] Runtime unit tests cover the reduced `AssistantSession` contract without persistence or compaction assumptions.
- [ ] Managed-layer tests cover any remaining persistence and compaction behavior that moved out of `runtime/`.
- [ ] `just test` passes.
- [ ] `just lint` passes.

## Risks

- [ ] Public API breakage from removing `runtime` exports such as `SessionOptions`, `ToolSpec`, and `FileHistoryStore`.
- [ ] Resume/history behavior can regress when save/load checkpoints move out of the session lifecycle.
- [ ] Compaction may need to be deliberately disabled for one pass rather than preserved immediately if keeping it blocks a clean runtime boundary.
