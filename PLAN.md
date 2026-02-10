# Cohesion-First Actor Refactor PLAN

## Goal
- Maximize cohesion by moving actor-specific logic into actor folders.
- Keep only truly shared logic in common/shared modules.
- Preserve behavior with compatibility facades during migration.

## Tracking
- [x] Phase 1 complete
- [x] Phase 2 complete
- [x] Phase 3 complete
- [x] Phase 4 complete
- [ ] Phase 5 complete
- [ ] Phase 6 complete
- [ ] Final cleanup complete

## Runtime handoff protocol (must hold)
- [x] Add explicit handoff messages in `src/coding_assistant/framework/actors/common/messages.py`:
- [ ] `RunAgentRequest` (User/Orchestrator -> Agent)
- [x] `AgentYieldedToUser` (Agent -> User) for "no tool calls" yield
- [x] `UserTextSubmitted`, `ClearHistoryRequested`, `CompactionRequested`, `ImageAttachRequested`, `SessionExitRequested` (User -> Agent)
- [x] Ensure `AgentActor` never directly calls prompt/ask on `UserActor`; handoff is message-driven only
- [x] Ensure user input resumes only after receiving `AgentYieldedToUser` (or at session start)
- [x] Keep request IDs on messages where correlation is required
- [x] Run `just lint`
- [x] Run `just test`

## Phase 1: User input intents and domain messages
- [x] Add chat intent/domain message types in `src/coding_assistant/framework/actors/common/messages.py`
- [x] Let `UserActor` parse slash commands and emit domain messages (not raw command strings)
- [x] Keep `UserActor` focused on input interpretation + transport only
- [x] Ensure `UserActor` does not mutate agent history/state directly
- [x] Keep cross-actor communication message-only via shared contracts
- [x] Run `just lint`
- [x] Run `just test`

## Phase 2: Agent consumes domain messages only
- [x] Remove command parsing/dispatch from `src/coding_assistant/framework/actors/agent/actor.py`
- [x] Handle only domain messages from `UserActor` (text, clear, compact, image, exit)
- [x] Keep AgentActor as state owner (history mutation and execution control)
- [x] Define chat execution rule: run agent steps continuously while tool calls exist; emit `AgentYieldedToUser` when a step has no tool calls
- [x] Move chat policy helpers into `src/coding_assistant/framework/actors/agent/chat_policy.py`
- [x] Move `_create_chat_start_message` from `src/coding_assistant/framework/chat.py`
- [x] Move `handle_tool_result_chat` from `src/coding_assistant/framework/chat.py`
- [x] Update `src/coding_assistant/framework/actors/agent/actor.py` imports
- [x] Keep `src/coding_assistant/framework/chat.py` as compatibility facade
- [x] Run `just lint`
- [x] Run `just test`

## Phase 3: Agent formatting + image helper
- [x] Create `src/coding_assistant/framework/actors/agent/formatting.py`
- [x] Move `format_parameters` from `src/coding_assistant/framework/parameters.py`
- [x] Create `src/coding_assistant/framework/actors/agent/image_io.py`
- [x] Move `get_image` from `src/coding_assistant/framework/image.py`
- [x] Update `src/coding_assistant/framework/actors/agent/actor.py` imports
- [x] Keep compatibility facades in original files
- [x] Run `just lint`
- [x] Run `just test`

## Phase 4: Tool-call executor colocation
- [x] Create `src/coding_assistant/framework/actors/tool_call/executor.py`
- [x] Move `ToolExecutor` from `src/coding_assistant/framework/tool_executor.py`
- [x] Update `src/coding_assistant/framework/actors/tool_call/actor.py` imports
- [x] Keep `src/coding_assistant/framework/tool_executor.py` as compatibility facade
- [x] Run `just lint`
- [x] Run `just test`

## Phase 5: User actor split from terminal UI
- [ ] Create `src/coding_assistant/framework/actors/user/actor.py`
- [ ] Move `ActorUI` from `src/coding_assistant/ui.py`
- [ ] Move `UserActor` from `src/coding_assistant/ui.py`
- [ ] Move actor message handling from `src/coding_assistant/ui.py`
- [ ] Keep terminal UI classes in `src/coding_assistant/ui.py` (`UI`, `PromptToolkitUI`, `DefaultAnswerUI`, `NullUI`)
- [ ] Re-export moved actor classes from `src/coding_assistant/ui.py` for compatibility
- [ ] Run `just lint`
- [ ] Run `just test`

## Phase 6: Import hygiene and anti-coupling checks
- [ ] Ensure `actors/agent` does not import `actors/llm`, `actors/tool_call`, `actors/user` directly
- [ ] Ensure `actors/llm` does not import `actors/agent`, `actors/tool_call`, `actors/user` directly
- [ ] Ensure `actors/tool_call` does not import `actors/agent`, `actors/llm`, `actors/user` directly
- [ ] Ensure `actors/user` does not import `actors/agent`, `actors/llm`, `actors/tool_call` directly
- [ ] Keep cross-actor communication via shared messages/contracts in `actors/common`
- [ ] Run `just lint`
- [ ] Run `just test`

## Final cleanup
- [ ] Remove no-longer-needed compatibility facades after all imports are migrated
- [ ] Remove dead helper code in old modules
- [ ] Update README architecture section to final module locations
- [ ] Verify no behavior regressions in actor integration tests
- [ ] Run `just lint`
- [ ] Run `just test`

## Shared modules to keep shared
- `src/coding_assistant/framework/actor_runtime.py`
- `src/coding_assistant/framework/history.py` (or later move to `actors/common/history_ops.py` if needed)
- `src/coding_assistant/framework/results.py`
- `src/coding_assistant/framework/types.py`
- `src/coding_assistant/llm/types.py`
- `src/coding_assistant/llm/openai.py`
