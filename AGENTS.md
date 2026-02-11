# AGENTS.md

This file captures repository-specific rules for contributors and coding agents.

## Non-Negotiables

- Run `just lint` and `just test` before committing.
- Keep commits focused and atomic.
- Do not bypass actor boundaries with direct cross-actor references.

## Architecture Snapshot

- Core runtime is actor-based under `src/coding_assistant/framework/actors/`.
- Actor messaging contracts live in `src/coding_assistant/framework/actors/common/messages.py`.
- Actor execution loop is in `src/coding_assistant/framework/actor_runtime.py`.
- Actor URI routing is handled by `src/coding_assistant/framework/actor_directory.py`.
- Session wiring and actor registration happen in `src/coding_assistant/session.py`.

## Actor Messaging Rules

- Actor-to-actor calls must use URI addressing via `ActorDirectory`.
- Treat URIs as explicit capabilities: only send to actors whose URI was provided/configured.
- Keep request/response correlation explicit with `request_id`.
- Prefer fail-fast behavior for unresolved URIs or missing runtime dependencies.
- Keep message dataclasses strongly typed and minimal.

## Public API Compatibility

- Preserve user-facing CLI/session behavior unless explicitly changing product behavior.
- Compatibility shims are acceptable at outer boundaries (for example UI-facing APIs), but avoid internal actor coupling.

## Testing Workflow

- Primary checks:
  - `just lint`
  - `just test`
- Add or update tests when changing actor protocols, runtime wiring, or session orchestration.
- Reuse shared helpers in `src/coding_assistant/framework/tests/helpers.py`.

## Editing Guidelines

- Prefer targeted edits over broad rewrites.
- Keep changes local to the relevant layer:
  - Message schema changes: `actors/common/messages.py`
  - Routing/wiring changes: `session.py`, `actor_directory.py`
  - Actor logic changes: corresponding actor module
- Update tests in the same change when behavior or contracts change.

## Quick File Map

- `src/coding_assistant/session.py`: top-level lifecycle and actor composition
- `src/coding_assistant/framework/actor_directory.py`: URI-to-actor registry
- `src/coding_assistant/framework/actors/agent/actor.py`: orchestration logic
- `src/coding_assistant/framework/actors/tool_call/actor.py`: tool call dispatch/cancellation
- `src/coding_assistant/framework/actors/llm/actor.py`: model completion actor
- `src/coding_assistant/framework/actors/user/actor.py`: UI/user interaction actor
- `src/coding_assistant/tools/tools.py`: high-level tools including nested agent launching
