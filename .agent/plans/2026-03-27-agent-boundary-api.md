# Agent Boundary API Plan

## Objective

Add a lower-level agent API that returns explicit boundaries while keeping `run_agent(...)` as the existing convenience wrapper.

## Approach

- Introduce boundary result types for "awaiting user" and "awaiting tools".
- Add a low-level `run_agent_until_boundary(...)` function that consumes one model turn and returns one of those boundaries.
- Factor tool-call execution into a reusable helper that the high-level `run_agent(...)` wrapper can loop over.
- Update tests, exports, and README to describe both surfaces.

## Steps

- [x] Add boundary dataclasses and the low-level boundary-returning API in `src/coding_assistant/agent.py`.
- [x] Refactor `run_agent(...)` to wrap the boundary API plus shared tool-execution helper.
- [x] Export the new public API and update README examples/docs.
- [x] Add tests for the low-level boundary API and keep the high-level wrapper covered.
- [x] Verify with `just test` and `just lint`.

## Verification

- `just test`
- `just lint`
