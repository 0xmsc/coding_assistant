# AGENTS.md

- Make sure `just test` and `just lint` are green before committing.
- This repository is hosted on Forgejo. Use the `fj` CLI for pull requests and other Forgejo operations.

## Philosophy & Architecture

- **Minimal Core:** Keep `coding_assistant.core` and the main agent execution loop as simple, lean, and unopinionated as possible.
- **Skills-First Extensibility:** New workflows, capabilities, domain knowledge, and specialized tools should be implemented as **skills**, instructions, or injected tool plugins rather than adding complex logic into the core engine.
- **Extensible Primitives:** The core should only provide essential, orthogonal primitives (such as tool dispatch, streaming, message history manipulation, and cancellation). Higher-level policies (when to compact, how to plan, reasoning strategies) belong in skills and prompts.

## Tests

- Use `just test` to run the test suite.
- Use `just lint` to lint the codebase.
