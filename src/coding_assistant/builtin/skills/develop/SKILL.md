---
name: develop
description: General implementation and refactoring guidance for codebase work.
---

# Develop

- Use this skill for code exploration, bug fixes, refactors, and feature work.
- Read the relevant files before editing and prefer `rg` or `fd` for search.
- Favor targeted edits over rewriting entire files.
- Match the existing style and fail fast on unexpected states instead of silently recovering.
- Avoid speculative abstractions and backwards-compatibility layers unless the user asked for them.
- Verify changes with focused checks first, then broader project validation.
