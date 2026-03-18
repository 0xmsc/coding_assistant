---
name: develop
description:
  General principles for exploring, developing, editing, and refactoring code.
  Use for codebase analysis, implementation tasks, and code quality
  improvements.
---

# Develop Skill

## Core Principles

- **Clean Code**: Follow readable, maintainable patterns.
- **Fail Early**: Assert or throw on unexpected conditions. Do not be
  defensive - if something unexpected happens (e.g., invalid JSON that should be
  valid), crash immediately rather than silently handling it.
- **Simplicity**: Prefer the simplest working solution.

## Repo Context

- When repo-specific capability skills are available, load the relevant ones and
  consult only the needed `references/` before implementation.
- Treat those repo-specific skill references as durable repo knowledge.
- Use `.agent/` for local plans, todos, and scratch state rather than scattering
  task state across tool-specific paths.
- If implementation reveals a stable, reusable repo-specific learning, promote
  it into the appropriate capability skill or reference file instead of leaving
  it only in `.agent/`.

## Planning

For non-trivial changes:

- Use `brainstorm` first when the objective, approach, or scope is still
  ambiguous.
- Ask clarifying questions during planning when needed to unblock the work, and
  do not treat them as a required section of the written plan.
- **Wait for explicit approval** before implementation.
- For complex strategy, use the `plan` skill.
- During execution, use the `todo` skill to track the approved work.

## Implementation & Git

### Git Safety

- **Permission First**: Always ask before initializing repos or switching
  branches.
- **Atomic Commits**: Group related changes logically.

### Code Quality

- **Focus on WHY**: Document intent, not mechanics.
- **Edge Cases**: Consider and test error scenarios.
- **Existing Style**: Match existing patterns and formatting.
- **No Abbreviations**: Use full words in identifiers unless the abbreviation is
  a widely accepted standard acronym in the domain (e.g., JPEG, PNG). E.g.,
  `Message` not `Msg`, `Number` not `Num`, `FileDescriptorSet` not `Fds`.

## Language-Specific References

When working with specific languages, consult the following references:

- **C++**: Refer to `references/cpp.md`.
- **Python**: Refer to `references/python.md`.
- **Rust**: Refer to `references/rust.md`.

## File Operations

- **Prefer targeted edits**: Use `apply_patch` (or equivalent targeted edit
  tools) instead of rewriting full files.
- **Verification**: Always verify changes after editing.
- **Tools**: Use `fd` and `rg` for exploration; standard shell tools for
  navigation.
- **Reading**: Use shell tools like `cat`, `sed`, etc. for reading file
  contents.
