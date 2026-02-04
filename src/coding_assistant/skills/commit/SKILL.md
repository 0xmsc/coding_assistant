---
name: commit
description: Helps write clear commit messages and follow git best practices. Use when the agent needs to help with git commits or commit message writing.
---

# Commit Skill

## Workflow
1. **Analyze Changes**: Review `git status` and `git diff --staged`.
2. **Draft Message**: Write a short, clear summary (max 50 chars).
3. **Atomic Commits**: One logical change per commit.

## Template
```
<short summary of what changed>

[optional: why this change was needed]
```

## Safe Git Commands
- `git add <files>`
- `git commit -m "message"`
- `git diff --staged`
- `git status`, `git show`, `git log --oneline`

## Limitations
- No interactive commands (`git add -p`, `git rebase -i`).
- No force pushing or amending without explicit user guidance.
- Always confirm before final commit/push.
