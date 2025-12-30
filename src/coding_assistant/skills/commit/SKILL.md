---
name: commit
description: Helps write conventional commit messages, create atomic commits, and follow git best practices. Use when the agent needs to help with git commits, commit message writing, or git workflow guidance.
---

# Commit Skill

## Prerequisites
- Basic understanding of git
- Changes staged or ready to commit

## Agent-Appropriate Workflow

### Writing Commit Messages
When asked to help write or review a commit message:
1. **Analyze the changes** - What was modified and why?
2. **Determine the type** - feat, fix, docs, style, refactor, test, chore
3. **Write a concise subject** - Max 50 characters, imperative mood
4. **Add body (if needed)** - Explain "why" not "what", wrap at 72 chars
5. **Include footer (optional)** - References, breaking changes

### Guiding Commit Structure
For multi-change scenarios, provide guidance:
- **Separate logical changes** - Each commit should do one thing
- **Stage files by category** - Group related changes together
- **Create independent commits** - One commit per logical change
- **Order matters** - Related changes together, dependencies first

## Safe Git Commands for Agents
The agent can safely execute:
- `git add <files>` - Add specific files
- `git commit -m "message"` - Commit with message
- `git diff --staged` - Review staged changes
- `git status` - Check current status
- `git show` - View commit details
- `git log --oneline` - View commit history

## Templates

Use the provided template for writing consistent commit messages:
```bash
cat src/coding_assistant/skills/commit/assets/commit-template.txt
```

## Examples

### Example 1: Simple feature
```bash
git add .
git commit -m "feat: add logout functionality"
```

### Example 2: Complex change
```bash
git add src/auth/
git commit -m "feat: implement OAuth2 flow

Add Google and GitHub OAuth providers.
Update login page with social buttons.
Store provider tokens securely.

BREAKING CHANGE: Auth config format changed"
```

### Example 3: Bug fix
```bash
git add src/utils/validation.js
git commit -m "fix: handle empty string in email validation

Add trim() before validation to prevent false negatives
for whitespace-only inputs."
```

## Best Practices

### Do's
✅ Use imperative mood in subject line ("add" not "added")
✅ Separate subject from body with blank line
✅ Limit subject to 50 characters
✅ Wrap body at 72 characters
✅ Use the body to explain "why" not "what"
✅ Reference issues at bottom: "Fixes #123"

### Don'ts
❌ Don't end subject with period
❌ Don't use capitalization excessively
❌ Don't use "and" to connect multiple changes
❌ Don't commit commented-out code
❌ Don't include trailing whitespace

## Agent Limitations
The agent cannot execute interactive git commands. For operations like:
- Interactive staging (`git add -p`)
- Commit amending (`git commit --amend`)
- Interactive rebasing (`git rebase -i`)
- Force pushing (`git push --force-with-lease`)

The agent can only provide guidance. These operations must be performed manually.

## Resources

### References
- **conventional-commits.md**: Full specification and examples
- See `references/` directory for detailed documentation

## Conventional Commits Reference
This skill follows the Conventional Commits specification:
- https://www.conventionalcommits.org/

For detailed examples and edge cases, see the reference documents.
