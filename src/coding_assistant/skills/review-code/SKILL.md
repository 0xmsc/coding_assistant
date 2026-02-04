---
name: review-code
description: Provides a workflow for code reviews. Use when asked to review code or PRs.
---

# Code Review Skill

## Workflow
1. **Context**: Read PR description and identify scope.
2. **Review**: Focus on what matters (see priorities below).
3. **Verify**: Run linting and tests.
4. **Feedback**: Be constructive, explain rationale.

## Priorities
1. **PR Description**: Must be filled with clear context and purpose.
2. **Simplicity**: The simplest solution that works. No over-engineering.
3. **Clean Code**: Readable, well-named, no duplication.
4. **Security**: No hardcoded secrets, validate inputs, handle errors.
5. **Tests**: Appropriate coverage for the change.