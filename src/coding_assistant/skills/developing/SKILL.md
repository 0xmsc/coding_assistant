---
name: developing
description: General principles for exploring, developing, editing, and refactoring code. Use for codebase analysis, implementation tasks, and code quality improvements.
---

# Developing Skill

This skill provides general principles and workflows for working with codebases. It covers exploration, development, editing, and refactoring tasks.

## When to use this skill
Use this skill when you need to:
- Explore or understand an existing codebase
- Implement new features or modify existing code
- Refactor code for better quality
- Perform code reviews
- Work with git repositories
- Apply clean code principles

## Core Principles

- **Follow clean code principles**: Write readable, maintainable code
- **Ensure proper error handling**: Don't let unexpected errors go unhandled
- **Prefer simplicity over complexity**: Choose the simplest solution that works
- **Most tasks are in repositories**: Work within existing repos unless specifically instructed otherwise

## Planning

For non-trivial changes:
- Present a coherent plan with **at least 3 clarifying questions** (with your proposed answers)
- **Wait for explicit approval** before starting implementation
- Update the plan when receiving feedback

**For complex planning tasks, use the `planning` skill** which provides detailed guidance on iterative planning, risk mitigation, and zero-impact planning workflows.

## Development Practices

### Code Quality
- **Documentation**: Document WHY, not WHAT. The code shows what it does.
- **Comments**: Avoid obvious, redundant comments
- **Functions/Classes**: Don't document trivial functions or classes
- **Error Handling**: **Fail early, never silently**. Only catch exceptions you expect and can handle. Do NOT catch unexpected exceptions just to ignore them - let them bubble up and cause the program to fail visibly

### Implementation Guidelines
- Follow existing code style and patterns in the project
- Write clear, self-documenting code
- Consider edge cases and error scenarios
- Test your changes as you go

## Git Workflow

### Important: Always ask permission first
- **Never** initialize a new git repository without asking
- **Never** commit changes without explicit approval
- **Never** switch branches without asking
- **Always** confirm before pushing changes

### When working with git
- Check the current repository status
- Understand the branch structure
- Review existing commit history if needed
- Plan your commits logically and explain them clearly when requesting approval

## Codebase Exploration

### Initial setup
- Use `pwd` to determine your current location
- Verify you're in the correct project directory

### Discovery tools
- Use `fd` (fast file finder) to locate files and directories
- Use `rg` (ripgrep) to search content within files
- Use `ls`, `find`, and other shell tools to navigate
- Review README files, documentation, and existing tests for context

## File Editing

### Preferred methods
- **Use `edit_file`**: For modifying existing files (line-by-line changes)
- **Use `sed`**: For search and replace operations (e.g., renaming variables)
- **Use `cp` & `mv`**: For copying and moving files (don't memorize contents)

### Avoid
- **Do NOT use `applypatch`**: It's not available/reliable
- **Avoid writing full files**: Use `edit_file` instead when possible
- **Don't memorize file contents**: Use proper file operations

### Best practices
- Make small, focused changes
- Verify changes after editing
- Use appropriate tools for the job
