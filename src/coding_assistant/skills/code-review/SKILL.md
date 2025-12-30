---
name: code-review
description: Provides a structured workflow for planning and executing code reviews like a senior engineer, including checklists for quality, security, and maintainability. Use when asked to review code, PRs, or plan a code review task.
---

# Code Review Skill

## Prerequisites
- Access to the codebase (repository, files, or PR)
- Understanding of the programming language(s) used
- Context about the change (purpose, requirements, constraints)

## Workflow

### 1. Understand the Change
- Read the associated ticket, issue, or description
- Identify the scope: what files are changed, what functionality is added/modified
- Ask clarifying questions if the intent is unclear

### 2. Plan the Review
- Decide on review depth (quick vs. thorough)
- Identify key risk areas: security, performance, compatibility
- Check for any known patterns or anti-patterns in the codebase

### 3. Execute the Review (Checklist)
Use the checklist from `references/checklist.md` or the summary below:

#### Code Quality
- **Readability**: Is the code self-documenting? Clear naming?
- **Complexity**: Are functions small and focused? Avoid deep nesting.
- **Duplication**: Could any code be reused or refactored?
- **Testing**: Are there unit tests? Are edge cases covered?

#### Design & Architecture
- **SOLID principles**: Are they followed? (Single responsibility, etc.)
- **Design patterns**: Appropriate usage, not over-engineered?
- **Dependencies**: Are they managed, minimized, and justified?

#### Security
- **Input validation**: Sanitized? Protected against injection?
- **Authentication/Authorization**: Proper checks?
- **Data exposure**: Sensitive data logged or exposed?
- **Known vulnerabilities**: Use tools like `npm audit`, `safety check`, etc.

#### Performance
- **Efficiency**: Time/space complexity acceptable?
- **Database queries**: N+1 problems? Missing indexes?
- **Caching**: Appropriate use of caching?

#### Maintainability
- **Consistency**: Follows project style guides?
- **Error handling**: Graceful? Informative? Not silent?
- **Logging**: Adequate for debugging?

### 4. Use Tools
- Run linters: `just lint` or language-specific linters
- Run static analysis: `safety`, `bandit`, `semgrep`
- Run tests: `just test`


### 5. Document Findings
- Provide constructive feedback: explain why and suggest improvements
- Use line comments in the PR/code
- Summarize overall assessment: approve, request changes, or comment

### 6. Follow Up
- Track resolution of issues
- Verify fixes are correct
- Close the loop

## Examples

### Example 1: Review a Pull Request
**User**: "Review PR #42 for the new authentication endpoint"
**Agent**: 
1. Fetch PR diff (using `gh pr diff 42`)
2. Understand the changes: added JWT validation, new endpoint
3. Run `just lint` and `just test`
4. Use checklist from references
5. Report findings: missing input validation, error handling gaps

### Example 2: Plan a Review
**User**: "Plan a code review for the migration script"
**Agent**:
1. Ask for the script location and purpose
2. Identify risks: data integrity, rollback, performance
3. Suggest specific checks: dry-run, logging, backup

## Troubleshooting
- **No access**: Ask user to provide code snippet or grant access
- **Unclear context**: Request ticket/description
- **Tool failures**: Report error, suggest manual review
- **Large PR**: Break into smaller commits, review incrementally

## References
- `checklist.md`: Detailed code review checklist
- `security_patterns.md`: Common security patterns to look for