---
name: plan
description: Guidelines for iteratively planning tasks and changes before implementation. Use this when the user requests a non-trivial task or when you need to align on a complex implementation strategy.
---

# Task Planning

## Workflow
1. **Gather Information**: Explore codebase and list dependencies.
2. **Draft Plan**:
   - **Objective**: Goal of the task.
   - **Approach**: High-level strategy.
   - **Steps**: Sequential actions.
   - **Verification**: How to prove it works.
3. **Clarify**: Include **at least 3 clarifying questions** with proposed answers.
4. **Approve**: Wait for user "go ahead" (empty response = approval).
5. **Implement**: Execute only after approval.

## Principles

### Zero-Impact
Do not modify files during planning.

### Iterative
Avoid big design up front. Plan a manageable milestone, get approval, implement, then plan the next milestone.

### Least Surprise
The plan acts as a contract. If you deviate during implementation, stop and explain why.

### Proactive Questioning
Search for flaws yourself. Present questions with proposed answers to reduce user cognitive load.
- "What happens if this network call fails?"
- "Is this the correct naming convention?"
- "Are there existing utilities that do this?"

### Environment Awareness
Always start with exploration. Understand the landscape before proposing a blueprint.

### Risk Mitigation
Identify dangerous parts (shared libraries, bulk deletions) and highlight them. Propose safe verification steps.
